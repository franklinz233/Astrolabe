from typing import List, Optional, Tuple
import re
import torch
import torch.distributed as dist

from pipeline.causal_inference import CausalInferencePipeline
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation


class SceneCausalInferencePipeline(CausalInferencePipeline):
    """CausalInferencePipeline with multi-scene prompt switching via '|' separator.

    Prompt format: "prompt1[5s] | prompt2[15s#] | prompt3"
      - [Xs]  specifies scene duration in seconds
      - #     marks a hard scene cut at that boundary
      - No duration → uses default_blocks_per_scene
    """

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.default_fps = getattr(args, "fps", 16)
        self.default_blocks_per_scene = 14  # 14 blocks = 10.5s at 16 fps

    # ------------------------------------------------------------------
    # Scene parsing
    # ------------------------------------------------------------------
    def _parse_scene_durations(self, prompt: str) -> Tuple[List[str], List[int], List[bool]]:
        """Parse scene prompts with optional durations and scene cut indicators.

        Returns:
            (prompt_texts, block_counts_per_scene, scene_cut_flags)
        """
        scene_parts = [part.strip() for part in prompt.split('|')]
        prompt_texts = []
        block_counts = []
        scene_cut_flags = []

        for scene_part in scene_parts:
            duration_match = re.search(r'\[(\d+\.?\d*)\s*s#?\]', scene_part)
            has_scene_cut = '#' in scene_part

            if duration_match:
                duration_seconds = float(duration_match.group(1))
                prompt_text = re.sub(r'\[\d+\.?\d*\s*s#?\]', '', scene_part).strip()
                blocks = max(1, int((duration_seconds * self.default_fps) / (4 * self.num_frame_per_block)))
            else:
                prompt_text = scene_part
                blocks = self.default_blocks_per_scene

            prompt_texts.append(prompt_text)
            block_counts.append(blocks)
            scene_cut_flags.append(has_scene_cut)

        return prompt_texts, block_counts, scene_cut_flags

    # ------------------------------------------------------------------
    # KV flush at scene boundaries
    # ------------------------------------------------------------------
    def _kv_flush(self, scene_cut_needed: bool, device: torch.device):
        """Flush KV cache at scene boundaries by rolling cache and resetting cross-attention."""
        n_layers = len(self.crossattn_cache)
        for i in range(n_layers):
            self.crossattn_cache[i]['is_init'] = False
            self.kv_cache1[i]['k'][:, 1560:4680] = self.kv_cache1[i]['k'][:, -3120:]
            self.kv_cache1[i]['v'][:, 1560:4680] = self.kv_cache1[i]['v'][:, -3120:]
            self.kv_cache1[i]['local_end_index'] = torch.tensor([4680], dtype=torch.long, device=device)
            self.kv_cache1[i]['scene_cut'] = scene_cut_needed

    # ------------------------------------------------------------------
    # Inference with multi-scene support
    # ------------------------------------------------------------------
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        # Parse scenes from prompt
        scene_prompts, scene_block_counts, scene_cut_flags = self._parse_scene_durations(text_prompts[0])
        conditional_dict_list = [self.text_encoder(text_prompts=[tp]) for tp in scene_prompts]

        # Calculate cumulative block indices for scene transitions
        scene_block_boundaries = []
        scene_cut_boundaries = []
        cumulative_blocks = 0
        for i, block_count in enumerate(scene_block_counts[:-1]):
            cumulative_blocks += block_count
            scene_block_boundaries.append(cumulative_blocks)
            if scene_cut_flags[i]:
                scene_cut_boundaries.append(cumulative_blocks)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("Scene configuration:")
            for i, (prompt, blocks, has_cut) in enumerate(zip(scene_prompts, scene_block_counts, scene_cut_flags)):
                duration_seconds = (blocks * 4 * self.num_frame_per_block) / self.default_fps
                cut_indicator = " [SCENE CUT]" if has_cut else ""
                print(f"  Scene {i+1}: {blocks} blocks ({duration_seconds:.2f}s){cut_indicator} - '{prompt[:50]}...'")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device, dtype=noise.dtype
        )

        # Initialize KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length

        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device, kv_cache_size_override=kv_cache_size)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Cache context feature (Initial Latent)
        if initial_latent is not None:
            initial_conditional_dict = conditional_dict_list[0]
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents.to(output.device)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=initial_conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks

        for current_block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            # Determine which scene this block belongs to
            scene_index = 0
            for boundary in scene_block_boundaries:
                if current_block_index < boundary:
                    break
                scene_index += 1
            conditional_dict = conditional_dict_list[scene_index]

            # Check if we need to flush KV cache (at scene boundaries)
            scene_cut_needed = current_block_index in scene_cut_boundaries
            if current_block_index in scene_block_boundaries:
                self._kv_flush(scene_cut_needed, noise.device)
            else:
                for i in range(len(self.kv_cache1)):
                    self.kv_cache1[i]['scene_cut'] = False

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            scene_cut = self.kv_cache1[0].get('scene_cut', False)

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    scene_cut=scene_cut
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])

            # Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                scene_cut=scene_cut
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Decode the output
        if getattr(self.args.model_kwargs, "use_infinite_attention", False):
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video
