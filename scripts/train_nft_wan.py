from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                                                                  
sys.path.insert(0, project_root)

import astrolabe.rewards
from astrolabe.stat_tracking import PerPromptStatTracker
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import imageio
import peft
from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from astrolabe.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

# Wan specific imports
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline import CausalInferencePipeline

# Create pipeline args from config
from types import SimpleNamespace

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    # Increase timeout to 30 minutes
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    rewards_avg = gathered_rewards["avg"]
    if rewards_avg.ndim > 1:
        rewards_avg = rewards_avg[:, 0]
    grouped_rewards = rewards_avg[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


class RiskCompensator:
    """Dynamic KL threshold controller based on reward uncertainty."""
    def __init__(self, base=0.1, beta=0.1):
        self.risk_buffer = []
        self.base = base
        self.alpha = 0.1
        self.beta = beta
        self.beta_min = 0.02
        self.beta_max = 0.2

    def update(self, pad_scores):
        if len(self.risk_buffer) == 0:
            self.beta = self.base
        elif np.mean(pad_scores) > max(self.risk_buffer):
            self.beta = self.beta * (1 + self.alpha)
        elif np.mean(pad_scores) < min(self.risk_buffer):
            self.beta = self.beta * (1 - self.alpha)
        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)
        self.risk_buffer.append(np.mean(pad_scores))
        if len(self.risk_buffer) > 20:
            self.risk_buffer = self.risk_buffer[-20:]


def _configure_lora_for_wan(transformer, rank=256, alpha=256, dropout=0.0, adapter_target_modules=None):
    """Configure LoRA for a WanDiffusionWrapper model"""
    if adapter_target_modules is None:
        adapter_target_modules = ['CausalWanAttentionBlock']
    target_linear_modules = set()
    
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = list(target_linear_modules)
    
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_linear_modules,
        bias="none",
        task_type=None
    )
    
    lora_model = get_peft_model(transformer, peft_config)
    return lora_model



def eval_fn(
    pipeline,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    torch.cuda.empty_cache()
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.generator.eval()
    if hasattr(pipeline.generator.model, "set_adapter"):
        pipeline.generator.model.set_adapter("default")

    all_rewards = defaultdict(list)
    
    
    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        # batch_size=2, #TODO 和训练不一致kv cache复用会报错
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    eval_batches = getattr(config, 'eval_batches', 20)
    for i, test_batch in enumerate(tqdm(eval_loader, desc="Eval: ", disable=not is_main_process(rank), position=0)):
        if i >= eval_batches:
            break
        prompts, prompt_metadata = test_batch
        
        num_frames = config.num_frames
        h, w = config.height, config.width
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = h // 8
        latent_w = w // 8
        
        noise = torch.randn([len(prompts), latent_t, 16, latent_h, latent_w], device=device, dtype=mixed_precision_dtype)

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                video = pipeline.inference(noise=noise, text_prompts=prompts, return_latents=False)

        rewards_future = executor.submit(reward_fn, video, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    torch.cuda.empty_cache()
    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        video_to_log = video[0].cpu().float() # 取 batch 中的第一个视频 [T, C, H, W]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"eval_step_{global_step}.mp4")
            
            # 视频归一化处理 (Wan 输出通常在 0-1 之间，但安全起见做一次检查)
            # video_to_log 形状是 [T, C, H, W]
            frames = []
            for t in range(video_to_log.shape[0]):
                frame = video_to_log[t].numpy().transpose(1, 2, 0) # [H, W, C]
                # 检查是否需要从 [-1, 1] 映射到 [0, 1]
                if frame.min() < -0.1:
                    frame = (frame + 1.0) / 2.0
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)
            
            # 使用 imageio 保存为 mp4
            # fps 设置为 8 或 16 (Wan 默认通常是 16fps，但为了预览 8fps 比较稳)
            imageio.v3.imwrite(
                video_path,
                frames,
                extension=".mp4",
                fps=16,
                codec="libx264",
                pixelformat="yuv420p",
            )

            mid_frame = frames[len(frames)//2]
            Image.fromarray(mid_frame).save(os.path.join(tmpdir, "mid_frame.jpg"))
            
            wandb.log(
                {
                    "eval_video": wandb.Video(video_path, caption=prompts[0], format="mp4"),
                    "eval_mid_frame": wandb.Image(os.path.join(tmpdir, "mid_frame.jpg"), caption=prompts[0]),
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )
            logger.info(f"Step {global_step}: Evaluated and uploaded video.")

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        # model_to_save should be the PEFT model (transformer_ddp.module.model)
        model_to_save = transformer_ddp.module.model

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        # Save LoRA weights
        lora_state_dict = get_peft_model_state_dict(model_to_save)
        torch.save(lora_state_dict, os.path.join(save_root_lora, "adapter_model.bin"))
        # Also save config
        model_to_save.peft_config['default'].save_pretrained(save_root_lora)

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def load_wan_weights(model, checkpoint_path):
    logger.info(f"Loading weights from {checkpoint_path}")

    # Support both .pt and .safetensors formats
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        logger.info(f"Loading from safetensors: {checkpoint_path}")
        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        for key in ["generator_ema", "generator", "model", "state_dict"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'model.' 或 'diffusion_model.' 前缀
        name = k
        if name.startswith("model."):
            name = name[6:] # 去掉 "model." 这 6 个字符
        if name.startswith("diffusion_model."):
            name = name[16:]

        new_state_dict[name] = v

    try:
        info = model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"Successfully loaded: {info}")
    except RuntimeError as e:
        logger.error(f"Weight mismatch error: {e}")
        # 如果还是报错，打印出模型预期的前 5 个 Key 看看
        logger.error(f"Model expected keys example: {list(model.state_dict().keys())[:5]}")
        raise e 
    

def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    logger.info(f"Rank {rank} (Local {local_rank}) / {world_size} initializing...")
    setup_distributed(rank, local_rank, world_size)
    logger.info(f"Rank {rank} initialized process group.")
    
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # Add timestamp to save_dir to prevent weight overwriting
    config.save_dir = config.save_dir + "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="flow-grpo-wan", name=config.run_name, config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = torch.float32
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype != torch.float32
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))

    # --- Load Wan models ---
    # Use model_kwargs from config (no YAML dependency)
    model_kwargs = getattr(config, "model_kwargs", {})

    # Convert ConfigDict to regular dict to allow modifications
    if hasattr(model_kwargs, 'to_dict'):
        model_kwargs = model_kwargs.to_dict()
    else:
        model_kwargs = dict(model_kwargs)

    # For Krea 14B, add timestep_shift and checkpoint_path
    if hasattr(config, 'is_krea_14b') and config.is_krea_14b:
        logger.info("Detected Krea 14B configuration")
        model_kwargs['timestep_shift'] = getattr(config, 'timestep_shift', 5.0)
        model_kwargs['checkpoint_path'] = config.pretrained.model
        logger.info(f"Using timestep_shift={model_kwargs['timestep_shift']}")
        logger.info(f"Loading from checkpoint: {model_kwargs['checkpoint_path']}")

    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)


    # Load base weights (skip if already loaded via checkpoint_path)
    if config.pretrained.model and not (hasattr(config, 'is_krea_14b') and config.is_krea_14b):
        logger.info(f"Loading Wan base weights from {config.pretrained.model}")
        load_wan_weights(generator.model, config.pretrained.model)
    
    text_encoder = WanTextEncoder()
    vae = WanVAEWrapper()
    
    vae.to(device, dtype=torch.bfloat16)
    text_encoder.to(device, dtype=mixed_precision_dtype)
    generator.to(device, dtype=mixed_precision_dtype)
    
    # Enable gradient checkpointing to save VRAM
    # Manually set the flag to avoid incompatible diffusers interface
    generator.model.gradient_checkpointing = True

    # --- Apply LoRA ---
    lora_cfg = config.lora
    transformer = _configure_lora_for_wan(
        generator.model,
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        adapter_target_modules=list(lora_cfg.target_modules),
    )
    generator.model = transformer

    # --- Load LongLive LoRA weights as initialization (if specified) ---
    if hasattr(config, 'longlive_lora_init') and config.longlive_lora_init:
        try:
            logger.info(f"Loading LongLive LoRA weights from {config.longlive_lora_init} as initialization")
            lora_checkpoint = torch.load(config.longlive_lora_init, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                lora_state_dict = lora_checkpoint["generator_lora"]
            else:
                lora_state_dict = lora_checkpoint

            # Load LoRA weights
            peft.set_peft_model_state_dict(generator.model, lora_state_dict)
            logger.info(f"Successfully loaded LongLive LoRA weights with {len(lora_state_dict)} keys")
        except FileNotFoundError:
            logger.warning(f"LongLive LoRA path {config.longlive_lora_init} does not exist, skipping initialization")
        except Exception as e:
            logger.error(f"Failed to load LongLive LoRA: {e}")

    transformer_ddp = DDP(generator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Set up adapters
    transformer_ddp.module.model.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

    # Reference model (old adapter)
    transformer_ddp.module.model.add_adapter("old", transformer_ddp.module.model.peft_config["default"])
    transformer_ddp.module.model.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.model.set_adapter("default")

    # GARDO: add old_ref adapter (lagging KL reference, updated less frequently)
    old_ref_transformer_trainable_parameters = None
    if getattr(config, 'use_select_kl', False):
        transformer_ddp.module.model.add_adapter("old_ref", transformer_ddp.module.model.peft_config["default"])
        transformer_ddp.module.model.set_adapter("old_ref")
        old_ref_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
        transformer_ddp.module.model.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError()

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    executor = futures.ThreadPoolExecutor(max_workers=4)

    reward_fn = getattr(astrolabe.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = reward_fn


    pipeline_args = SimpleNamespace(
        context_noise=config.sample.context_noise,
        model_kwargs=config.model_kwargs,
        denoising_step_list=config.sample.denoising_step_list,
        warp_denoising_step=config.sample.warp_denoising_step,
        num_frame_per_block=config.sample.num_frame_per_block
    )

    pipeline = CausalInferencePipeline(
        args=pipeline_args,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=vae
    )
    
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        lora_path = os.path.join(config.resume_from, "lora/adapter_model.bin")
        if os.path.exists(lora_path):
            transformer_ddp.module.model.load_adapter(os.path.dirname(lora_path), adapter_name="default")
            transformer_ddp.module.model.load_adapter(os.path.dirname(lora_path), adapter_name="old")
            if getattr(config, 'use_select_kl', False):
                transformer_ddp.module.model.load_adapter(os.path.dirname(lora_path), adapter_name="old_ref")
        
        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
        except:
            pass

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = len(config.sample.denoising_step_list)

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # Sync old adapter with default
    for src_param, tgt_param in zip(transformer_trainable_parameters, old_transformer_trainable_parameters):
        tgt_param.data.copy_(src_param.detach().data)

    # GARDO: sync old_ref and init dynamic controller
    if getattr(config, 'use_select_kl', False):
        assert len(config.reward_fn) >= 2, "use_select_kl requires at least 2 reward functions"
        for src_param, tgt_param in zip(transformer_trainable_parameters, old_ref_transformer_trainable_parameters):
            tgt_param.data.copy_(src_param.detach().data)
        dynamic_controller = RiskCompensator(base=0.1, beta=0.1)
        thres = 0.15
        prev_epoch = 0
        reset = False

    for epoch in range(config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # GARDO: update old_ref adapter periodically based on KL divergence
        if getattr(config, 'use_select_kl', False) and config.train.beta > 0:
            if (epoch != 0 and reset) or (epoch - prev_epoch) > 15:
                with torch.no_grad():
                    for src_param, tgt_param in zip(transformer_trainable_parameters, old_ref_transformer_trainable_parameters):
                        tgt_param.data.copy_(src_param.detach().clone().data)
                reset = False
                if epoch - prev_epoch > 15:
                    thres = max(thres - 1e-2, 1e-1)
                if epoch - prev_epoch < 5:
                    thres = min(thres + 1e-2, 5e-2)
                prev_epoch = epoch

        # SAMPLING
        generator.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
        ):
            transformer_ddp.module.model.set_adapter("default")
            prompts, prompt_metadata = next(train_iter)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(pipeline, 
                        test_dataloader, 
                        config, 
                        device, 
                        rank, 
                        world_size, 
                        global_step, 
                        eval_reward_fn, 
                        executor, 
                        mixed_precision_dtype, 
                        ema, 
                        transformer_trainable_parameters)
                # pass

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler)

            transformer_ddp.module.model.set_adapter("old")
            
            num_frames = config.num_frames
            h, w = config.height, config.width
            latent_t = (num_frames - 1) // 4 + 1
            latent_h = h // 8
            latent_w = w // 8
            
            noise = torch.randn(
                [len(prompts), latent_t, 16, latent_h, latent_w], 
                device=device, dtype=mixed_precision_dtype
            )

            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    video, latents = pipeline.inference(
                        noise=noise,
                        text_prompts=prompts,
                        return_latents=True,
                    )
            transformer_ddp.module.model.set_adapter("default")

            # latents is [B, T, C, H, W]
            rewards_future = executor.submit(reward_fn, video, prompts, prompt_metadata, only_strict=True)
            time.sleep(1)

            samples_data_list.append({
                "prompts": prompts,
                "latents_clean": latents.detach(),
                "rewards_future": rewards_future,
                "video": video.detach() if i == config.sample.num_batches_per_epoch - 1 else None,
            })

        for sample_item in samples_data_list:
            rewards, _ = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate
        collated_samples = {
            "prompts": [p for s in samples_data_list for p in s["prompts"]],
            "latents_clean": torch.cat([s["latents_clean"] for s in samples_data_list], dim=0),
            "rewards": {k: torch.cat([s["rewards"][k] for s in samples_data_list], dim=0) for k in samples_data_list[0]["rewards"].keys()}
        }

        # Logging images/videos (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            # Use the last sampled video from this rank
            video_to_log = samples_data_list[-1]["video"][0].cpu().float()
            prompt_to_log = samples_data_list[-1]["prompts"][0]
            reward_to_log = samples_data_list[-1]["rewards"]["avg"][0].item()

            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = os.path.join(tmpdir, f"sample_epoch_{epoch}.mp4")
                frames = []
                for t in range(video_to_log.shape[0]):
                    frame = video_to_log[t].numpy().transpose(1, 2, 0)
                    if frame.min() < -0.1:
                        frame = (frame + 1.0) / 2.0
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    frames.append(frame)
                imageio.v3.imwrite(video_path, frames, extension=".mp4", fps=16, codec="libx264", pixelformat="yuv420p")
                
                mid_frame = frames[len(frames)//2]
                Image.fromarray(mid_frame).save(os.path.join(tmpdir, "mid_frame.jpg"))

                wandb.log(
                    {
                        "sample_video": wandb.Video(video_path, caption=f"{prompt_to_log:.100} | avg: {reward_to_log:.2f}", format="mp4"),
                        "sample_mid_frame": wandb.Image(os.path.join(tmpdir, "mid_frame.jpg"), caption=f"{prompt_to_log:.100} | avg: {reward_to_log:.2f}"),
                    },
                    step=global_step,
                )

        # Advantages
        gathered_rewards_dict = {k: gather_tensor_to_all(v, world_size).numpy() for k, v in collated_samples["rewards"].items()}

        if is_main_process(rank) and getattr(config, 'use_select_kl', False):
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{k}": v.mean() for k, v in gathered_rewards_dict.items()
                       if "_strict_accuracy" not in k and "_accuracy" not in k},
                },
                step=global_step,
            )

        if stat_tracker:
            all_prompts_gathered = [None for _ in range(world_size)]
            dist.all_gather_object(all_prompts_gathered, collated_samples["prompts"])
            flat_prompts = [p for sublist in all_prompts_gathered for p in sublist]

            advantages = stat_tracker.update(flat_prompts, gathered_rewards_dict["avg"])

            # GARDO: compute reward uncertainty mask based on ranking differences
            if getattr(config, 'use_select_kl', False):
                reward_std = []
                main_reward = getattr(config, 'main_reward', 'avg')
                reward_keys = [k for k in gathered_rewards_dict.keys() if k not in ['avg', main_reward]][:2]
                reward_keys.append(main_reward)

                for item in reward_keys:
                    if item in gathered_rewards_dict:
                        _, indices = torch.sort(torch.tensor(gathered_rewards_dict[item]), descending=True)
                        ranks = torch.empty_like(torch.tensor(gathered_rewards_dict[item])).long()
                        ranks[indices] = torch.arange(1, len(gathered_rewards_dict[item]) + 1)
                        reward_std.append(ranks.float())

                if len(reward_std) >= 2:
                    reward_std_ = reward_std[-1] - sum(reward_std[:-1]) / len(reward_std[:-1])
                    dynamic_controller.update(np.array([p for p in reward_std_.cpu().numpy() if p >= 0]))
                    threshold = np.percentile(np.array([p for p in reward_std_.cpu().numpy() if p >= 0]), 100 - dynamic_controller.beta * 100)
                    reward_std_mask = reward_std_ > threshold
                else:
                    reward_std_mask = torch.ones(len(gathered_rewards_dict["avg"]), dtype=torch.bool)
                    reward_std_ = torch.zeros(len(gathered_rewards_dict["avg"]))
                    threshold = 0.0

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(flat_prompts, gathered_rewards_dict)
                log_dict = {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                    "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                    "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                    "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                    "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                    "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                }
                if getattr(config, 'use_select_kl', False):
                    log_dict.update({
                        "reward_rank_min": reward_std_.min().item(),
                        "reward_rank_mean": reward_std_.mean().item(),
                        "reward_rank_max": reward_std_.max().item(),
                        "std_thres": threshold if isinstance(threshold, float) else threshold.item(),
                    })
                wandb.log(log_dict, step=global_step)
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
            if getattr(config, 'use_select_kl', False):
                reward_std_mask = torch.ones(len(avg_rewards_all), dtype=torch.bool)

        samples_per_gpu = len(collated_samples["prompts"])
        local_advantages = torch.from_numpy(advantages.reshape(world_size, samples_per_gpu)[rank]).to(device)
        collated_samples["advantages"] = local_advantages

        # GARDO: distribute reward_std_mask to local process
        if getattr(config, 'use_select_kl', False):
            local_reward_std = reward_std_mask.reshape(world_size, samples_per_gpu)[rank].to(device)
            collated_samples["reward_std"] = local_reward_std

        # TRAINING
        transformer_ddp.train()
        current_accumulated_steps = 0
        info_accumulated = defaultdict(list)

        for inner_epoch in range(config.train.num_inner_epochs):
            indices = torch.randperm(len(collated_samples["prompts"]), device=device)

            for i in range(0, len(indices), config.train.batch_size):
                batch_idx = indices[i : i + config.train.batch_size]
                if len(batch_idx) < config.train.batch_size: continue

                batch_prompts = [collated_samples["prompts"][idx] for idx in batch_idx.cpu().numpy()]
                batch_latents_x0 = collated_samples["latents_clean"][batch_idx]
                batch_advantages = collated_samples["advantages"][batch_idx]

                with torch.no_grad():
                    cond_dict = text_encoder(text_prompts=batch_prompts)

                # Randomly sample ONE timestep for training
                t_idx = random.randint(0, len(config.sample.denoising_step_list) - 1)
                current_t = config.sample.denoising_step_list[t_idx]

                # Flow matching forward: x_t = (1 - t) * x0 + t * noise
                t_val = current_t / 1000.0
                noise = torch.randn_like(batch_latents_x0)
                xt = (1 - t_val) * batch_latents_x0 + t_val * noise

                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    # Old prediction
                    transformer_ddp.module.model.set_adapter("old")
                    with torch.no_grad():
                        _, old_pred = transformer_ddp(
                            noisy_image_or_video=xt,
                            timestep=torch.ones([len(batch_idx), xt.shape[1]], device=device, dtype=torch.long) * current_t,
                            conditional_dict=cond_dict,
                        )

                    # Default prediction
                    transformer_ddp.module.model.set_adapter("default")
                    _, forward_pred_x0 = transformer_ddp(
                        noisy_image_or_video=xt,
                        timestep=torch.ones([len(batch_idx), xt.shape[1]], device=device, dtype=torch.long) * current_t,
                        conditional_dict=cond_dict,
                    )

                    # Reference prediction
                    with torch.no_grad():
                        if getattr(config, 'use_select_kl', False):
                            # GARDO: use old_ref adapter as KL reference
                            transformer_ddp.module.model.set_adapter("old_ref")
                            _, ref_pred_x0 = transformer_ddp(
                                noisy_image_or_video=xt,
                                timestep=torch.ones([len(batch_idx), xt.shape[1]], device=device, dtype=torch.long) * current_t,
                                conditional_dict=cond_dict,
                            )
                            transformer_ddp.module.model.set_adapter("default")
                        else:
                            # Standard: use frozen base model as KL reference
                            with transformer_ddp.module.model.disable_adapter():
                                _, ref_pred_x0 = transformer_ddp(
                                    noisy_image_or_video=xt,
                                    timestep=torch.ones([len(batch_idx), xt.shape[1]], device=device, dtype=torch.long) * current_t,
                                    conditional_dict=cond_dict,
                                )

                # Clip and normalize advantages
                adv = torch.clamp(batch_advantages, -config.train.adv_clip_max, config.train.adv_clip_max)
                r = (adv / config.train.adv_clip_max) / 2.0 + 0.5
                r = torch.clamp(r, 0, 1)

                # Policy loss
                pos_pred_x0 = config.beta * forward_pred_x0 + (1 - config.beta) * old_pred.detach()
                neg_pred_x0 = (1.0 + config.beta) * old_pred.detach() - config.beta * forward_pred_x0

                loss_pos = ((pos_pred_x0 - batch_latents_x0)**2).mean(dim=tuple(range(1, batch_latents_x0.ndim)))
                loss_neg = ((neg_pred_x0 - batch_latents_x0)**2).mean(dim=tuple(range(1, batch_latents_x0.ndim)))

                policy_loss = (r * loss_pos + (1 - r) * loss_neg).mean()

                # KL loss
                if getattr(config, 'use_select_kl', False):
                    # GARDO: selective KL — only apply to high-uncertainty samples
                    batch_reward_std = collated_samples["reward_std"][batch_idx]
                    idx = batch_reward_std > 0
                    if idx.any():
                        kl_loss = ((forward_pred_x0[idx] - ref_pred_x0[idx])**2).mean(dim=tuple(range(1, batch_latents_x0.ndim)))
                    else:
                        kl_loss = ((forward_pred_x0 - ref_pred_x0)**2).mean(dim=tuple(range(1, batch_latents_x0.ndim))) * 0
                    loss = policy_loss + config.train.beta * torch.mean(kl_loss)
                else:
                    kl_loss = torch.mean((forward_pred_x0 - ref_pred_x0)**2)
                    loss = policy_loss + config.train.beta * kl_loss

                loss_terms = {}
                loss_terms["total_loss"] = loss.detach()
                loss_terms["policy_loss"] = policy_loss.detach()
                loss_terms["kl_loss"] = (torch.mean(kl_loss) if getattr(config, 'use_select_kl', False) else kl_loss).detach()
                loss_terms["x0_norm"] = torch.mean(batch_latents_x0**2).detach()
                loss_terms["old_deviate"] = torch.mean((forward_pred_x0 - old_pred) ** 2).detach()
                loss_terms["old_kl_div"] = torch.mean((old_pred - ref_pred_x0) ** 2).detach()
                if getattr(config, 'use_select_kl', False):
                    loss_terms["kl_div"] = torch.mean((forward_pred_x0 - ref_pred_x0)**2).detach()

                scaled_loss = loss / config.train.gradient_accumulation_steps
                if config.mixed_precision == "fp16" and scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Clear intermediate tensors to free VRAM
                del old_pred, forward_pred_x0, ref_pred_x0, pos_pred_x0, neg_pred_x0, loss_pos, loss_neg, policy_loss, kl_loss, loss, xt, noise

                current_accumulated_steps += 1
                for k_info, v_info in loss_terms.items():
                    info_accumulated[k_info].append(v_info)

                if current_accumulated_steps % config.train.gradient_accumulation_steps == 0:
                        if config.mixed_precision == "fp16" and scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)
                        if config.mixed_precision == "fp16" and scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                        if config.train.ema and ema is not None:
                            if global_step >= config.train.ema_start_step:
                                if global_step == config.train.ema_start_step:
                                    ema.sync_with_model(transformer_trainable_parameters)
                                ema.step(transformer_trainable_parameters, global_step)

                        # Log averaged info across accumulation steps
                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}

                        # GARDO: trigger old_ref reset if KL exceeds threshold
                        if getattr(config, 'use_select_kl', False) and reduced_log_info.get("kl_loss", 0) > thres:
                            reset = True

                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                },
                                step=global_step
                            )
                        global_step += 1
                        info_accumulated = defaultdict(list)

        # Update old adapter
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(transformer_trainable_parameters, old_transformer_trainable_parameters):
                tgt_param.data.copy_(tgt_param.data * decay + src_param.data * (1.0 - decay))

    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
