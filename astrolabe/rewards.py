from PIL import Image
import os
import numpy as np
import torch


def aesthetic_score(device):
    from astrolabe.scorers.image.aesthetic import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def video_hpsv2_local(device):
    from astrolabe.scorers.image.hpsv2 import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        # Normalize input to Tensor [B, F, C, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # Flatten frames and replicate each prompt once per frame
        flat_images = videos.reshape(-1, c, h, w)
        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Score in mini-batches to avoid OOM
        reward_batch_size = 8
        all_flat_scores = []
        for i in range(0, len(flat_images), reward_batch_size):
            batch_scores = scorer(flat_images[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            all_flat_scores.append(batch_scores)
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards = []
        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            # Aggregate via top-30% mean to reduce noise from low-quality frames
            scores.sort(reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards.append(sum(scores[:top_k]) / top_k)

        return np.array(video_rewards, dtype=np.float32), {}

    return _fn


def video_hpsv3_local(device):
    from astrolabe.scorers.video.hpsv3 import HPSv3RewardInferencer

    # HPSv3 is a 7B VLM scorer; it returns (mu, sigma) per sample
    scorer = HPSv3RewardInferencer(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scorers/configs/HPSv3_7B.yaml'),
            checkpoint_path='./reward_ckpts/HPSv3.safetensors',
            device=device
        )

    def _fn(videos, prompts, metadata=None):
        # Normalize input to Tensor [B, F, C, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # HPSv3 expects a list of PIL images, so convert each frame
        flat_images_tensor = videos.reshape(-1, c, h, w)
        flat_images_np = (flat_images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        flat_images_np = flat_images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        flat_images_pil = [Image.fromarray(img) for img in flat_images_np]

        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Smaller batch size than HPSv2 due to the 7B model's higher VRAM usage
        reward_batch_size = 4
        all_flat_scores = []

        for i in range(0, len(flat_images_pil), reward_batch_size):
            batch_scores = scorer.reward(flat_images_pil[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            # scorer.reward returns [batch, 2] as (mu, sigma); take mu
            if batch_scores.ndim == 2:
                batch_scores = batch_scores[:, 0]
            all_flat_scores.append(batch_scores.cpu())
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards_top30 = []
        video_rewards_all = []

        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            avg_all = sum(scores) / len(scores)
            video_rewards_all.append(float(avg_all))
            # Top-30% mean reduces sensitivity to outlier frames
            scores_sorted = sorted(scores, reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards_top30.append(sum(scores_sorted[:top_k]) / top_k)

        # Return top-30% as the primary reward; full-frame avg carried in metadata
        return np.array(video_rewards_top30, dtype=np.float32), {"all_frame_avg": video_rewards_all}

    return _fn


def videoalign_score(device, reward_type="Overall", use_grayscale=False):
    from astrolabe.scorers.video.videoalign import VideoAlignScorer

    # reward_type selects the scoring dimension: "Overall", "VQ" (visual quality),
    # "MQ" (motion quality, grayscale-sensitive), or "TA" (text alignment)
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16, reward_type=reward_type, use_grayscale=use_grayscale)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)
        
        if videos.dtype != torch.uint8:
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)

        scores_tensor = scorer(list(videos), prompts)
        return scores_tensor.tolist(), {}

    return _fn


# Convenience wrappers for individual VideoAlign dimensions
def videoalign_vq_score(device):
    return videoalign_score(device, reward_type="VQ", use_grayscale=False)

def videoalign_mq_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=True)

def videoalign_mq_rgb_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=False)

def videoalign_ta_score(device):
    return videoalign_score(device, reward_type="TA", use_grayscale=False)


def motion_smoothness_score(device):
    from astrolabe.scorers.video.flowscorer import OpticalFlowSmoothnessScorer
    scorer = OpticalFlowSmoothnessScorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        # RAFT optical flow scorer expects (B, C, T, H, W), not (B, T, C, H, W)
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            if videos.ndim == 5 and videos.shape[2] == 3 and videos.shape[1] != 3:
                videos = videos.permute(0, 2, 1, 3, 4)  # (B,F,C,H,W) -> (B,C,F,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        scores = scorer(videos, prompts)
        return scores.cpu().tolist(), {}

    return _fn

def dynamic_degree_score(device):
    from astrolabe.scorers.video.dynamic_degree import DynamicDegreeScorer
    ckpt_path = ""  # set to checkpoint path if a pretrained model is available
    scorer = DynamicDegreeScorer(device=device, model_path=ckpt_path)

    def _fn(videos, prompts, metadata=None):
        # Normalize to Tensor [B, C, T, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            if videos.ndim == 5 and videos.shape[1] != 3 and videos.shape[2] == 3:
                videos = videos.permute(0, 2, 1, 3, 4)  # (B,F,C,H,W) -> (B,C,F,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        scores = scorer(videos)
        return scores.cpu().tolist(), {}

    return _fn


def unifiedreward_score_sglang(device):
    # Requires a running sglang server:
    #   python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5
    #       --api-key flowgrpo --port 17140 --chat-template chatml-llava
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        # Parse "Final Score: X" (1-5) from model free-text output; default to 0 on failure
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Resize to 512x512 to match the model's expected input resolution
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]  # Normalize 1-5 scale to [0, 1]
        return score, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "aesthetic": aesthetic_score,
        "unifiedreward": unifiedreward_score_sglang,
        "video_hpsv2": video_hpsv2_local,
        "video_hpsv2_local": video_hpsv2_local,
        "video_hpsv3_local": video_hpsv3_local,
        "motion_smoothness_score": motion_smoothness_score,
        "videoalign_score": lambda dev: videoalign_score(dev, "Overall"),
        "videoalign_vq_score": videoalign_vq_score,
        "videoalign_mq_score": videoalign_mq_score,
        "videoalign_ta_score": videoalign_ta_score,
        "dynamic_degree_score": dynamic_degree_score,
    }
    score_fns = {name: score_functions[name](device) for name in score_dict}

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        # Detect if input is video [B, F, C, H, W]
        is_video = isinstance(images, torch.Tensor) and images.ndim == 5

        for score_name, weight in score_dict.items():
            current_score_name = score_name
            if is_video and score_name == "hpsv2":
                current_score_name = "video_hpsv2"
                if "video_hpsv2" not in score_fns:
                    score_fns["video_hpsv2"] = video_hpsv2_local(device)

            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


if __name__ == "__main__":
    pass
