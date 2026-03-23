#!/usr/bin/env python3
"""
特点：
- 零配置依赖：所有参数都有默认值，YAML 配置完全可选
- 灵活输入：支持单个 prompt 或 prompt 文件
- 评分可选：默认跳过奖励模型评分
- 完全独立：不依赖 nft.py 训练配置

使用示例：
    # 最简单的用法
    torchrun --nproc_per_node=1 scripts/inference_wan.py \
        --base_model checkpoints/longlive_base.pt \
        --prompt "A cat running in the park" \
        --output_dir outputs/test

    # 使用 prompt 文件和 LoRA
    torchrun --nproc_per_node=8 scripts/inference_wan.py \
        --base_model checkpoints/longlive_models/models/longlive_base.pt \
        --lora_path logs/nft/wan21/casual_wan21_video_multi_reward_48gpu_hpsv3_casualforcing/checkpoints/checkpoint-420 \
        --num_frames 81 \
        --prompt_file prompts/MovieGenVideoBench_extended.txt \
        --output_dir outputs/lora_test
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import imageio
from tqdm import tqdm
from omegaconf import OmegaConf
from peft import PeftModel

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# Wan specific imports
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline import CausalInferencePipeline, SceneCausalInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 禁用不必要的日志

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    # 模型配置
    'model_kwargs': {
        'timestep_shift': 5.0,
        'local_attn_size': 12,
        'sink_size': 3,
        'is_causal': True,
        'infer_only': True,
    },

    # 推理配置
    'denoising_step_list': [1000, 750, 500, 250],
    'warp_denoising_step': True,
    'num_frame_per_block': 3,
    'context_noise': 0,
    'independent_first_frame': False,

    # 视频规格
    'height': 480,
    'width': 832,
    'num_frames': 81,
    'guidance_scale': 3.0,
    'fps': 16,

    # 混合精度
    'mixed_precision': 'bf16',
}


# ============================================================================
# 工具函数
# ============================================================================

def setup_distributed(rank: int, world_size: int):
    """初始化分布式进程组"""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """检查是否为主进程"""
    return rank == 0


def load_wan_weights(model, checkpoint_path: str):
    """加载 Wan 基础权重，处理前缀剥离"""
    logger.info(f"Loading Wan base weights from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 尝试从不同的键中提取 state_dict
    state_dict = checkpoint
    for key in ["generator_ema", "generator", "model", "state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    # 移除前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("model."):
            name = name[6:]
        if name.startswith("diffusion_model."):
            name = name[16:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    logger.info("Successfully loaded Wan base weights")


def save_video(video_tensor: torch.Tensor, path: str, fps: int = 16):
    """保存视频张量到 MP4 文件

    Args:
        video_tensor: [T, C, H, W] 张量
        path: 输出文件路径
        fps: 帧率
    """
    frames = []
    video_tensor = video_tensor.float().cpu()

    for t in range(video_tensor.shape[0]):
        frame = video_tensor[t].numpy().transpose(1, 2, 0)  # [H, W, C]

        # 归一化到 [0, 1]
        if frame.min() < -0.1:
            frame = (frame + 1.0) / 2.0
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame)

    # 保存为 MP4
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
    )


# ============================================================================
# 配置管理
# ============================================================================

def get_default_config() -> Dict:
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()


def load_config_from_yaml(yaml_path: str) -> Dict:
    """从 YAML 文件加载配置"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")

    yaml_config = OmegaConf.load(yaml_path)
    return OmegaConf.to_container(yaml_config, resolve=True)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """合并配置，override_config 优先"""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # 递归合并字典
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def override_from_cli(config: Dict, args: argparse.Namespace) -> Dict:
    """从 CLI 参数覆盖配置"""
    # 视频规格
    if args.num_frames is not None:
        config['num_frames'] = args.num_frames
    if args.height is not None:
        config['height'] = args.height
    if args.width is not None:
        config['width'] = args.width
    if args.guidance_scale is not None:
        config['guidance_scale'] = args.guidance_scale
    if args.fps is not None:
        config['fps'] = args.fps

    # 推理配置
    if args.denoising_steps is not None:
        config['denoising_step_list'] = [int(x) for x in args.denoising_steps.split(',')]
    if args.num_frame_per_block is not None:
        config['num_frame_per_block'] = args.num_frame_per_block
    if args.local_attn_size is not None:
        config['model_kwargs']['local_attn_size'] = args.local_attn_size
    if args.timestep_shift is not None:
        config['model_kwargs']['timestep_shift'] = args.timestep_shift

    # 混合精度
    if args.mixed_precision is not None:
        config['mixed_precision'] = args.mixed_precision

    return config


def build_config(args: argparse.Namespace) -> Dict:
    """构建最终配置

    优先级：CLI 参数 > YAML 配置 > 默认值
    """
    # 1. 加载默认值
    config = get_default_config()

    # 2. 如果提供了 YAML，从 YAML 覆盖
    if args.yaml_config:
        yaml_config = load_config_from_yaml(args.yaml_config)
        config = merge_configs(config, yaml_config)

    # 3. 从 CLI 参数覆盖（优先级最高）
    config = override_from_cli(config, args)

    return config


# ============================================================================
# 数据加载
# ============================================================================

class PromptDataset(Dataset):
    """简单的 Prompt 数据集"""

    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "metadata": {},
            "original_index": idx
        }


def collate_fn(examples):
    """数据整理函数"""
    prompts = [ex["prompt"] for ex in examples]
    metadatas = [ex["metadata"] for ex in examples]
    indices = [ex["original_index"] for ex in examples]
    return prompts, metadatas, indices


def load_prompts(args: argparse.Namespace) -> List[str]:
    """加载 prompts

    支持：
    - 单个 prompt（--prompt）
    - Prompt 文件（--prompt_file）
    """
    if args.prompt:
        return [args.prompt]

    elif args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            raise ValueError(f"No prompts found in {args.prompt_file}")

        return prompts

    else:
        raise ValueError("Must provide either --prompt or --prompt_file")


# ============================================================================
# 奖励模型
# ============================================================================

def setup_reward_models(args: argparse.Namespace, device: torch.device):
    """设置奖励模型（可选）

    默认返回 None（不启用评分）
    """
    if not args.reward_models:
        return None

    # 解析 JSON 格式的奖励模型配置
    try:
        reward_fn_dict = json.loads(args.reward_models)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for --reward_models: {e}")

    # 初始化奖励模型
    import astrolabe.rewards
    logger.info(f"Initializing reward models: {reward_fn_dict}")
    return astrolabe.rewards.multi_score(device, reward_fn_dict)


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="独立的 Wan 视频生成推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument("--base_model", type=str, required=True,
                        help="基础模型权重路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")

    # Prompt 输入（二选一）
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str,
                              help="单个 prompt")
    prompt_group.add_argument("--prompt_file", type=str,
                              help="Prompt 文件路径（每行一个 prompt）")

    # 可选参数 - 模型配置
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重路径（可选）")
    parser.add_argument("--yaml_config", type=str, default=None,
                        help="YAML 配置路径（可选，用于覆盖默认值）")
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["bf16", "fp16", "no"],
                        help="混合精度")

    # 可选参数 - 视频规格
    parser.add_argument("--num_frames", type=int, default=None,
                        help="帧数")
    parser.add_argument("--height", type=int, default=None,
                        help="高度")
    parser.add_argument("--width", type=int, default=None,
                        help="宽度")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="引导尺度")
    parser.add_argument("--fps", type=int, default=None,
                        help="视频帧率")

    # 可选参数 - 推理配置
    parser.add_argument("--denoising_steps", type=str, default=None,
                        help="去噪步数（逗号分隔，例如：1000,750,500,250）")
    parser.add_argument("--num_frame_per_block", type=int, default=None,
                        help="每块帧数")
    parser.add_argument("--local_attn_size", type=int, default=None,
                        help="本地注意力大小（-1 为全局注意力）")
    parser.add_argument("--timestep_shift", type=float, default=None,
                        help="时间步偏移")

    # 可选参数 - 评估控制
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大样本数（-1 为全部）")
    parser.add_argument("--save_videos", action="store_true", default=True,
                        help="是否保存视频")
    parser.add_argument("--no_save_videos", action="store_false", dest="save_videos",
                        help="不保存视频")

    # 可选参数 - 奖励模型
    parser.add_argument("--reward_models", type=str, default=None,
                        help='奖励模型列表（JSON 格式，例如：\'{"hpsv3": 1.0}\'）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 分布式设置
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if is_main_process(rank):
        logger.info(f"Rank {rank}/{world_size} initialized")
        logger.info(f"Arguments: {args}")

    # 构建配置
    config = build_config(args)

    if is_main_process(rank):
        logger.info(f"Final config: {json.dumps(config, indent=2)}")

    # 创建输出目录
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        if args.save_videos:
            os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # 混合精度设置
    mixed_precision_dtype = torch.float32
    if config['mixed_precision'] == "bf16":
        mixed_precision_dtype = torch.bfloat16
    elif config['mixed_precision'] == "fp16":
        mixed_precision_dtype = torch.float16

    # 加载模型
    if is_main_process(rank):
        logger.info("Loading Wan models...")

    # 创建 OmegaConf 对象（CausalInferencePipeline 需要）
    wan_config = OmegaConf.create(config)

    # 初始化模型组件
    text_encoder = WanTextEncoder()
    vae = WanVAEWrapper()

    # 检测是否为 safetensors 格式（例如 KREA 14B）
    is_safetensors = args.base_model.endswith('.safetensors')

    if is_safetensors:
        # Safetensors 模式：直接通过 checkpoint_path 加载，自动检测架构（1.3B / 14B）
        config['model_kwargs']['checkpoint_path'] = args.base_model
        # KREA 14B 默认 timestep_shift=5.0，已在 DEFAULT_CONFIG 中设置；
        # 用户可通过 --timestep_shift 覆盖
        generator = WanDiffusionWrapper(**config['model_kwargs'])
        if is_main_process(rank):
            logger.info(f"Loaded model from safetensors: {args.base_model} (auto-detected architecture)")
    else:
        # .pt 模式：原有 Wan 1.3B 流程
        generator = WanDiffusionWrapper(**config['model_kwargs'])
        if os.path.exists(args.base_model):
            load_wan_weights(generator.model, args.base_model)
        else:
            raise FileNotFoundError(f"Base model not found: {args.base_model}")

    # 移动到设备
    vae.to(device, dtype=torch.bfloat16)
    text_encoder.to(device, dtype=mixed_precision_dtype)
    generator.to(device, dtype=mixed_precision_dtype)

    # 启用梯度检查点
    if hasattr(generator.model, "gradient_checkpointing"):
        generator.model.gradient_checkpointing = True

    # 加载 LoRA（可选）
    if args.lora_path:
        # 智能路径检测：如果提供的是 checkpoint 目录，自动添加 /lora 后缀
        lora_path = args.lora_path

        # 检查是否需要添加 /lora 后缀
        if not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            # 尝试在子目录中查找
            if os.path.exists(os.path.join(lora_path, "lora", "adapter_config.json")):
                lora_path = os.path.join(lora_path, "lora")
                if is_main_process(rank):
                    logger.info(f"Detected LoRA in subdirectory: {lora_path}")

        if is_main_process(rank):
            logger.info(f"Loading LoRA from: {lora_path}")

        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            generator.model = PeftModel.from_pretrained(generator.model, lora_path)
            generator.model = generator.model.merge_and_unload()
        else:
            raise FileNotFoundError(
                f"LoRA adapter_config.json not found at: {lora_path}\n"
                f"Please check the path. Expected structure:\n"
                f"  {lora_path}/adapter_config.json\n"
                f"  {lora_path}/adapter_model.bin"
            )

    generator.eval()

    # 加载 prompts（需要在创建 pipeline 之前，以便判断是否需要 scene 模式）
    prompts = load_prompts(args)
    if is_main_process(rank):
        logger.info(f"Loaded {len(prompts)} prompts")

    # 创建推理管道：如果任何 prompt 包含 '|' 则使用 SceneCausalInferencePipeline
    has_scene_prompts = any('|' in p for p in prompts)
    pipeline_cls = SceneCausalInferencePipeline if has_scene_prompts else CausalInferencePipeline
    if is_main_process(rank) and has_scene_prompts:
        logger.info("Detected scene prompts (containing '|'), using SceneCausalInferencePipeline")

    pipeline = pipeline_cls(
        args=wan_config,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=vae
    )

    # 创建数据集和数据加载器
    dataset = PromptDataset(prompts)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False
    )

    # 设置奖励模型（可选）
    scoring_fn = setup_reward_models(args, device)

    # 推理循环
    results_this_rank = []

    # 计算 latent 维度
    latent_t = (config['num_frames'] - 1) // 4 + 1
    latent_h = config['height'] // 8
    latent_w = config['width'] // 8

    if is_main_process(rank):
        logger.info(f"Generating videos: {config['num_frames']} frames, {config['height']}x{config['width']}")
        logger.info(f"Latent dimensions: T={latent_t}, H={latent_h}, W={latent_w}")

    total_processed = 0
    enable_amp = mixed_precision_dtype != torch.float32

    for batch in tqdm(dataloader, desc=f"Inference (Rank {rank})", disable=not is_main_process(rank)):
        if args.max_samples > 0 and total_processed >= args.max_samples:
            break

        prompts_batch, metadata_batch, indices_batch = batch
        current_batch_size = len(prompts_batch)
        total_processed += current_batch_size

        # 生成随机噪声
        noise = torch.randn(
            [current_batch_size, latent_t, 16, latent_h, latent_w],
            device=device,
            dtype=mixed_precision_dtype
        )

        # 推理
        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                videos = pipeline.inference(
                    noise=noise,
                    text_prompts=prompts_batch,
                    return_latents=False
                )

        # 计算奖励分数（可选）
        if scoring_fn is not None:
            all_scores, _ = scoring_fn(videos, prompts_batch, metadata_batch, only_strict=False)
        else:
            all_scores = {}

        # 保存结果
        for i in range(current_batch_size):
            sample_idx = indices_batch[i]
            result_item = {
                "sample_id": sample_idx,
                "prompt": prompts_batch[i],
                "metadata": metadata_batch[i],
                "scores": {}
            }

            # 保存视频
            if args.save_videos:
                video_filename = f"{sample_idx:05d}.mp4"
                video_path = os.path.join(args.output_dir, "videos", video_filename)
                save_video(videos[i], video_path, fps=config['fps'])
                result_item["video_path"] = video_path

            # 保存分数
            if scoring_fn is not None:
                for score_name, score_values in all_scores.items():
                    if isinstance(score_values, torch.Tensor):
                        result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                    else:
                        result_item["scores"][score_name] = float(score_values[i])

            results_this_rank.append(result_item)

        # 清理显存
        if hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()

        del videos, all_scores, noise
        torch.cuda.empty_cache()

    # 收集所有结果
    if world_size > 1:
        dist.barrier()
        all_gathered_results = [None] * world_size
        dist.all_gather_object(all_gathered_results, results_this_rank)
    else:
        all_gathered_results = [results_this_rank]

    # 保存结果（仅主进程）
    if is_main_process(rank):
        results_filepath = os.path.join(args.output_dir, "results.jsonl")
        flat_results = [item for sublist in all_gathered_results for item in sublist]
        flat_results.sort(key=lambda x: x["sample_id"])

        with open(results_filepath, "w") as f:
            for result_item in flat_results:
                f.write(json.dumps(result_item) + "\n")

        logger.info(f"\nInference finished. Saved {len(flat_results)} results to {results_filepath}")

        # 计算平均分数（如果有）
        if scoring_fn is not None:
            from collections import defaultdict
            all_scores_agg = defaultdict(list)

            for result in flat_results:
                for score_name, score_value in result["scores"].items():
                    if score_value > -9.0:
                        all_scores_agg[score_name].append(score_value)

            average_scores = {k: np.mean(v) for k, v in all_scores_agg.items() if v}

            logger.info("\n--- Average Scores ---")
            for name, avg_score in sorted(average_scores.items()):
                logger.info(f"{name:<20}: {avg_score:.4f}")

            with open(os.path.join(args.output_dir, "average_scores.json"), "w") as f:
                json.dump(average_scores, f, indent=4)

    # 清理
    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
