import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


# ============================================================================
# GPU 配置预设
# ============================================================================

GPU_CONFIGS = {
    "sd3": {
        8: {"bsz": 9, "num_image_per_prompt": 24, "num_groups": 48, "test_batch_size": 14},
    },
    "wan": {
        8:  {"bsz": 3, "num_image_per_prompt": 24,  "num_groups": 16,  "test_batch_size": 3},
        16: {"bsz": 2, "num_image_per_prompt": 16, "num_groups": 16, "test_batch_size": 2},
        24: {"bsz": 4, "num_image_per_prompt": 24, "num_groups": 48, "test_batch_size": 4},
        48: {"bsz": 4, "num_image_per_prompt": 24, "num_groups": 48, "test_batch_size": 4},
    },
    "krea14b": {
        8:  {"bsz": 1, "num_image_per_prompt": 8,  "num_groups": 8,  "test_batch_size": 1},
        16: {"bsz": 1, "num_image_per_prompt": 16, "num_groups": 16, "test_batch_size": 1},
        24: {"bsz": 1, "num_image_per_prompt": 24, "num_groups": 24, "test_batch_size": 1},
        48: {"bsz": 1, "num_image_per_prompt": 16, "num_groups": 48, "test_batch_size": 1},
    },
}


def _get_gpu_config(model_type, n_gpus):
    """获取指定模型类型和 GPU 数量的配置

    Args:
        model_type: "sd3", "wan", 或 "krea14b"
        n_gpus: GPU 数量
    """
    if n_gpus not in GPU_CONFIGS[model_type]:
        raise ValueError(f"不支持的 GPU 数量: {n_gpus}，支持的值: {list(GPU_CONFIGS[model_type].keys())}")
    return GPU_CONFIGS[model_type][n_gpus]


def _apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch):
    """计算并设置 batch size 相关配置"""
    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1
    return config


def _apply_common_fields(config, base_model, dataset, reward_fn, name):
    """设置公共字段：prompt_fn, run_name, save_dir, reward_fn 等"""
    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"
    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn
    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"
    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def _make_base_config(dataset):
    """创建基础 config 并设置 dataset 路径"""
    assert dataset in ["pickscore", "ocr", "geneval", "vidprom"]
    config = base.get_config()
    config.dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"dataset/{dataset}")
    return config


def _apply_wan_common(config):
    """设置所有 Wan 变体共用的参数"""
    config.mixed_precision = "bf16"
    config.sample.denoising_step_list = [1000, 750, 500, 250]
    config.sample.warp_denoising_step = True
    config.sample.num_frame_per_block = 3
    config.sample.independent_first_frame = False
    config.sample.context_noise = 0
    config.sample.causal = True
    config.height = 480
    config.width = 832
    config.num_frames = 81
    config.sample.guidance_scale = 3.0
    config.train.beta = 0.0001
    config.sample.noise_level = 0.7
    return config
