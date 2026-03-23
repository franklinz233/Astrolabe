import os
import imp

_base = imp.load_source("_base_clean", os.path.join(os.path.dirname(__file__), "_base_clean.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, gradient_step_per_epoch=1, dataset="vidprom", reward_fn={}, name=""):
    """LongLive (wan_longlive) 专用配置构建函数"""
    config = _base._make_base_config(dataset)
    config.base_model = "wan_longlive"

    _base._apply_wan_common(config)

    config.pretrained.model = "./checkpoints/longlive_models/models/longlive_base.pt"
    config.model_kwargs = {"timestep_shift": 5.0, "local_attn_size": 12, "sink_size": 3}

    gpu_config = _base._get_gpu_config("wan", n_gpus)
    bsz = gpu_config["bsz"]
    config.sample.num_image_per_prompt = gpu_config["num_image_per_prompt"]
    num_groups = gpu_config["num_groups"]
    config.sample.test_batch_size = gpu_config["test_batch_size"]

    _base._apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch)
    _base._apply_common_fields(config, "wan_longlive", dataset, reward_fn, name)
    return config


# ============================================================================
# 8 GPU
# ============================================================================

def longlive_video_hpsv3():
    """LongLive + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def longlive_video_multi_reward():
    """LongLive + 多奖励 (MQ + HPSv3) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_multi_reward",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def longlive_video_mq():
    """LongLive + Motion Quality - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0},
        name="longlive_video_mq",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 20
    return config


def longlive_video_long():
    """LongLive + HPSv3 + 长视频 (240 帧) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_long",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.num_frames = 240
    return config


def longlive_video_hpsv3_with_lora_init():
    """LongLive + HPSv3 + LoRA 初始化 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_with_lora_init",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.longlive_lora_init = "./checkpoints/longlive_models/models/lora.pt"
    return config


def longlive_video_multi_reward_with_lora_init():
    """LongLive + 多奖励 + LoRA 初始化 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_multi_reward_with_lora_init",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.longlive_lora_init = "./checkpoints/longlive_models/models/lora.pt"
    return config


# ============================================================================
# 16 GPU
# ============================================================================

def longlive_video_hpsv3_16gpu():
    """LongLive + HPSv3 - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def longlive_video_mq_16gpu():
    """LongLive + Motion Quality - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0},
        name="longlive_video_mq_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 20
    return config


def longlive_video_multi_reward_16gpu():
    """LongLive + 多奖励 (MQ + HPSv3) - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_multi_reward_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 24 GPU
# ============================================================================

def longlive_video_hpsv3_24gpu():
    """LongLive + HPSv3 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def longlive_video_hpsv3_with_lora_init_24gpu():
    """LongLive + HPSv3 + LoRA 初始化 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_with_lora_init_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.longlive_lora_init = "./checkpoints/longlive_models/models/lora.pt"
    return config


# ============================================================================
# 48 GPU
# ============================================================================

def longlive_video_hpsv3_48gpu():
    """LongLive + HPSv3 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_48gpu",
    )
    config.beta = 0.1
    config.eval_freq = 30
    config.train.learning_rate = 1e-5
    return config


def longlive_video_multireward_48gpu():
    """LongLive + 多奖励 (MQ + HPSv3) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_multireward_48gpu",
    )
    config.beta = 0.1
    config.eval_freq = 30
    config.train.learning_rate = 1e-5
    return config


def longlive_video_hpsv3_with_lora_init_48gpu():
    """LongLive + HPSv3 + LoRA 初始化 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_with_lora_init_48gpu",
    )
    config.beta = 0.1
    config.eval_freq = 30
    config.train.learning_rate = 1e-5
    config.longlive_lora_init = "./checkpoints/longlive_models/models/lora.pt"
    return config


# ============================================================================
# Streaming RL (长视频 streaming 训练)
# ============================================================================

def longlive_video_hpsv3_streaming_with_lora_init():
    """LongLive + HPSv3 + Streaming RL (960 帧) + LoRA 初始化 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_streaming_with_lora_init",
    )

    # streaming 训练脚本需要读取 casual_wan_inference.yaml
    config.config_path = "configs/casual_wan_inference.yaml"

    # Streaming 训练参数
    config.streaming.enabled = True
    config.streaming.window_size = 21       # 训练窗口 (latent 帧)
    config.streaming.sample_frames = 240    # 最大采样帧数 (latent)
    config.streaming.window_selection = "random_choice"
    config.streaming.window_choices = [21,]

    # Window-level reward: 每个 epoch 固定 window_start
    config.train.num_inner_epochs = 1

    # 视频规格
    config.num_frames = 960                 # pixel 帧数 (最大值，实际按需采样)
    config.height = 480
    config.width = 832
    config.eval_num_frames = 240           # 评估时采样帧数，完整 960 帧太慢

    # NFT 参数
    config.beta = 0.1
    config.train.beta = 0.0001
    config.train.learning_rate = 1e-5
    config.eval_freq = 30

    # LoRA 初始化
    config.longlive_lora_init = "./checkpoints/longlive_models/models/lora.pt"

    return config


# ============================================================================
# GARDO (use_select_kl=True) — requires multi-reward
# ============================================================================

def longlive_video_hpsv3_gardo():
    """LongLive + GARDO (HPSv3 + MQ) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_gardo",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.use_select_kl = True
    config.main_reward = "video_hpsv3_local"
    return config


def longlive_video_hpsv3_gardo_24gpu():
    """LongLive + GARDO (HPSv3 + MQ) - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_gardo_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.use_select_kl = True
    config.main_reward = "video_hpsv3_local"
    return config


def longlive_video_hpsv3_gardo_48gpu():
    """LongLive + GARDO (HPSv3 + MQ) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="longlive_video_hpsv3_gardo_48gpu",
    )
    config.beta = 0.1
    config.eval_freq = 30
    config.train.learning_rate = 1e-5
    config.use_select_kl = True
    config.main_reward = "video_hpsv3_local"
    return config
