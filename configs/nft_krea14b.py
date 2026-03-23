import os
import imp

_base = imp.load_source("_base_clean", os.path.join(os.path.dirname(__file__), "_base_clean.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, gradient_step_per_epoch=1, dataset="vidprom", reward_fn={}, name=""):
    """Krea 14B 专用配置构建函数"""
    config = _base._make_base_config(dataset)
    config.base_model = "krea14b"

    _base._apply_wan_common(config)

    config.pretrained.model = "./checkpoints/krea-realtime-video-14b.safetensors"
    config.is_krea_14b = True
    config.timestep_shift = 5.0
    config.model_kwargs = {"timestep_shift": 5.0, "local_attn_size": -1, "sink_size": 0}

    gpu_config = _base._get_gpu_config("krea14b", n_gpus)
    bsz = gpu_config["bsz"]
    config.sample.num_image_per_prompt = gpu_config["num_image_per_prompt"]
    num_groups = gpu_config["num_groups"]
    config.sample.test_batch_size = gpu_config["test_batch_size"]

    _base._apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch)
    _base._apply_common_fields(config, "krea14b", dataset, reward_fn, name)
    return config


# ============================================================================
# 8 GPU
# ============================================================================

def krea14b_video_hpsv3():
    """Krea 14B + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="krea14b_video_hpsv3",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def krea14b_video_multi_reward():
    """Krea 14B + 多奖励 (HPSv3 + MQ) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0, "videoalign_mq_score": 1.0},
        name="krea14b_video_multi_reward",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


# ============================================================================
# 16 GPU
# ============================================================================

def krea14b_video_hpsv3_16gpu():
    """Krea 14B + HPSv3 - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="krea14b_video_hpsv3_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


# ============================================================================
# 24 GPU
# ============================================================================

def krea14b_video_hpsv3_24gpu():
    """Krea 14B + HPSv3 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="krea14b_video_hpsv3_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def krea14b_video_multi_reward_24gpu():
    """Krea 14B + 多奖励 (HPSv3 + MQ) - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0, "videoalign_mq_score": 1.0},
        name="krea14b_video_multi_reward_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


# ============================================================================
# 48 GPU
# ============================================================================

def krea14b_video_hpsv3_48gpu():
    """Krea 14B + HPSv3 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="krea14b_video_hpsv3_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config
