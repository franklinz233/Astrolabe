import os
import imp

_base = imp.load_source("_base_clean", os.path.join(os.path.dirname(__file__), "_base_clean.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, gradient_step_per_epoch=1, dataset="vidprom", reward_fn={}, name=""):
    """Self Forcing (wan_self) 专用配置构建函数"""
    config = _base._make_base_config(dataset)
    config.base_model = "wan_self"

    _base._apply_wan_common(config)

    config.pretrained.model = "./checkpoints/self_forcing_dmd.pt"
    config.model_kwargs = {"timestep_shift": 5.0}

    gpu_config = _base._get_gpu_config("wan", n_gpus)
    bsz = gpu_config["bsz"]
    config.sample.num_image_per_prompt = gpu_config["num_image_per_prompt"]
    num_groups = gpu_config["num_groups"]
    config.sample.test_batch_size = gpu_config["test_batch_size"]

    _base._apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch)
    _base._apply_common_fields(config, "wan_self", dataset, reward_fn, name)
    return config


# ============================================================================
# 8 GPU
# ============================================================================

def self_forcing_video_hpsv3():
    """Self Forcing + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="self_forcing_video_hpsv3",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def self_forcing_video_multi_reward():
    """Self Forcing + 多奖励 (MQ + HPSv3) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="self_forcing_video_multi_reward",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 16 GPU
# ============================================================================

def self_forcing_video_hpsv3_16gpu():
    """Self Forcing + HPSv3 - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="self_forcing_video_hpsv3_16gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.001
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def self_forcing_video_multi_reward_16gpu():
    """Self Forcing + 多奖励 (MQ + HPSv3) - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="self_forcing_video_multi_reward_16gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.001
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


# ============================================================================
# 24 GPU
# ============================================================================

def self_forcing_video_hpsv3_24gpu():
    """Self Forcing + HPSv3 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="self_forcing_video_hpsv3_24gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.001
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def self_forcing_video_multi_reward_24gpu():
    """Self Forcing + 多奖励 (MQ + HPSv3) - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="self_forcing_video_multi_reward_24gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.001
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


# ============================================================================
# 48 GPU
# ============================================================================

def self_forcing_video_hpsv3_48gpu():
    """Self Forcing + HPSv3 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="self_forcing_video_hpsv3_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def self_forcing_video_multi_reward_48gpu():
    """Self Forcing + 多奖励 (MQ + HPSv3) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="self_forcing_video_multi_reward_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config
