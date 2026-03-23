import os
import imp

_base = imp.load_source("_base_clean", os.path.join(os.path.dirname(__file__), "_base_clean.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, gradient_step_per_epoch=1, dataset="vidprom", reward_fn={}, name=""):
    """Casual Forcing Chunkwise (wan_casual_chunk) 专用配置构建函数"""
    config = _base._make_base_config(dataset)
    config.base_model = "wan_casual_chunk"

    _base._apply_wan_common(config)

    config.pretrained.model = "./checkpoints/casualforcing/chunkwise/causal_forcing.pt"
    config.model_kwargs = {"timestep_shift": 5.0}
    config.sample.num_frame_per_block = 3

    gpu_config = _base._get_gpu_config("wan", n_gpus)
    bsz = gpu_config["bsz"]
    config.sample.num_image_per_prompt = gpu_config["num_image_per_prompt"]
    num_groups = gpu_config["num_groups"]
    config.sample.test_batch_size = gpu_config["test_batch_size"]

    _base._apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch)
    _base._apply_common_fields(config, "wan_casual_chunk", dataset, reward_fn, name)
    return config


# ============================================================================
# 8 GPU
# ============================================================================

def casual_forcing_video_hpsv3():
    """Casual Forcing + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3",
    )
    config.beta = 0.1
    config.eval_freq = 30
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_multi_reward():
    """Casual Forcing + 多奖励 (HPSv3 + MQ) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0, "videoalign_mq_score": 1.0},
        name="casual_forcing_video_multi_reward",
    )
    config.beta = 0.1
    config.train.beta = 0.0001
    config.eval_freq = 30
    config.train.learning_rate = 2e-5
    return config


# ============================================================================
# 16 GPU
# ============================================================================

def casual_forcing_video_hpsv3_16gpu():
    """Casual Forcing + HPSv3 - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_multi_reward_16gpu():
    """Casual Forcing + 多奖励 (MQ + HPSv3) - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="casual_forcing_video_multi_reward_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 24 GPU
# ============================================================================

def casual_forcing_video_hpsv3_24gpu():
    """Casual Forcing + HPSv3 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 48 GPU
# ============================================================================

def casual_forcing_video_hpsv3_48gpu():
    """Casual Forcing + HPSv3 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_multireward_48gpu():
    """Casual Forcing + 多奖励 (HPSv3 + MQ) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0, "videoalign_mq_score": 1.0},
        name="casual_forcing_video_multireward_48gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.0005
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_ta_48gpu():
    """Casual Forcing + Temporal Alignment - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0},
        name="casual_forcing_video_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 2e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_hpsv3_mq_ta_48gpu():
    """Casual Forcing + HPSv3 + MQ + TA - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0, "videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_mq_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_hpsv3_ta_48gpu():
    """Casual Forcing + TA + HPSv3 (0.1 权重) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0, "video_hpsv3_local": 0.1},
        name="casual_forcing_video_hpsv3_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config
