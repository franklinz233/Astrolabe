
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Tuple
from easydict import EasyDict as edict

# 自动定位 SEA-RAFT 路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEA_RAFT_CORE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, 'reward_models/sea_raft/core'))
if SEA_RAFT_CORE_PATH not in sys.path:
    sys.path.insert(0, SEA_RAFT_CORE_PATH) # 插入到最前面

# 直接导入，不要写 from core.raft
from raft import RAFT
from raft_utils.utils import load_ckpt, InputPadder


class DynamicDegreeScorer(nn.Module):
    """
    简化版动态度评分器(仅用于训练 reward)
    
    归一化公式（来自 Krea 论文）：
    m = min{1, sqrt(u² + v²) / (sigma * sqrt(H² + W²))}
    
    评估维度（仅两项）：
    1. Motion Magnitude (运动幅度)：适度运动得分高，静止/过度运动得分低
    2. Temporal Consistency (时间一致性)：平滑过渡得分高，突变/闪烁得分低
    
    最终 reward = magnitude_weight * magnitude_reward + temporal_weight * temporal_reward
    """
    
    def __init__(
        self,
        device,
        model_path=None,
        dtype=torch.float32,
        # 归一化参数
        sigma: float = 0.15,                    # 归一化系数（Krea 默认值）
        clip_normalized: bool = True,            # 是否将 m 限制在 [0, 1]
        # 权重配置
        magnitude_weight: float = 1.5,          # 运动幅度权重
        temporal_weight: float = 1,           # 时间一致性权重
        # 运动幅度参数（归一化后的目标值，范围 [0, 1]）
        magnitude_target: float = 0.6,          # 理想运动幅度
        magnitude_tolerance: float = 0.2,       # 容忍范围
        magnitude_max: float = 1.2,             # 最大合理值
        # 时间一致性参数
        temporal_smoothness_sigma: float = 0.1,  # 平滑度参数
        # SEA-RAFT 推理参数
        iters: int = 4,                         # 光流迭代次数
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # 归一化参数
        self.sigma = sigma
        self.clip_normalized = clip_normalized
        
        # 权重
        self.magnitude_weight = magnitude_weight
        self.temporal_weight = temporal_weight
        
        # 运动幅度参数
        self.magnitude_target = magnitude_target
        self.magnitude_tolerance = magnitude_tolerance
        self.magnitude_max = magnitude_max
        
        # 时间一致性参数
        self.temporal_smoothness_sigma = temporal_smoothness_sigma
        
        # SEA-RAFT 配置
        args = edict({
            "dim": 128,
            "radius": 4,
            "use_var": True,
            "var_min": 0,
            "var_max": 10,
            "scale": 0,
            "model": "./reward_ckpts/sea-raft/model.safetensor",
            "initial_dim": 64,
            "block_dims": [64, 128, 256],
            "pretrain": "resnet34",
            "num_blocks": 2,
            "iters": iters,
            "url": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
        })
        
        self.args = args
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.model = RAFT(args)
            load_ckpt(self.model, model_path)
        else:
            self.model = RAFT.from_pretrained(args.url, args=args)
            
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    def normalize_flow_magnitude(self, magnitude, H, W):
        """
        使用对角线归一化光流幅度
        
        公式：m = min{1, sqrt(u² + v²) / (σ * sqrt(H² + W²))}
        
        Args:
            magnitude: [..., H, W] 光流幅度 sqrt(u² + v²)
            H, W: 图像高度和宽度
            
        Returns:
            normalized_magnitude: 归一化后的幅度
        """
        # 计算对角线长度
        diagonal = torch.sqrt(torch.tensor(H ** 2 + W ** 2, dtype=torch.float32, device=magnitude.device))
        
        # 归一化因子
        normalization_factor = self.sigma * diagonal
        
        # 归一化
        normalized = magnitude / normalization_factor
        
        # 限制在 [0, 1]
        if self.clip_normalized:
            normalized = torch.clamp(normalized, min=0, max=1)
        
        return normalized

    def get_top_percentile_mean(self, magnitude, percentile=95):
        """
        计算 top-k% 的平均幅度（过滤噪声）
        
        Args:
            magnitude: [B, T, H, W]
            percentile: 百分位数（95 表示 top 5%）
            
        Returns:
            top_mean: [B, T]
        """
        B, T, H, W = magnitude.shape
        mag_flat = magnitude.view(B, T, -1)
        
        # 计算 top-k
        total_pixels = H * W
        k = max(1, int(total_pixels * (100 - percentile) / 100))
        
        top_k_mag, _ = torch.topk(mag_flat, k, dim=-1)  # [B, T, k]
        top_mean = top_k_mag.mean(dim=-1)  # [B, T]
        
        return top_mean

    def compute_optical_flow(self, videos):
        """
        计算光流并归一化
        
        Args:
            videos: [B, C, T, H, W], 范围 [0, 1]
            
        Returns:
            normalized_magnitude: [B, T-1, H, W] 归一化后的幅度
        """
        B, C, T, H, W = videos.shape
        
        if T < 2:
            return torch.zeros(B, 0, H, W, device=self.device, dtype=self.dtype)
        
        # 转换为 [0, 255]
        videos = videos.to(dtype=self.dtype)
    
        # 准备相邻帧 [B*(T-1), C, H, W]
        frames1 = videos[:, :, :-1, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).contiguous()
        frames2 = videos[:, :, 1:, :, :].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).contiguous()
        
        # SEA-RAFT 推理
        padder = InputPadder(frames1.shape)
        f1, f2 = padder.pad(frames1, frames2)

        # with torch.cuda.amp.autocast(enabled=False):
        output = self.model(f1, f2, iters=self.args.iters, test_mode=True)

        flow_up = padder.unpad(output['final'])
        
        # 重塑为 [B, T-1, 2, H, W]
        flows = flow_up.view(B, T-1, 2, H, W)
        
        # 计算幅度：sqrt(u² + v²)
        magnitude = torch.sqrt(flows[:, :, 0]**2 + flows[:, :, 1]**2)  # [B, T-1, H, W]
        
        # 使用对角线归一化
        normalized_magnitude = self.normalize_flow_magnitude(magnitude, H, W)
        
        return normalized_magnitude

    def compute_magnitude_reward(self, normalized_magnitude):
        """
        运动幅度奖励
        
        目标：奖励适度运动，惩罚静止和过度运动
        使用高斯形状的奖励函数
        
        Args:
            normalized_magnitude: [B, T, H, W] 归一化后的光流幅度
            
        Returns:
            reward: [B] 运动幅度奖励，范围 [0, 1]
        """
        B, T, H, W = normalized_magnitude.shape
        
        # 计算 top-5% 平均幅度（过滤背景噪声）
        avg_magnitude = self.get_top_percentile_mean(normalized_magnitude, percentile=95)  # [B, T]
        
        # 视频整体平均
        video_magnitude = avg_magnitude.mean(dim=-1)  # [B]
        
        # 高斯奖励函数：以 target 为中心
        # reward = exp(-((m - target)² / (2 * tolerance²)))
        reward = torch.exp(
            -((video_magnitude - self.magnitude_target) ** 2) / 
            (2 * self.magnitude_tolerance ** 2)
        )
        
        # 过度运动额外惩罚
        if self.magnitude_max > 0:
            over_motion_penalty = torch.clamp(
                (video_magnitude - self.magnitude_max) / self.magnitude_max, 
                min=0, max=1
            )
            reward = reward * (1 - over_motion_penalty)
        
        # 静止惩罚（确保完全静止的视频得分低）
        static_penalty = torch.exp(-video_magnitude * 3.0)
        reward = reward * (1 - 0.5 * static_penalty)
        
        return reward

    def compute_temporal_reward(self, normalized_magnitude):
        """
        时间一致性奖励
        
        目标：奖励平滑运动，惩罚闪烁/突变
        
        Args:
            normalized_magnitude: [B, T, H, W]
            
        Returns:
            reward: [B] 时间一致性奖励，范围 [0, 1]
        """
        B, T, H, W = normalized_magnitude.shape
        
        if T < 2:
            return torch.ones(B, device=normalized_magnitude.device)
        
        # 计算每帧的平均幅度
        frame_magnitude = normalized_magnitude.mean(dim=(2, 3))  # [B, T]
        
        # 计算相邻帧幅度差异（"加速度"）
        mag_diff = torch.abs(frame_magnitude[:, 1:] - frame_magnitude[:, :-1])  # [B, T-1]
        avg_acceleration = mag_diff.mean(dim=-1)  # [B]
        
        # 指数衰减奖励：加速度越大，奖励越低
        # reward = exp(-acceleration / sigma)
        reward = torch.exp(-avg_acceleration / self.temporal_smoothness_sigma)
        
        return reward

    @torch.no_grad()
    def forward(self, videos, return_details=False):
        """
        前向传播
        
        Args:
            videos: [B, C, T, H, W], 范围 [0, 1]
            return_details: 是否返回详细信息
            
        Returns:
            如果 return_details=False: [B] 综合 reward
            如果 return_details=True: dict 包含各维度得分
        """
        B, C, T, H, W = videos.shape
        
        if T < 2:
            if return_details:
                return {
                    'reward': torch.zeros(B, device=self.device),
                    'magnitude_reward': torch.zeros(B, device=self.device),
                    'temporal_reward': torch.zeros(B, device=self.device),
                    'normalized_magnitude_mean': torch.zeros(B, device=self.device),
                }
            return torch.zeros(B, device=self.device)
        
        # 计算光流（已归一化）
        normalized_magnitude = self.compute_optical_flow(videos)
        
        # 计算两个维度的 reward
        mag_reward = self.compute_magnitude_reward(normalized_magnitude)
        temp_reward = self.compute_temporal_reward(normalized_magnitude)
        
        # 加权综合
        total_reward = (
            self.magnitude_weight * mag_reward + 
            self.temporal_weight * temp_reward
        )
        
        if return_details:
            # 额外统计信息
            avg_normalized_mag = self.get_top_percentile_mean(
                normalized_magnitude, percentile=95
            ).mean(dim=1)  # [B]
            
            return {
                'reward': total_reward,
                'magnitude_reward': mag_reward,
                'temporal_reward': temp_reward,
                'normalized_magnitude_mean': avg_normalized_mag,
            }
        else:
            return total_reward

    def __call__(self, videos, return_details=False):
        """简化调用接口"""
        return self.forward(videos, return_details)



if __name__ == "__main__":
    example_usage()
