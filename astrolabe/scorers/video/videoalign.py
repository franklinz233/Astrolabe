import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from astrolabe.reward_models.videoalign.wan_inference import VideoVLMRewardInference
from astrolabe import VIDEOALIGN_CKPT_PATH

class VideoAlignScorer(nn.Module):
    def __init__(self, model_path=VIDEOALIGN_CKPT_PATH, device='cuda', dtype=torch.bfloat16, reward_type="MQ", use_grayscale=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.reward_type = reward_type  # 新增：存储需要的 reward 类型
        self.use_grayscale = use_grayscale
        
        # 检查 reward_type 是否合法
        valid_types = ["VQ", "MQ", "TA", "Overall"]
        assert self.reward_type in valid_types, f"reward_type must be one of {valid_types}"

        self.scorer = VideoVLMRewardInference(
            load_from_pretrained=model_path,
            device=device,
            dtype=dtype
        )

    def __call__(self, videos, prompts):
        """
        Args:
            videos: List[torch.Tensor] or torch.Tensor.
            prompts: List[str]
        """
        # 兼容处理：如果是 Tensor [B, T, C, H, W]，先转为 List
        if isinstance(videos, torch.Tensor):
            if videos.dtype != torch.uint8:
                videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)
            videos = [v for v in videos]
        
        if self.use_grayscale:
            processed_videos = []
            for v in videos: # v 的形状是 [T, C, H, W]
                # F.rgb_to_grayscale 可以处理 (..., C, H, W) 形状的 Tensor，其中 C=3
                # 这里对 [T, 3, H, W] 形状的 v 进行灰度转换，会得到 [T, num_output_channels, H, W]
                processed_videos.append(F.rgb_to_grayscale(v, num_output_channels=3))
            videos = processed_videos
            

        # reward_from_frames 返回字典: {'VQ': ..., 'MQ': ..., 'TA': ..., 'Overall': ...}
        results = self.scorer.reward_from_frames(videos, prompts, use_norm=True)
        
        # [修改] 根据 self.reward_type 获取对应的 Tensor
        scores = results[self.reward_type]
        
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, device=self.device, dtype=self.dtype)
        else:
            scores = scores.to(dtype=self.dtype, device=self.device)
            
        return scores



def main():
    import torchvision
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16)
    
    # 测试使用本地视频文件
    video_path = "./videos/self_forcing_dmd/A 3D animation of a small, round, fluffy creature with big, expressive eyes exploring a vibrant, enc-0.mp4"
    
    if not os.path.exists(video_path):
        print(f"警告: 找不到测试视频 {video_path}, 使用随机张量测试")
        # 创建随机视频张量 [1, 10, 3, 480, 832]
        videos = torch.randint(0, 256, (1, 10, 3, 480, 832), dtype=torch.uint8).float() / 255.0
        prompts = ["A test video prompt"]
    else:
        # 加载视频
        filename = os.path.basename(video_path)
        prompt = filename.split(", enc-")[0]
        prompts = [prompt]
        
        # 使用 torchvision.io.read_video 加载视频
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format='THWC')
        # [F, H, W, C] -> [F, C, H, W]
        video_frames = video_frames.permute(0, 3, 1, 2)
        # 添加 batch 维度
        videos = video_frames.unsqueeze(0).float() / 255.0
        print(f"加载视频: {filename}")
        print(f"Prompt: {prompt}")
        print(f"视频形状: {videos.shape}")
    
    # 计算分数
    with torch.no_grad():
        scores = scorer(videos, prompts)
        print(f"VideoAlign Overall Score: {scores.item():.4f}")


if __name__ == "__main__":
    main()
