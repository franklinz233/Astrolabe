#!/bin/bash

# torchrun --nproc_per_node=8 scripts/inference_wan.py \
#         --base_model checkpoints/krea-realtime-video-14b.safetensors \
#         --lora_path logs/nft/krea14b/krea14b_video_hpsv3_48gpu_2026.03.13_23.26.08/checkpoints/checkpoint-330 \
#         --num_frames 480 \
#         --prompt_file prompts/MovieGenVideoBench_extended.txt \
#         --output_dir outputs/krea48-330-480

torchrun --nproc_per_node=8 scripts/inference_wan.py \
        --base_model checkpoints/krea-realtime-video-14b.safetensors \
        --lora_path logs/nft/krea14b/krea14b_video_hpsv3_48gpu_2026.03.13_23.26.08/checkpoints/checkpoint-330 \
        --num_frames 960 \
        --prompt_file prompts/interactive_benchmark.txt \
        --output_dir outputs/krea48-330-960-switch