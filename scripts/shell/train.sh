#!/bin/bash


export WANDB_API_KEY=REDACTED_WANDB_KEY
export WANDB_ENTITY=1241400738
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_TOKEN=REDACTED_HF_TOKEN


torchrun --nproc_per_node=8 scripts/train_nft_wan.py \
--config configs/nft_longlive.py:longlive_video_hpsv3_gardo

# torchrun --nproc_per_node=8 scripts/train_nft_wan_streaming.py \
# --config configs/nft_clean.py:longlive_video_hpsv3_streaming_960frames_random_choice

# torchrun --nproc_per_node=8 scripts/train_nft_wan.py \
#       --config configs/nft_clean.py:krea14b_video_hpsv3