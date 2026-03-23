#!/bin/bash

torchrun --nproc_per_node=8 scripts/train_nft_wan.py \
      --config configs/nft_clean.py:krea14b_video_hpsv3