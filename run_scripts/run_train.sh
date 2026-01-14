#!/bin/bash
# DDP Training Launcher
# Usage: ./run_train.sh [exp_name]

EXP_NAME=${1:-"logs/encoder_v1"}

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

# Launch
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    scripts/02_encoder_training/train_encoder.py \
    --config configs/AudioGS_config.yaml \
    --exp_name $EXP_NAME \
    --use_wandb
