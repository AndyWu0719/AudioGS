#!/bin/bash
# ============================================================
# 01: Codec Training (AE/VAE) - Debug
# ============================================================
# Trains the Stage01 Gabor-frame codec on a small validation split.
# ============================================================

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="configs/codec_config.yaml"
VAL_RATIO="${VAL_RATIO:-0.01}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - Codec Training (Debug)"
echo "============================================================"
echo "Config: $CONFIG"
echo "Val ratio: $VAL_RATIO"
echo "GPU: $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/01_encoder_training/run_encoder_train.py \
    --config "$CONFIG" \
    --val_ratio "$VAL_RATIO"

echo "============================================================"
echo "Debug training complete!"
echo "============================================================"
