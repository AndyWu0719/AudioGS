#!/bin/bash
# ============================================================
# 01: Codec Training (AE/VAE) - Multi-GPU DDP
# ============================================================
# Trains the Stage01 Gabor-frame codec on multiple GPUs.
#
# Usage:
#   bash scripts/run/01_encoder_ddp.sh
#
# Env:
#   CONFIG       Codec config (default: configs/codec_config.yaml)
#   NUM_GPUS     Number of GPUs (default: 4)
#   MASTER_PORT  torchrun port (default: 29510)
#   VAL_RATIO    Validation split ratio (default: 0.01)
#   RESUME       Path to checkpoint to resume (optional)
#   CUDA_VISIBLE_DEVICES  Optional, restrict GPUs (e.g. "2,3")
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/codec_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29510}"
VAL_RATIO="${VAL_RATIO:-0.01}"
RESUME="${RESUME:-}"

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

echo "============================================================"
echo "AudioGS - Stage01 Codec Training (DDP)"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Val ratio: $VAL_RATIO"
echo "Resume: ${RESUME:-disabled}"
echo "============================================================"

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  scripts/01_encoder_training/run_encoder_train.py \
  --config "$CONFIG" \
  --val_ratio "$VAL_RATIO" \
  "${RESUME_ARGS[@]}"

echo "============================================================"
echo "Stage01 training complete!"
echo "============================================================"
