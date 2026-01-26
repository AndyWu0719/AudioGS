#!/bin/bash
# ============================================================
# 04: Flow Training - Multi-GPU DDP
# ============================================================
# Trains the Flow Matching DiT model on multiple GPUs using
# DistributedDataParallel for ~4x speedup.
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/flow_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXP_NAME="${EXP_NAME:-}"
RESUME="${RESUME:-}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "============================================================"
echo "Flow Matching - Multi-GPU DDP Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
fi
echo "Resume: ${RESUME:-disabled}"
echo "============================================================"

RESUME_ARGS=""
if [[ -n "$RESUME" ]]; then
    RESUME_ARGS="--resume $RESUME"
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/04_flow_training/run_flow_train.py \
    --config "$CONFIG" \
    $RESUME_ARGS

echo "DDP Training complete!"
