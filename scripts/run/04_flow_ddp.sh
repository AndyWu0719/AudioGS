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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/flow_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXP_NAME="${EXP_NAME:-}"
RESUME="${RESUME:-}"

echo "============================================================"
echo "Flow Matching - Multi-GPU DDP Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Resume: ${RESUME:-disabled}"
echo "============================================================"

RESUME_ARGS=""
if [[ -n "$RESUME" ]]; then
    if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
        RESUME_ARGS="--resume"
    else
        RESUME_ARGS="--resume --checkpoint $RESUME"
    fi
fi

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/04_flow_training/run_flow_train.py \
    --config "$CONFIG" \
    ${EXP_NAME:+--exp_name "$EXP_NAME"} \
    $RESUME_ARGS

echo "DDP Training complete!"
