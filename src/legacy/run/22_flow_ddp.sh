#!/bin/bash
# ============================================================
# 22: Flow Training - Multi-GPU DDP
# ============================================================
# Trains the Flow Matching DiT model on multiple GPUs using
# DistributedDataParallel for ~4x speedup.
#
# Effective batch size = per_gpu_batch × num_gpus
# For 4 GPUs with batch_size=4: effective = 16
#
# RESUME TRAINING:
#   Add --resume flag and optionally --checkpoint <path>
#   Example: torchrun ... train_ddp.py --config ... --resume
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="${CONFIG:-configs/flow_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
EXP_NAME="${EXP_NAME:-}"
RESUME="${RESUME:-}"  # Set to "1" or path to enable resume

echo "============================================================"
echo "Flow Matching - Multi-GPU DDP Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Resume: ${RESUME:-disabled}"
echo "============================================================"

# Build resume args
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
    scripts/03_flow_training/train_ddp.py \
    --config "$CONFIG" \
    ${EXP_NAME:+--exp_name "$EXP_NAME"} \
    $RESUME_ARGS

echo "============================================================"
echo "DDP Training complete!"
echo "============================================================"
