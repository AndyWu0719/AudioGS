#!/bin/bash
# ============================================================
# 21: Flow Training - Single GPU
# ============================================================
# Trains the Flow Matching DiT model on a single GPU.
# Use 22_flow_ddp.sh for multi-GPU training (faster).
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="${CONFIG:-configs/flow_config.yaml}"
GPU="${GPU:-0}"
EXP_NAME="${EXP_NAME:-}"

echo "============================================================"
echo "Flow Matching - Single GPU Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPU: $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/03_flow_training/train_single.py \
    --config "$CONFIG" \
    ${EXP_NAME:+--exp_name "$EXP_NAME"}

echo "============================================================"
echo "Training complete!"
echo "============================================================"
