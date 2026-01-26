#!/bin/bash
# ============================================================
# 05: Flow Evaluation (paired)
# ============================================================
# Samples audio from Flow-on-latents and (optionally) compares to paired GT.
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-logs/flow_latent/checkpoints/final.pt}"
FLOW_CONFIG="${FLOW_CONFIG:-configs/flow_config.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - Flow Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $FLOW_CONFIG"
echo "Samples: $NUM_SAMPLES"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/05_flow_eval/run_flow_eval.py \
    --flow_ckpt "$CHECKPOINT" \
    --flow_config "$FLOW_CONFIG" \
    --num_samples "$NUM_SAMPLES" \
    --out_dir "${OUT_DIR:-logs/flow_eval}"

echo "Evaluation complete!"
