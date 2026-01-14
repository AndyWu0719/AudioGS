#!/bin/bash
# ============================================================
# 05: Flow/TTS Evaluation
# ============================================================
# Evaluates TTS quality using WER and Speaker Similarity.
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-logs/flow/best.pt}"
DATA_DIR="${2:-data/atoms/LibriTTS_R/train/train-clean-100}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - TTS Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Data dir: $DATA_DIR"
echo "Samples: $NUM_SAMPLES"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/05_flow_eval/run_flow_eval.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES"

echo "Evaluation complete!"
