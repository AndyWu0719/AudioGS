#!/bin/bash
# ============================================================
# 03: Encoder Evaluation
# ============================================================
# Evaluates atom reconstruction quality (PESQ, STOI, SI-SDR, MCD)
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${1:-data/atoms/LibriTTS_R/train/train-clean-100}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - Encoder Evaluation"
echo "============================================================"
echo "Data dir: $DATA_DIR"
echo "Max samples: $MAX_SAMPLES"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/03_encoder_eval/run_encoder_eval.py \
    --data_dir "$DATA_DIR" \
    --max_samples "$MAX_SAMPLES"

echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
