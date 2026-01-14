#!/bin/bash
# ============================================================
# 01: Encoder Training - Single Audio Debug
# ============================================================
# Trains Gabor atoms on a single audio file to verify the
# AudioGS pipeline works correctly. Use this for debugging.
# ============================================================

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="configs/AudioGS_config.yaml"
TARGET_FILE="${1:-"data/raw/LibriTTS_R/train/train-clean-100/19/198/19_198_000000_000002.wav"}"
MAX_ITERS="${MAX_ITERS:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/debug}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - Single Audio Debug Training"
echo "============================================================"
echo "Config: $CONFIG"
echo "Target: ${TARGET_FILE:-'Random sample from dataset'}"
echo "Max iterations: $MAX_ITERS"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU python scripts/01_encoder_training/run_encoder_train.py \
    --config "$CONFIG" \
    ${TARGET_FILE:+--target_file "$TARGET_FILE"} \
    --max_iters "$MAX_ITERS" \
    --output_dir "$OUTPUT_DIR"

echo "============================================================"
echo "Debug training complete! Check $OUTPUT_DIR for results."
echo "============================================================"
