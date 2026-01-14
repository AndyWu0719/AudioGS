#!/bin/bash
# ============================================================
# 14: Atom Fitting - Batch Dataset Preprocessing
# ============================================================
# Converts raw audio files into Gabor atom parameters using
# the optimized AudioGS pipeline for the entire dataset.
#
# Input:  LibriTTS-R audio files
# Output: data/atoms/LibriTTS_R/train/train-clean-*/*.pt
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="${CONFIG:-configs/AudioGS_config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"

echo "============================================================"
echo "AudioGS - Dataset Preprocessing"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "============================================================"

python scripts/02_data_prep/generate_dataset.py \
    --config "$CONFIG" \
    --num_gpus "$NUM_GPUS" \
    --num_workers_per_gpu "$WORKERS_PER_GPU" \
    # --resume

echo "============================================================"
echo "Preprocessing complete!"
echo "============================================================"
