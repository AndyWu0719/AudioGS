#!/bin/bash
# ============================================================
# 03: Encoder Evaluation
# ============================================================
# Evaluates codec reconstruction quality (PESQ, SI-SDR, MSS, SNR)
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-logs/codec/checkpoints/final.pt}"
CODEC_CONFIG="${CODEC_CONFIG:-configs/codec_config.yaml}"
DATA_DIR="${2:-data/raw/LibriTTS_R/dev/dev-clean}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
GPU="${GPU:-0}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-0}"

echo "============================================================"
echo "AudioGS - Codec Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Codec config: $CODEC_CONFIG"
echo "Data dir: $DATA_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Segment seconds: $SEGMENT_SECONDS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/03_encoder_eval/run_encoder_eval.py \
    --checkpoint "$CHECKPOINT" \
    --codec_config "$CODEC_CONFIG" \
    --data_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --segment_seconds "$SEGMENT_SECONDS"

echo "============================================================"
echo "Evaluation complete!"
echo "============================================================"
