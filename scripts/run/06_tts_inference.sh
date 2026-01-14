#!/bin/bash
# ============================================================
# 06: TTS Inference
# ============================================================
# Generate audio from text using trained Flow Matching model.
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-logs/flow/best.pt}"
TEXT="${2:-"Hello, this is a test of the audio Gaussian splatting text to speech system."}"
OUTPUT="${3:-output.wav}"
SPEAKER_ID="${SPEAKER_ID:-0}"
STEPS="${STEPS:-25}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - TTS Inference"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Text: $TEXT"
echo "Output: $OUTPUT"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/06_tts_inference/run_tts_inference.py \
    --checkpoint "$CHECKPOINT" \
    --text "$TEXT" \
    --speaker_id "$SPEAKER_ID" \
    --steps "$STEPS" \
    --output "$OUTPUT"

echo "Inference complete! Audio saved to: $OUTPUT"
