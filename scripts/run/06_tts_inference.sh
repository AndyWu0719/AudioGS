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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="${1:-logs/flow_latent/checkpoints/final.pt}"
FLOW_CONFIG="${FLOW_CONFIG:-configs/flow_config.yaml}"
TEXT="${2:-"Hello, this is a test of the audio Gaussian splatting text to speech system."}"
OUTPUT="${3:-output.wav}"
SPEAKER_ID="${SPEAKER_ID:-0}"
STEPS="${STEPS:-25}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - TTS Inference"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $FLOW_CONFIG"
echo "Text: $TEXT"
echo "Output: $OUTPUT"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/06_tts_inference/run_tts_inference.py \
    --flow_ckpt "$CHECKPOINT" \
    --flow_config "$FLOW_CONFIG" \
    --text "$TEXT" \
    --speaker_id "$SPEAKER_ID" \
    --steps "$STEPS" \
    --out_wav "$OUTPUT"

echo "Inference complete! Audio saved to: $OUTPUT"
