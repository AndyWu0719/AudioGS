#!/bin/bash
# ============================================================
# AGS TTS Evaluation
# ============================================================
# Evaluate TTS quality using WER and Speaker Similarity.
#
# Usage:
#   ./scripts/run/32_eval_tts.sh                     # Default 50 samples
#   ./scripts/run/32_eval_tts.sh --samples 100       # Custom sample count
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT="${CHECKPOINT:-logs/flow/best.pt}"
ATOM_DIR="${ATOM_DIR:-data/atoms/LibriTTS_R/train/train-clean-100}"
TEST_DATA="/data0/determined/users/andywu/GS-TS/data/raw/LibriTTS_R/test/test-clean"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
WHISPER_MODEL="${WHISPER_MODEL:-base}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --whisper)
            WHISPER_MODEL="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Activate conda
source ~/.bashrc
conda activate qwen2_CALM

echo "============================================================"
echo "AGS TTS Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Atom Dir: $ATOM_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Whisper: $WHISPER_MODEL"
echo "============================================================"

python scripts/04_inference_eval/run_eval_tts.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$ATOM_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --whisper_model "$WHISPER_MODEL"

echo ""
echo "============================================================"
echo "TTS Evaluation Complete!"
echo "============================================================"
