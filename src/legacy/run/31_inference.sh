#!/bin/bash
# ============================================================
# AGS TTS Inference
# ============================================================
# Generate audio from text using trained Flow Matching model.
#
# Usage:
#   ./scripts/run/31_inference.sh                    # Random sample from test-clean
#   ./scripts/run/31_inference.sh "Hello world"      # Custom text
#   ./scripts/run/31_inference.sh --random 5         # 5 random samples
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT="${CHECKPOINT:-logs/flow/flow_ddp4_20260113_141444/checkpoints/best.pt}"
TEST_DATA="/data0/determined/users/andywu/GS-TS/data/raw/LibriTTS_R/test/test-clean"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference}"
STEPS="${STEPS:-25}"
METHOD="${METHOD:-rk4}"
SPEAKER_ID="${SPEAKER_ID:-0}"

# Activate conda
# source ~/.bashrc
# conda init
# conda activate qwen2_CALM

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Parse arguments
TEXT=""
NUM_RANDOM=1

if [[ "$1" == "--random" ]]; then
    NUM_RANDOM="${2:-1}"
    echo "Mode: Random samples from test-clean (n=$NUM_RANDOM)"
elif [[ -n "$1" ]]; then
    TEXT="$1"
    echo "Mode: Custom text"
else
    echo "Mode: Single random sample from test-clean"
fi

echo "============================================================"
echo "AGS TTS Inference"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS ($METHOD)"
echo "============================================================"

if [[ -n "$TEXT" ]]; then
    # Custom text mode
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$OUTPUT_DIR/custom_${TIMESTAMP}.wav"
    
    python scripts/04_inference_eval/run_inference.py \
        --checkpoint "$CHECKPOINT" \
        --text "$TEXT" \
        --speaker_id "$SPEAKER_ID" \
        --output "$OUTPUT_FILE" \
        --steps "$STEPS" \
        --method "$METHOD"
    
    echo ""
    echo "Generated: $OUTPUT_FILE"
else
    # Random sample mode
    echo ""
    echo "Sampling $NUM_RANDOM text(s) from test-clean..."
    
    # Find random transcript files
    TRANSCRIPT_FILES=$(find "$TEST_DATA" -name "*.normalized.txt" | shuf -n "$NUM_RANDOM")
    
    for TRANSCRIPT_FILE in $TRANSCRIPT_FILES; do
        TEXT_CONTENT=$(cat "$TRANSCRIPT_FILE")
        BASENAME=$(basename "$TRANSCRIPT_FILE" .normalized.txt)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_${TIMESTAMP}.wav"
        
        echo ""
        echo "Text: $TEXT_CONTENT"
        echo "Output: $OUTPUT_FILE"
        
        python scripts/04_inference_eval/run_inference.py \
            --checkpoint "$CHECKPOINT" \
            --text "$TEXT_CONTENT" \
            --speaker_id "$SPEAKER_ID" \
            --output "$OUTPUT_FILE" \
            --steps "$STEPS" \
            --method "$METHOD"
    done
fi

echo ""
echo "============================================================"
echo "Inference Complete!"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
