#!/bin/bash
# ============================================================
# AGS Editability Evaluation
# ============================================================
# Evaluate pitch shift and time stretch capabilities.
#
# Usage:
#   ./scripts/run/33_eval_edit.sh                    # Default 20 samples
#   ./scripts/run/33_eval_edit.sh --samples 50       # Custom sample count
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT="${CHECKPOINT:-logs/flow/best.pt}"
ATOM_DIR="${ATOM_DIR:-data/atoms/LibriTTS_R/train/train-clean-100}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

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
        *)
            shift
            ;;
    esac
done

# Activate conda
source ~/.bashrc
conda activate qwen2_CALM

echo "============================================================"
echo "AGS Editability Evaluation"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Atom Dir: $ATOM_DIR"
echo "Samples: $NUM_SAMPLES"
echo "============================================================"

python scripts/04_inference_eval/run_eval_edit.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$ATOM_DIR" \
    --num_samples "$NUM_SAMPLES"

echo ""
echo "============================================================"
echo "Editability Evaluation Complete!"
echo "============================================================"
