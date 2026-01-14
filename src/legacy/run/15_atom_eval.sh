#!/bin/bash
# ============================================================
# 15: Atom Fitting - Quality Evaluation
# ============================================================
# Evaluates the quality of generated atoms by reconstructing
# audio and computing metrics (PESQ, STOI, MCD, SI-SDR).
#
# Run this after preprocessing to verify atom quality.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG="${CONFIG:-configs/AudioGS_config.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
GPU="${GPU:-0}"

echo "============================================================"
echo "AudioGS - Atom Quality Evaluation"
echo "============================================================"
echo "Config: $CONFIG"
echo "Max samples: $MAX_SAMPLES"
echo "GPU: $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/01_atom_fitting/eval_reconstruction.py \
    --config "$CONFIG" \
    --max_samples "$MAX_SAMPLES"

echo "============================================================"
echo "Evaluation complete! Check logs/ for results."
echo "============================================================"
