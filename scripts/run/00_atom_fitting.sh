#!/bin/bash
# ============================================================
# 00: Gabor Frame Benchmark
# ============================================================
# Runs the deterministic STFT/ISTFT reconstruction benchmark.
#
# Usage:
#   bash scripts/run/00_atom_fitting.sh               # benchmark on 1s/3s/5s
#   bash scripts/run/00_atom_fitting.sh path/to.wav   # single-file run
#
# Env:
#   CONFIG    Path to stage00 config (default: scripts/00_atom_fitting/config.yaml)
#   GPU       CUDA device index (default: 0)
#   DURATIONS Space-separated seconds list (default: "1.0 3.0 5.0")
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-scripts/00_atom_fitting/config.yaml}"
GPU="${GPU:-0}"
AUDIO_FILE="${1:-}"
DURATIONS="${DURATIONS:-"1.0 3.0 5.0"}"

ARGS=(--config "$CONFIG")
if [[ -n "$AUDIO_FILE" ]]; then
  ARGS+=(--audio_file "$AUDIO_FILE")
else
  read -r -a DURS <<<"$DURATIONS"
  ARGS+=(--durations "${DURS[@]}")
fi

echo "============================================================"
echo "AudioGS - Stage00 Gabor Frame Benchmark"
echo "============================================================"
echo "Config: $CONFIG"
echo "GPU: $GPU"
if [[ -n "$AUDIO_FILE" ]]; then
  echo "Audio file: $AUDIO_FILE"
else
  echo "Durations: $DURATIONS"
fi
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/00_atom_fitting/run_benchmark.py "${ARGS[@]}"

echo "============================================================"
echo "Stage00 complete!"
echo "============================================================"
