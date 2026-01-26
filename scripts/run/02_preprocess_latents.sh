#!/bin/bash
# ============================================================
# 02: Preprocess Dataset -> Codec Latents
# ============================================================
# Encodes a LibriTTS subset into per-utterance latent `.pt` files
# (plus latent_stats.pt and speaker_to_id.json) for Stage04 Flow training.
#
# Usage:
#   bash scripts/run/02_preprocess_latents.sh \
#     [codec_ckpt] [subset_dir] [out_dir]
#
# Defaults:
#   codec_ckpt: logs/codec/checkpoints/final.pt
#   subset_dir: data/raw/LibriTTS_R/train/train-clean-100
#   out_dir:    data/latents/LibriTTS_R/train-clean-100
#
# Env:
#   CODEC_CONFIG  Codec config (default: configs/codec_config.yaml)
#   MAX_SECONDS   Truncate utterances to this duration (default: 15)
#   LIMIT         Limit number of wavs (optional)
#   GPU           CUDA device index (default: 0)
# ============================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CODEC_CKPT="${1:-logs/codec/checkpoints/final.pt}"
SUBSET_DIR="${2:-data/raw/LibriTTS_R/train/train-clean-100}"
OUT_DIR="${3:-data/latents/LibriTTS_R/train-clean-100}"

CODEC_CONFIG="${CODEC_CONFIG:-configs/codec_config.yaml}"
MAX_SECONDS="${MAX_SECONDS:-15}"
LIMIT="${LIMIT:-}"
GPU="${GPU:-0}"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "============================================================"
echo "AudioGS - Stage02 Preprocess Latents"
echo "============================================================"
echo "Codec ckpt: $CODEC_CKPT"
echo "Codec config: $CODEC_CONFIG"
echo "Subset dir: $SUBSET_DIR"
echo "Out dir: $OUT_DIR"
echo "Max seconds: $MAX_SECONDS"
echo "Limit: ${LIMIT:-disabled}"
echo "GPU: $GPU"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU python scripts/02_data_preprocessing/run_preprocess.py \
  --codec_ckpt "$CODEC_CKPT" \
  --codec_config "$CODEC_CONFIG" \
  --subset_dir "$SUBSET_DIR" \
  --out_dir "$OUT_DIR" \
  --max_seconds "$MAX_SECONDS" \
  "${LIMIT_ARGS[@]}"

echo "============================================================"
echo "Stage02 complete!"
echo "============================================================"
