#!/bin/bash
# Stage 00: Single Audio Atom Fitting
# Uses proven high-quality hyperparameters

# Default: specific test audio
AUDIO_FILE="data/raw/LibriTTS_R/train/train-clean-100/19/198/19_198_000016_000000.wav"
NUM_ATOMS=16384
NUM_STEPS=8000

python scripts/00_atom_fitting/fit_single_audio.py \
    --config configs/atom_fitting_config.yaml \
    --audio_file "$AUDIO_FILE" \
    --num_atoms $NUM_ATOMS \
    --num_steps $NUM_STEPS \
    $@
