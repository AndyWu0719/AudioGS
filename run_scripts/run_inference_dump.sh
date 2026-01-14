#!/bin/bash
# Atom Dumping Launcher
# Usage: ./run_inference_dump.sh [checkpoint_path] [output_dir]

CHECKPOINT=${1:-"logs/encoder_v1/best_model.pt"}
OUTPUT_DIR=${2:-"data/processed/atoms_v1"}

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwen2_CALM

# Launch
torchrun --nproc_per_node=4 \
    scripts/03_dataset_preprocessing/dump_atoms.py \
    --config configs/AudioGS_config.yaml \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --batch_size 32
