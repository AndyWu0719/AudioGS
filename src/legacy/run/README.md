# Run Scripts

Organized by stage-step numbering: `XY_name.sh` where X=stage, Y=step.

## Stage 1: Atom Fitting

| Script | Description |
|--------|-------------|
| `11_atom_debug.sh` | Debug single audio fitting |
| `12_atom_sweep.sh` | WandB hyperparameter sweep |
| `13_sweep_batch.sh` | Batch sweep agents across GPUs |
| `14_atom_batch.sh` | Batch dataset preprocessing |
| `15_atom_eval.sh` | Evaluate atom reconstruction quality |

## Stage 2: Flow Training

| Script | Description |
|--------|-------------|
| `21_flow_single.sh` | Single-GPU training |
| `22_flow_ddp.sh` | Multi-GPU DDP training (supports resume) |

## Stage 3: Inference & Evaluation

| Script | Description |
|--------|-------------|
| `31_inference.sh` | TTS inference |
| `32_eval_tts.sh` | TTS quality evaluation (WER, Speaker Sim) |
| `33_eval_edit.sh` | Editability evaluation (pitch, speed) |

## Quick Start

```bash
# 1. Debug atom fitting on single audio
./scripts/run/11_atom_debug.sh

# 2. Run hyperparameter sweep (optional)
./scripts/run/12_atom_sweep.sh --create
./scripts/run/13_sweep_batch.sh <sweep_path> 4 "0,1,2,3" 50

# 3. Batch preprocess dataset
./scripts/run/14_atom_batch.sh

# 4. Evaluate atom quality
./scripts/run/15_atom_eval.sh

# 5. Train Flow model
./scripts/run/22_flow_ddp.sh

# 6. Run inference
./scripts/run/31_inference.sh "Hello world"
```

## Environment Variables

Common environment variables for customization:

```bash
CONFIG=configs/flow_config.yaml  # Config file
NUM_GPUS=4                       # Number of GPUs
GPU=0                            # Single GPU ID
MAX_ITERS=8000                   # Training iterations
OUTPUT_DIR=outputs/              # Output directory
RESUME=1                         # Enable resume (22_flow_ddp.sh)
```
