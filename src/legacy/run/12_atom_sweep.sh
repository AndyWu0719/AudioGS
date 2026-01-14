#!/bin/bash
# ============================================================
# WandB Sweep for Atom Fitting Hyperparameter Search
# ============================================================
# Optimize atom fitting config for best reconstruction quality
# with fewer iterations and optimal atom count.
#
# Usage:
#   ./scripts/run/12_atom_sweep.sh [--create|--run <sweep_id>]
#
# Examples:
#   ./scripts/run/12_atom_sweep.sh --create        # Create new sweep
#   ./scripts/run/12_atom_sweep.sh --run abc123    # Run agent for sweep
#   ./scripts/run/12_atom_sweep.sh                 # Interactive mode
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
SWEEP_CONFIG="configs/sweep_atom_fitting.yaml"
WANDB_PROJECT="${WANDB_PROJECT:-AGS-AtomFitting}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set your entity if needed
NUM_AGENTS="${NUM_AGENTS:-1}"

echo "============================================================"
echo "WandB Sweep: Atom Fitting Optimization"
echo "============================================================"
echo "Config: $SWEEP_CONFIG"
echo "Project: $WANDB_PROJECT"
echo "============================================================"

if [[ "$1" == "--create" ]]; then
    echo ""
    echo "Creating new sweep..."
    if [[ -n "$WANDB_ENTITY" ]]; then
        wandb sweep --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "$SWEEP_CONFIG"
    else
        wandb sweep --project "$WANDB_PROJECT" "$SWEEP_CONFIG"
    fi
    echo ""
    echo "Sweep created! Copy the sweep ID and run:"
    echo "  ./scripts/run/12_atom_sweep.sh --run <sweep_id>"

elif [[ "$1" == "--run" ]]; then
    SWEEP_ID="$2"
    if [[ -z "$SWEEP_ID" ]]; then
        echo "Error: Please provide sweep ID or full path"
        echo "Usage: ./scripts/run/12_atom_sweep.sh --run <sweep_id>"
        echo "   or: ./scripts/run/12_atom_sweep.sh --run entity/project/sweep_id"
        exit 1
    fi
    
    echo ""
    echo "Starting sweep agent for: $SWEEP_ID"
    echo "Press Ctrl+C to stop"
    echo ""
    
    # If sweep_id contains /, use as-is; otherwise construct full path
    if [[ "$SWEEP_ID" == *"/"* ]]; then
        wandb agent "$SWEEP_ID"
    elif [[ -n "$WANDB_ENTITY" ]]; then
        wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"
    else
        # Try to get entity from wandb
        ENTITY=$(wandb status 2>/dev/null | grep "Logged in" | grep -oP '(?<=\().*(?=\))' || echo "")
        if [[ -n "$ENTITY" ]]; then
            wandb agent "$ENTITY/$WANDB_PROJECT/$SWEEP_ID"
        else
            echo "Error: Could not determine WandB entity."
            echo "Please use full path: ./scripts/run/09_sweep_atom_fitting.sh --run entity/project/sweep_id"
            echo "Or set WANDB_ENTITY environment variable"
            exit 1
        fi
    fi

else
    echo ""
    echo "Interactive mode:"
    echo ""
    echo "1. Create new sweep:    ./scripts/run/12_atom_sweep.sh --create"
    echo "2. Run sweep agent:     ./scripts/run/12_atom_sweep.sh --run <sweep_id>"
    echo ""
    echo "Or run manually:"
    echo "  wandb sweep configs/sweep_atom_fitting.yaml"
    echo "  wandb agent <entity>/<project>/<sweep_id>"
fi
