#!/bin/bash
# ============================================================
# Batch WandB Sweep Agent Launcher
# ============================================================
# Launch multiple sweep agents across GPUs for parallel hyperparameter search.
#
# Usage:
#   ./scripts/run/13_sweep_batch.sh <sweep_path> [agents_per_gpu] [gpu_ids] [run_cap]
#
# Examples:
#   # 2 agents per GPU on all 8 GPUs (50 runs each)
#   ./scripts/run/13_sweep_batch.sh entity/project/sweep_id 2
#
#   # 4 agents per GPU on GPUs 0,1,2,3, 100 runs each
#   ./scripts/run/13_sweep_batch.sh entity/project/sweep_id 4 "0,1,2,3" 100
#
#   # 8 agents per GPU on GPU 0 only
#   ./scripts/run/13_sweep_batch.sh entity/project/sweep_id 8 "0"
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
SWEEP_PATH="${1:-}"
AGENTS_PER_GPU="${2:-2}"
GPU_IDS="${3:-0,1,2,3,4,5,6,7}"
RUN_CAP="${4:-50}"  # Max runs per agent (default 50)

# Validate
if [[ -z "$SWEEP_PATH" ]]; then
    echo "============================================================"
    echo "Batch WandB Sweep Agent Launcher"
    echo "============================================================"
    echo ""
    echo "Usage: $0 <sweep_path> [agents_per_gpu] [gpu_ids] [run_cap]"
    echo ""
    echo "Arguments:"
    echo "  sweep_path       Full sweep path (entity/project/sweep_id)"
    echo "  agents_per_gpu   Number of agents per GPU (default: 2, max: 8)"
    echo "  gpu_ids          Comma-separated GPU IDs (default: 0,1,2,3,4,5,6,7)"
    echo "  run_cap          Max runs per agent (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0 andywu-hkust/AGS-AtomFitting/abc123 2"
    echo "  $0 andywu-hkust/AGS-AtomFitting/abc123 4 '0,1,2,3' 100"
    echo ""
    exit 1
fi

# Validate agents per GPU
if [[ "$AGENTS_PER_GPU" -lt 1 || "$AGENTS_PER_GPU" -gt 8 ]]; then
    echo "Error: agents_per_gpu must be between 1 and 8"
    exit 1
fi

# Convert GPU IDs to array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
TOTAL_AGENTS=$((NUM_GPUS * AGENTS_PER_GPU))

echo "============================================================"
echo "Batch WandB Sweep Agent Launcher"
echo "============================================================"
echo "Sweep:          $SWEEP_PATH"
echo "GPUs:           ${GPU_ARRAY[*]} ($NUM_GPUS GPUs)"
echo "Agents per GPU: $AGENTS_PER_GPU"
echo "Total Agents:   $TOTAL_AGENTS"
echo "Runs per Agent: $RUN_CAP"
echo "Total Runs Cap: $((TOTAL_AGENTS * RUN_CAP))"
echo "============================================================"
echo ""

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs/sweep_agents_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"
echo ""

# PID file for cleanup
PID_FILE="$LOG_DIR/pids.txt"
touch "$PID_FILE"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping all agents..."
    while read pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
    echo "All agents stopped."
}

trap cleanup EXIT

# Launch agents
AGENT_NUM=0
for GPU_ID in "${GPU_ARRAY[@]}"; do
    for ((i=1; i<=AGENTS_PER_GPU; i++)); do
        AGENT_NUM=$((AGENT_NUM + 1))
        LOG_FILE="$LOG_DIR/agent_gpu${GPU_ID}_${i}.log"
        
        echo "Starting agent $AGENT_NUM/$TOTAL_AGENTS on GPU $GPU_ID (max $RUN_CAP runs)..."
        
        # Launch agent in background with run count limit
        CUDA_VISIBLE_DEVICES="$GPU_ID" wandb agent --count "$RUN_CAP" "$SWEEP_PATH" \
            > "$LOG_FILE" 2>&1 &
        
        PID=$!
        echo "$PID" >> "$PID_FILE"
        echo "  PID: $PID, Log: $LOG_FILE"
        
        # Small delay to avoid API rate limiting
        sleep 2
    done
done

echo ""
echo "============================================================"
echo "All $TOTAL_AGENTS agents started!"
echo "============================================================"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_DIR/agent_gpu*.log"
echo ""
echo "To stop all agents:"
echo "  Press Ctrl+C or kill the PIDs in $PID_FILE"
echo ""
echo "Waiting for agents... (Ctrl+C to stop)"

# Wait for all background processes
wait
