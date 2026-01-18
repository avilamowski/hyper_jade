#!/bin/bash

# Multi-run script: Runs multiple experiments concurrently
# Usage: ./runners/multi_run.sh <iterations> <experiment1> [experiment2] [experiment3] ...
#
# Example: ./runners/multi_run.sh 3 plain rag ex_theory
# This will run plain, rag, and ex_theory experiments 3 times each, all concurrently

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <iterations> <experiment1> [experiment2] [experiment3] ..."
    echo ""
    echo "Available experiments:"
    echo "  plain, rag, ex_theory, co_theory"
    echo "  rag_ex_theory, rag_co_theory, rag_ex_theory_co_theory, ex_theory_co_theory"
    echo ""
    echo "Example: $0 3 plain rag ex_theory"
    exit 1
fi

ITERATIONS=$1
shift  # Remove first argument, leaving only experiment names

EXPERIMENTS=("$@")
PIDS=()
EXPERIMENT_NAMES=()

echo "============================================================"
echo "Multi-Run: Running ${#EXPERIMENTS[@]} experiments concurrently"
echo "Iterations per experiment: $ITERATIONS"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "============================================================"
echo ""

# Start all experiments in background
for exp in "${EXPERIMENTS[@]}"; do
    RUNNER="./runners/eval_theory_rag_${exp}.sh"
    
    if [ ! -f "$RUNNER" ]; then
        echo "ERROR: Runner script not found: $RUNNER"
        continue
    fi
    
    echo "Starting: $exp (PID will be logged)"
    
    # Run in background and capture PID
    $RUNNER $ITERATIONS &
    PID=$!
    PIDS+=($PID)
    EXPERIMENT_NAMES+=("$exp")
    
    echo "  -> Started $exp with PID $PID"
done

echo ""
echo "============================================================"
echo "All ${#PIDS[@]} experiments started. Waiting for completion..."
echo "PIDs: ${PIDS[*]}"
echo "============================================================"
echo ""

# Wait for all background processes and track results
FAILED=()
SUCCEEDED=()

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    EXP=${EXPERIMENT_NAMES[$i]}
    
    if wait $PID; then
        SUCCEEDED+=("$EXP")
        echo "✓ Completed: $EXP (PID $PID)"
    else
        FAILED+=("$EXP")
        echo "✗ Failed: $EXP (PID $PID)"
    fi
done

echo ""
echo "============================================================"
echo "Multi-Run Complete!"
echo "============================================================"
echo "Succeeded: ${#SUCCEEDED[@]} - ${SUCCEEDED[*]}"
echo "Failed: ${#FAILED[@]} - ${FAILED[*]}"
echo "============================================================"

# Exit with error if any failed
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
