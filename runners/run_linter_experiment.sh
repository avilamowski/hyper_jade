#!/bin/bash

# =============================================================================
# Linter vs No-Linter Comparison Experiment
# =============================================================================
# This script runs multiple iterations of the evaluation pipeline with and
# without the linter agent, then generates comparison plots.
#
# Usage:
#   ./runners/run_linter_experiment.sh [NUM_RUNS] [NUM_SUBMISSIONS]
#
# Arguments:
#   NUM_RUNS        - Number of runs for each configuration (default: 5)
#   NUM_SUBMISSIONS - Number of submissions to evaluate per run (default: 10)
#
# Example:
#   ./runners/run_linter_experiment.sh 5 10
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_RUNS=${1:-5}
NUM_SUBMISSIONS=${2:-10}
EXPERIMENT_NAME="lint_vs_no_lint_${NUM_SUBMISSIONS}subs"

# Validate NUM_SUBMISSIONS (max 10 available)
if [ "$NUM_SUBMISSIONS" -gt 10 ]; then
    echo "⚠️  Warning: Only 10 submissions available. Using 10."
    NUM_SUBMISSIONS=10
fi

# Build submission and reference lists dynamically
SUBMISSIONS=""
REFERENCE_CORRECTIONS=""
for i in $(seq 1 $NUM_SUBMISSIONS); do
    SUBMISSIONS="$SUBMISSIONS ejemplos/3p/alu${i}.py"
    REFERENCE_CORRECTIONS="$REFERENCE_CORRECTIONS ejemplos/3p/alu${i}.json"
done
# Trim leading space
SUBMISSIONS=$(echo $SUBMISSIONS | xargs)
REFERENCE_CORRECTIONS=$(echo $REFERENCE_CORRECTIONS | xargs)

# Common configuration
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

# Output directory for this experiment
BASE_OUTPUT_DIR="outputs/evaluation/${EXPERIMENT_NAME}"

echo "============================================================"
echo "🧪 LINTER COMPARISON EXPERIMENT"
echo "============================================================"
echo "Configuration:"
echo "  - Runs per config: $NUM_RUNS"
echo "  - Submissions per run: $NUM_SUBMISSIONS"
echo "  - Output directory: $BASE_OUTPUT_DIR"
echo "============================================================"
echo ""

# Function to run a single evaluation
run_evaluation() {
    local config_name=$1
    local config_file=$2
    local run_number=$3
    
    local timestamp=$(date +"%Y%m%dT%H%M%S")
    local output_dir="${BASE_OUTPUT_DIR}/${config_name}/${timestamp}"
    
    echo "📝 Running ${config_name} - Run $run_number/$NUM_RUNS (${timestamp})"
    
    mkdir -p "$output_dir"
    
    uv run runners/run_supervised_individual_evaluator.py \
        --assignment "$ASSIGNMENT" \
        --requirements $REQUIREMENTS \
        --submissions $SUBMISSIONS \
        --reference-corrections $REFERENCE_CORRECTIONS \
        --output-dir "$output_dir" \
        --config "$config_file" \
        --evaluator-config "$EVALUATOR_CONFIG" \
        --experiment-name "${config_name}_run${run_number}_${timestamp}" \
        --verbose
    
    echo "✅ Completed ${config_name} - Run $run_number"
    echo ""
}

# Track start time
START_TIME=$(date +%s)

# Run experiments WITH linter
echo "============================================================"
echo "🔍 PHASE 1: Running WITH LINTER ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation "with_linter" "runners/config/with_linter.yaml" $run
done

# Run experiments WITHOUT linter
echo "============================================================"
echo "🔍 PHASE 2: Running WITHOUT LINTER ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation \"without_linter\" \"runners/config/plain.yaml\" $run
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "============================================================"
echo "📊 PHASE 3: Generating comparison plots"
echo "============================================================"

# Generate plots
uv run python plots/plot_experiments.py \
    "$BASE_OUTPUT_DIR" \
    with_linter without_linter \
    --title "Linter Agent Comparison (${NUM_SUBMISSIONS} submissions, ${NUM_RUNS} runs)" \
    --output linter_comparison

echo ""
echo "============================================================"
echo "✅ EXPERIMENT COMPLETE"
echo "============================================================"
echo "Summary:"
echo "  - Total runs: $((NUM_RUNS * 2))"
echo "  - Time elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  - Results: ${BASE_OUTPUT_DIR}/"
echo "  - Plots: ${BASE_OUTPUT_DIR}/linter_comparison.png"
echo "============================================================"
