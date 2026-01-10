#!/bin/bash

# =============================================================================
# Code Corrector Model Comparison Experiment
# =============================================================================
# This script runs multiple iterations of the evaluation pipeline with different
# models for the code_corrector agent, then generates comparison plots.
#
# Models tested for code_corrector:
#   - gpt-4o-mini
#   - gemini-2.0-flash
#   - gemini-2.5-pro
#   - gemini-3-flash-preview
#
# All other agents use gemini-2.0-flash consistently across all runs.
#
# Usage:
#   ./runners/run_model_comparison_experiment.sh [NUM_RUNS] [NUM_SUBMISSIONS]
#
# Arguments:
#   NUM_RUNS        - Number of runs for each model (default: 5)
#   NUM_SUBMISSIONS - Number of submissions to evaluate per run (default: 10)
#
# Example:
#   ./runners/run_model_comparison_experiment.sh 5 10
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_RUNS=${1:-5}
NUM_SUBMISSIONS=${2:-10}
EXPERIMENT_NAME="model_comparison_${NUM_SUBMISSIONS}subs"

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
echo "🧪 CODE CORRECTOR MODEL COMPARISON EXPERIMENT"
echo "============================================================"
echo "Configuration:"
echo "  - Runs per model: $NUM_RUNS"
echo "  - Submissions per run: $NUM_SUBMISSIONS"
echo "  - Output directory: $BASE_OUTPUT_DIR"
echo "  - Models to test:"
echo "    • gpt-4o-mini"
echo "    • gemini-2.0-flash"
echo "    • gemini-2.5-pro"
echo "    • gemini-3-flash-preview"
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

# Run experiments for gpt-4o-mini
echo "============================================================"
echo "🔍 PHASE 1/4: Running with gpt-4o-mini ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation "gpt4o_mini" "runners/config/model_comparison/corrector_gpt4o_mini.yaml" $run
done

# Run experiments for gemini-2.0-flash
echo "============================================================"
echo "🔍 PHASE 2/4: Running with gemini-2.0-flash ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation "gemini_2_0_flash" "runners/config/model_comparison/corrector_gemini_2_0_flash.yaml" $run
done

# Run experiments for gemini-2.5-pro
echo "============================================================"
echo "🔍 PHASE 3/4: Running with gemini-2.5-pro ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation "gemini_2_5_pro" "runners/config/model_comparison/corrector_gemini_2_5_pro.yaml" $run
done

# Run experiments for gemini-3-flash-preview
echo "============================================================"
echo "🔍 PHASE 4/4: Running with gemini-3-flash-preview ($NUM_RUNS runs)"
echo "============================================================"
for run in $(seq 1 $NUM_RUNS); do
    run_evaluation "gemini_3_flash_preview" "runners/config/model_comparison/corrector_gemini_3_flash_preview.yaml" $run
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "============================================================"
echo "📊 PHASE 5: Generating comparison plots"
echo "============================================================"

# Generate plots
uv run python plots/plot_experiments.py \
    "$BASE_OUTPUT_DIR" \
    gpt4o_mini gemini_2_0_flash gemini_2_5_pro gemini_3_flash_preview \
    --title "Code Corrector Model Comparison (${NUM_SUBMISSIONS} submissions, ${NUM_RUNS} runs)" \
    --output model_comparison

echo ""
echo "============================================================"
echo "✅ EXPERIMENT COMPLETE"
echo "============================================================"
echo "Summary:"
echo "  - Total runs: $((NUM_RUNS * 4))"
echo "  - Time elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  - Results: ${BASE_OUTPUT_DIR}/"
echo "  - Plots: ${BASE_OUTPUT_DIR}/model_comparison.png"
echo "============================================================"
