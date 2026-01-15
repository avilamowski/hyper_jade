#!/bin/bash

# Simple supervised individual evaluator runner for ejemplos/3p
# Modify the file lists below to choose which files to evaluate

# Parse command line argument for number of iterations (default: 1)
ITERATIONS=${1:-1}

# Define which files to use (modify these lists as needed)
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
# REQUIREMENTS="ejemplos/3p/requirements/requirement_06.json ejemplos/3p/requirements/requirement_07.json"
SUBMISSIONS="ejemplos/3p/*.py"
# SUBMISSIONS="ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py"
REFERENCE_CORRECTIONS="ejemplos/3p/*.json"
# SUBMISSIONS="ejemplos/3p/alu1.py"
# REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json"
SYSTEM_CONFIG="runners/config/with_theory.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

# Set JADE_AGENT_CONFIG so system uses evaluator config instead of assignment config
export JADE_AGENT_CONFIG="$EVALUATOR_CONFIG"

echo "Running evaluation with theory summary $ITERATIONS time(s)..."

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "============================================================"
    echo "ITERATION $i of $ITERATIONS (WITH THEORY SUMMARY)"
    echo "============================================================"
    
    # Add a timestamp so multiple runs don't overwrite each other. Format: YYYYMMDDTHHMMSS
    TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
    OUTPUT_DIR="outputs/evaluation/with_theory/${TIMESTAMP}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run the evaluator
    uv run runners/run_supervised_individual_evaluator.py \
        --assignment "$ASSIGNMENT" \
        --requirements $REQUIREMENTS \
        --submissions $SUBMISSIONS \
        --reference-corrections $REFERENCE_CORRECTIONS \
        --output-dir "$OUTPUT_DIR" \
        --config "$SYSTEM_CONFIG" \
        --evaluator-config "$EVALUATOR_CONFIG" \
        --experiment-name "with_theory_${TIMESTAMP}" \
        --verbose
    
    # Small delay between iterations to ensure unique timestamps
    if [ $i -lt $ITERATIONS ]; then
        sleep 2
    fi
done

echo ""
echo "============================================================"
echo "Completed $ITERATIONS evaluation run(s) with theory summary"
echo "============================================================"
