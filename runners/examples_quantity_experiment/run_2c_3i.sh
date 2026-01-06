#!/bin/bash

# Script to run examples quantity experiment: 2 Correct, 3 Incorrect
# Evaluates 5 selected students: alu1, alu2, alu3, alu8, alu10
# WITHOUT RAG, WITHOUT FUNCTION ANALYZER

set -e  # Exit on error

# Configuration
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/requirement_*.json"
SUBMISSIONS="ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py ejemplos/3p/alu8.py ejemplos/3p/alu10.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json ejemplos/3p/alu2.json ejemplos/3p/alu3.json ejemplos/3p/alu8.json ejemplos/3p/alu10.json"

# Add a timestamp so multiple runs don't overwrite each other
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
OUTPUT_DIR="outputs/examples_quantity_experiment/2c_3i/${TIMESTAMP}"
SYSTEM_CONFIG="runners/config/examples_quantity_experiment/2c_3i.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

echo "=========================================="
echo "EXAMPLES QUANTITY EXPERIMENT - 2C_3I"
echo "=========================================="
echo "Assignment: $ASSIGNMENT"
echo "Submissions: 5 students (alu1, alu2, alu3, alu8, alu10)"
echo "Output directory: $OUTPUT_DIR"
echo "System config: $SYSTEM_CONFIG"
echo "Evaluator config: $EVALUATOR_CONFIG"
echo "Example config: 2 correct, 3 incorrect"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if requirements exist
if ! ls ejemplos/3p/requirements_es/requirement_*.json 1> /dev/null 2>&1; then
    echo "❌ Error: Requirement files not found in ejemplos/3p/requirements_es/"
    exit 1
fi

echo "🚀 Running evaluation with 2 correct and 3 incorrect examples..."
echo ""

# Run the evaluator
uv run python runners/run_supervised_individual_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --config "$SYSTEM_CONFIG" \
    --evaluator-config "$EVALUATOR_CONFIG" \
    --experiment-name "examples_quantity_2c_3i_${TIMESTAMP}" \
    --verbose

echo ""
echo "✅ Evaluation completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
