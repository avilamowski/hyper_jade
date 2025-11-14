#!/bin/bash

# Simple supervised individual evaluator runner for ejemplos/3p
# WITHOUT FUNCTION ANALYZER: Does NOT use AST-based function grouping
# Modify the file lists below to choose which files to evaluate

# Define which files to use (modify these lists as needed)
ASSIGNMENT="ejemplos/3p/consigna.txt"
# REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
REQUIREMENTS="ejemplos/3p/requirements_es/requirement_02.json"
# SUBMISSIONS="ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py ejemplos/3p/alu4.py ejemplos/3p/alu5.py"
# REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json ejemplos/3p/alu2.json ejemplos/3p/alu3.json ejemplos/3p/alu4.json ejemplos/3p/alu5.json"
SUBMISSIONS="ejemplos/3p/alu1.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json"
# Add a timestamp so multiple runs don't overwrite each other. Format: YYYYMMDDTHHMMSS
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
OUTPUT_DIR="outputs/evaluation/without_function_analyzer/${TIMESTAMP}"
SYSTEM_CONFIG="runners/config/without_function_analyzer.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"
# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluator
python runners/run_supervised_individual_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --config "$SYSTEM_CONFIG" \
    --evaluator-config "$EVALUATOR_CONFIG" \
    --experiment-name "without_function_analyzer_${TIMESTAMP}" \
    --verbose
