#!/bin/bash

# Simple supervised evaluator runner for ejemplos/3p
# Modify the file lists below to choose which files to evaluate

# Define which files to use (modify these lists as needed)
ASSIGNMENT="ejemplos/3p/consigna.txt"
# REQUIREMENTS="ejemplos/3p/requirements/*.json"
REQUIREMENTS="ejemplos/3p/requirements/requirement_06.json ejemplos/3p/requirements/requirement_07.json"
SUBMISSIONS="ejemplos/3p/alu6.py ejemplos/3p/alu8.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu6.txt ejemplos/3p/alu8.txt"
OUTPUT_DIR="outputs/supervised_evaluation"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluator
python3 run_supervised_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --verbose