#!/bin/bash

# Simple supervised individual evaluator runner for ejemplos/3p
# Modify the file lists below to choose which files to evaluate

# Define which files to use (modify these lists as needed)
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
# REQUIREMENTS="ejemplos/3p/requirements/requirement_06.json ejemplos/3p/requirements/requirement_07.json"
# SUBMISSIONS="ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py ejemplos/3p/alu4.py ejemplos/3p/alu5.py ejemplos/3p/alu6.py"
# REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json ejemplos/3p/alu2.json ejemplos/3p/alu3.json ejemplos/3p/alu4.json ejemplos/3p/alu5.json ejemplos/3p/alu6.json"
SUBMISSIONS="ejemplos/3p/alu1.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json"
# Add a timestamp so multiple runs don't overwrite each other. Format: YYYYMMDDTHHMMSS
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
OUTPUT_DIR="outputs/supervised_individual_evaluation/${TIMESTAMP}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluator
python3 runners/run_supervised_individual_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --verbose
