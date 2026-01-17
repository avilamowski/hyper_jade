#!/bin/bash
# Quick test of error locator integration
# Runs on a single submission to verify error locator runs in pipeline

# Configuration files
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
SUBMISSIONS="ejemplos/3p/alu1.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json"
SYSTEM_CONFIG="runners/config/plain.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
OUTPUT_DIR="outputs/evaluation/test_error_locator/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Testing Error Locator Agent integration..."
echo "NOTE: Make sure error_locator.enabled is set to true in $SYSTEM_CONFIG"
echo ""

# Run the evaluator on single submission
uv run runners/run_supervised_individual_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --config "$SYSTEM_CONFIG" \
    --evaluator-config "$EVALUATOR_CONFIG" \
    --experiment-name "test_error_locator_${TIMESTAMP}" \
    --verbose

echo ""
echo "============================================================"
echo "Test completed. Check output at: $OUTPUT_DIR/alu1/"
echo "Look for 'locations' field in generated_corrections.json"
echo "============================================================"


