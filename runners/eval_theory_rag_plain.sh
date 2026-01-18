#!/bin/bash

# Theory/RAG Experiment Runner: plain
# Variables: RAG=false, Example Theory=false, Corrector Theory=false

# Parse command line argument for number of iterations (default: 1)
ITERATIONS=${1:-1}

# Define which files to use
ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/*.json"
SUBMISSIONS="ejemplos/3p/*.py"
REFERENCE_CORRECTIONS="ejemplos/3p/*.json"
SYSTEM_CONFIG="runners/config/theory_rag_experiment/plain.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

# Set JADE_AGENT_CONFIG so RAG system uses evaluator config
export JADE_AGENT_CONFIG="$EVALUATOR_CONFIG"

echo "Running plain experiment $ITERATIONS time(s)..."
echo "Config: RAG=false, Example Theory=false, Corrector Theory=false"

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "============================================================"
    echo "ITERATION $i of $ITERATIONS"
    echo "============================================================"
    
    TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
    OUTPUT_DIR="outputs/evaluation/theory_rag_experiment/plain/${TIMESTAMP}"
    
    mkdir -p "$OUTPUT_DIR"
    
    uv run runners/run_supervised_individual_evaluator.py \
        --assignment "$ASSIGNMENT" \
        --requirements $REQUIREMENTS \
        --submissions $SUBMISSIONS \
        --reference-corrections $REFERENCE_CORRECTIONS \
        --output-dir "$OUTPUT_DIR" \
        --config "$SYSTEM_CONFIG" \
        --evaluator-config "$EVALUATOR_CONFIG" \
        --experiment-name "plain_${TIMESTAMP}" \
        --verbose
    
    if [ $i -lt $ITERATIONS ]; then
        sleep 2
    fi
done

echo ""
echo "============================================================"
echo "Completed $ITERATIONS plain evaluation run(s)"
echo "============================================================"
