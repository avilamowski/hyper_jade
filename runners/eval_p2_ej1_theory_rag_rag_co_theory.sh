#!/bin/bash

# Theory/RAG Experiment Runner (P2 Ej1): rag_co_theory

# Parse command line argument for number of iterations (default: 1)
ITERATIONS=${1:-1}

# Define which files to use
ASSIGNMENT="ejemplos/ej1-2025-s2-p2-ej1/consigna.txt"
REQUIREMENTS="ejemplos/ej1-2025-s2-p2-ej1/requirements_es/*.json"
SUBMISSIONS="ejemplos/ej1-2025-s2-p2-ej1/alu*.py"
REFERENCE_CORRECTIONS="ejemplos/ej1-2025-s2-p2-ej1/alu*.json"

SYSTEM_CONFIG="runners/config/theory_rag_experiment/rag_co_theory.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

# Set JADE_AGENT_CONFIG so RAG system uses evaluator config
export JADE_AGENT_CONFIG="$EVALUATOR_CONFIG"

echo "Running rag_co_theory experiment (P2 Ej1) $ITERATIONS time(s)..."

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "============================================================"
    echo "ITERATION $i of $ITERATIONS"
    echo "============================================================"
    
    TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
    OUTPUT_DIR="outputs/evaluation/p2_ej1_theory_rag_experiment/rag_co_theory/${TIMESTAMP}"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Use uv run with wildcard expansion (shell handles the expansion)
    uv run runners/run_supervised_individual_evaluator.py \
        --assignment "$ASSIGNMENT" \
        --requirements $REQUIREMENTS \
        --submissions $SUBMISSIONS \
        --reference-corrections $REFERENCE_CORRECTIONS \
        --output-dir "$OUTPUT_DIR" \
        --config "$SYSTEM_CONFIG" \
        --evaluator-config "$EVALUATOR_CONFIG" \
        --experiment-name "rag_co_theory_p2_ej1_${TIMESTAMP}" \
        --verbose
    
    if [ $i -lt $ITERATIONS ]; then
        sleep 2
    fi
done

echo ""
echo "============================================================"
echo "Completed $ITERATIONS rag_co_theory evaluation run(s)"
echo "============================================================"
