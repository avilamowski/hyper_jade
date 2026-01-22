#!/bin/bash

# Master runner for Theory/RAG Experiment (P2 Ej1)
# Runs all 8 experiment configurations

# Parse command line argument for number of iterations per experiment (default: 1)
ITERATIONS=${1:-1}

echo "============================================================"
echo "Theory/RAG Experiment (P2 Ej1) - Running all 8 configurations"
echo "Iterations per experiment: $ITERATIONS"
echo "============================================================"

EXPERIMENTS=(
    "plain"
    "ex_theory"
    "co_theory"
    "ex_theory_co_theory"
#     "rag_ex_theory"
#     "rag_co_theory"
#     "rag_ex_theory_co_theory"
#     "rag"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running experiment: $exp"
    echo "============================================================"
    
    ./runners/eval_p2_ej1_theory_rag_${exp}.sh $ITERATIONS
    
    echo ""
    echo "Completed: $exp"
    sleep 5
done

echo ""
echo "============================================================"
echo "All 8 experiments completed!"
echo "============================================================"
