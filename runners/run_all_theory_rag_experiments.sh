#!/bin/bash

# Master runner for Theory/RAG Experiment
# Runs all 8 experiment configurations
#
# Experiment matrix:
# | Name                    | RAG   | Ex Theory | Co Theory |
# |-------------------------|-------|-----------|-----------|
# | plain                   | false | false     | false     |
# | rag                     | true  | false     | false     |
# | ex_theory               | false | true      | false     |
# | co_theory               | false | false     | true      |
# | rag_ex_theory           | true  | true      | false     |
# | rag_co_theory           | true  | false     | true      |
# | rag_ex_theory_co_theory | true  | true      | true      |
# | ex_theory_co_theory     | false | true      | true      |

# Parse command line argument for number of iterations per experiment (default: 1)
ITERATIONS=${1:-1}

echo "============================================================"
echo "Theory/RAG Experiment - Running all 8 configurations"
echo "Iterations per experiment: $ITERATIONS"
echo "============================================================"

EXPERIMENTS=(
    "plain"
    "rag"
    "ex_theory"
    "co_theory"
    "rag_ex_theory"
    "rag_co_theory"
    "rag_ex_theory_co_theory"
    "ex_theory_co_theory"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running experiment: $exp"
    echo "============================================================"
    
    ./runners/eval_theory_rag_${exp}.sh $ITERATIONS
    
    echo ""
    echo "Completed: $exp"
    sleep 5
done

echo ""
echo "============================================================"
echo "All 8 experiments completed!"
echo "============================================================"
