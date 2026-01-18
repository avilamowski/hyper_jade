#!/bin/bash

# RAG and Theory Experiment: Run all 8 experiments concurrently
# Usage: ./runners/rag_and_theory.sh <iterations>
#
# This runs all 8 theory/RAG experiment configurations in parallel

ITERATIONS=${1:-1}

echo "============================================================"
echo "RAG and Theory Experiment"
echo "Running all 8 configurations concurrently"
echo "Iterations: $ITERATIONS"
echo "============================================================"

./runners/multi_run.sh $ITERATIONS \
    rag \
    rag_ex_theory \
    rag_co_theory \
    rag_ex_theory_co_theory
    # plain \
    # ex_theory \
    # co_theory \
    # ex_theory_co_theory
