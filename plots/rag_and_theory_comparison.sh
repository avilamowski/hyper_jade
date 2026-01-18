#!/bin/bash
# Shell script to compare RAG and Theory experiments
# Compares all 8 configurations from the theory_rag_experiment

# Set the base directory for the experiment
BASE_DIR="outputs/evaluation/theory_rag_experiment"

# Define the configurations to compare (all 8 experiments)
CONFIGS="plain rag ex_theory co_theory rag_ex_theory rag_co_theory rag_ex_theory_co_theory ex_theory_co_theory"

# Optional: Set a custom title
TITLE="RAG and Theory Summary Impact Comparison"

# Optional: Set custom output filename
OUTPUT="rag_and_theory_comparison_plots"

# Run the plot generation script
uv run plots/plot_experiments.py \
    "$BASE_DIR" \
    $CONFIGS \
    --title "$TITLE" \
    --output "$OUTPUT"
