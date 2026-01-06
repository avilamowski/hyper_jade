#!/bin/bash
# Shell script to compare experiments with and without RAG

# Set the base directory for the experiment
BASE_DIR="outputs/evaluation"

# Define the configurations to compare
CONFIGS="without_rag with_rag"

# Optional: Set a custom title
TITLE="RAG Impact Comparison"

# Optional: Set custom output filename
OUTPUT="rag_comparison_plots"

# Run the plot generation script
uv run plots/plot_experiments.py \
    "$BASE_DIR" \
    $CONFIGS \
    --title "$TITLE" \
    --output "$OUTPUT"
