#!/bin/bash
# Shell script to run the examples quantity experiment comparison plots

# Set the base directory for the experiment
BASE_DIR="outputs/examples_quantity_experiment"

# Define the configurations to compare
CONFIGS="0c_0i 1c_3i 2c_3i 3c_3i 3c_0i"

# Optional: Set a custom title
TITLE="Example Quantity Experiment Comparison"

# Optional: Set custom output filename
OUTPUT="comparison_plots_with_std"

# Run the plot generation script
uv run python plots/plot_experiments.py \
    "$BASE_DIR" \
    $CONFIGS \
    --title "$TITLE" \
    --output "$OUTPUT"
