#!/bin/bash
# Shell script to run the model comparison experiment plots

# Set the base directory for the experiment
BASE_DIR="outputs/model_comparison_experiment"

# Define the configurations (model names) to compare
CONFIGS="gpt_4o_mini gemini_2_0_flash gemini_2_5_pro gemini_3_flash_preview"

# Optional: Set a custom title
TITLE="Model Comparison Experiment"

# Optional: Set custom output filename
OUTPUT="model_comparison_plots"

# Run the plot generation script
uv run python plots/plot_experiments.py \
    "$BASE_DIR" \
    $CONFIGS \
    --title "$TITLE" \
    --output "$OUTPUT"
