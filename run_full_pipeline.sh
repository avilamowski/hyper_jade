#!/bin/bash

# Master script to run EVERYTHING: experiments + aggregation + plots
# Run this and go for a run - everything will be done when you return!

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     FULL MODEL COMPARISON PIPELINE - AUTOMATED RUN         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Run 4 models × 5 runs × 10 students = 200 evaluations"
echo "  2. Aggregate all metrics"
echo "  3. Generate comparison plots"
echo ""
echo "Estimated time: 45-90 minutes"
echo "════════════════════════════════════════════════════════════"
echo ""

OVERALL_START=$(date +%s)

# Step 1: Run all experiments
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  STEP 1/3: Running all model experiments                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

bash runners/model_comparison_experiment/run_all_models.sh

if [ $? -ne 0 ]; then
    echo "❌ Error: Experiments failed"
    exit 1
fi

# Step 2: Aggregate metrics
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  STEP 2/3: Aggregating metrics                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

uv run python runners/model_comparison_experiment/aggregate_metrics.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Metric aggregation failed"
    exit 1
fi

# Step 3: Generate plots
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  STEP 3/3: Generating comparison plots                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

bash plots/model_comparison_experiment.sh

if [ $? -ne 0 ]; then
    echo "❌ Error: Plot generation failed"
    exit 1
fi

# Calculate total time
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              🎉 ALL DONE! WELCOME BACK! 🎉                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total pipeline duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "📊 Results available:"
echo "  - Raw results: outputs/model_comparison_experiment/"
echo "  - Aggregated metrics: outputs/model_comparison_experiment/aggregated_metrics.json"
echo "  - Plots: outputs/model_comparison_experiment/model_comparison_plots.png"
echo "  - PDF: outputs/model_comparison_experiment/model_comparison_plots.pdf"
echo ""
echo "Next steps:"
echo "  - Check the plots: open outputs/model_comparison_experiment/model_comparison_plots.png"
echo "  - Review metrics: cat outputs/model_comparison_experiment/aggregated_metrics.json | jq"
echo "  - Analyze results in your thesis!"
echo ""
