#!/usr/bin/env python3
"""
Generate comparison plots for example quantity experiments with standard deviation.
Dynamically reads ALL aggregate_metrics.json files from each configuration directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from glob import glob

def load_all_metrics(base_dir: Path, config: str):
    """Load all aggregate_metrics.json files for a given configuration."""
    pattern = str(base_dir / config / "*" / "aggregate_metrics.json")
    files = glob(pattern)
    
    if not files:
        print(f"⚠️  Warning: No metrics files found for {config}")
        return []
    
    metrics_list = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics_list.append({
                    'completeness': data['metric_averages']['completeness'],
                    'content_similarity': data['metric_averages']['content_similarity'],
                    'restraint': data['metric_averages']['restraint'],
                    'average_overall_score': data['average_overall_score']
                })
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    print(f"✓ Loaded {len(metrics_list)} runs for {config}")
    return metrics_list

# Base directory for experiments
base_dir = Path('outputs/examples_quantity_experiment')

# Configurations to analyze
configs = ['0c_0i', '1c_3i', '2c_3i', '3c_3i', '3c_0i']

# Load all data dynamically
print("Loading metrics from all runs...")
print("="*60)
all_data = {}
for config in configs:
    all_data[config] = load_all_metrics(base_dir, config)

print("="*60)

# Calculate mean and std for each metric
metrics = ['completeness', 'content_similarity', 'restraint', 'average_overall_score']
stats = {}

for config, runs in all_data.items():
    if not runs:
        print(f"⚠️  Skipping {config} - no data available")
        continue
        
    stats[config] = {}
    for metric in metrics:
        values = [run[metric] for run in runs]
        stats[config][metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0,  # Sample std dev
            'n': len(values)
        }

# Configuration labels
config_labels = {
    '0c_0i': '0 Correct\n0 Erroneous',
    '1c_3i': '1 Correct\n3 Erroneous',
    '2c_3i': '2 Correct\n3 Erroneous',
    '3c_3i': '3 Correct\n3 Erroneous',
    '3c_0i': '3 Correct\n0 Erroneous'
}

metric_titles = {
    'completeness': 'Completeness',
    'content_similarity': 'Content Similarity',
    'restraint': 'Restraint',
    'average_overall_score': 'Average Overall Score'
}

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Example Quantity Experiment Comparison (Mean ± Std Dev)', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

# Color palette
colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Added purple for 0c_0i

# Create bar plots with error bars
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Extract mean and std for this metric
    means = [stats[config][metric]['mean'] for config in configs if config in stats]
    stds = [stats[config][metric]['std'] for config in configs if config in stats]
    labels = [config_labels[config] for config in configs if config in stats]
    n_runs = [stats[config][metric]['n'] for config in configs if config in stats]
    
    # Create bars with error bars
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, color=colors[:len(means)], alpha=0.8, edgecolor='black', linewidth=1.5,
                   yerr=stds, capsize=8, error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add value labels on top of bars (mean ± std)
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Styling
    ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0, 1.1)  # Increased to accommodate error bars
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add n= annotation (show range if different)
    if len(set(n_runs)) == 1:
        n_text = f'n={n_runs[0]} runs per config'
    else:
        n_text = f'n={min(n_runs)}-{max(n_runs)} runs'
    
    ax.text(0.02, 0.98, n_text, 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save plot
output_path = base_dir / 'comparison_plots_with_std.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_path}")

# Also save as PDF for better quality
output_path_pdf = base_dir / 'comparison_plots_with_std.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight')
print(f"✅ PDF saved to: {output_path_pdf}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (Mean ± Std Dev)")
print("="*60)
for config in configs:
    if config not in stats:
        continue
    print(f"\n{config} ({config_labels[config].replace(chr(10), ' ')}):")
    for metric in metrics:
        mean = stats[config][metric]['mean']
        std = stats[config][metric]['std']
        n = stats[config][metric]['n']
        print(f"  {metric_titles[metric]:25s}: {mean:.3f} ± {std:.3f} (n={n})")

plt.show()
