#!/usr/bin/env python3
"""
Generate comparison plots for experiments with standard deviation.
Dynamically reads ALL aggregate_metrics.json files from each configuration directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from glob import glob
import argparse
import sys
import re


def is_timestamp_folder(folder_name: str) -> bool:
    """
    Check if a folder name matches timestamp format: YYYYMMDDTHHMMSS
    Example: 20260104T193421
    """
    timestamp_pattern = r'^\d{8}T\d{6}$'
    return bool(re.match(timestamp_pattern, folder_name))


def load_all_metrics(base_dir: Path, config: str):
    """Load all aggregate_metrics.json files for a given configuration.
    Only includes folders that match timestamp format (YYYYMMDDTHHMMSS).
    """
    pattern = str(base_dir / config / "*" / "aggregate_metrics.json")
    files = glob(pattern)
    
    # Filter to only include timestamp folders
    filtered_files = []
    for file_path in files:
        folder_name = Path(file_path).parent.name
        if is_timestamp_folder(folder_name):
            filtered_files.append(file_path)
        else:
            print(f"⏭️  Skipping non-timestamp folder: {folder_name}")
    
    files = filtered_files
    
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


def calculate_statistics(all_data, configs):
    """Calculate mean and standard deviation for each metric across all configurations."""
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
    
    return stats, metrics


def create_config_labels(configs):
    """Generate readable labels from configuration names."""
    labels = {}
    for config in configs:
        # Try to parse format like "0c_0i" -> "0 Correct\n0 Erroneous"
        parts = config.split('_')
        if len(parts) == 2 and parts[0].endswith('c') and parts[1].endswith('i'):
            correct = parts[0][:-1]
            erroneous = parts[1][:-1]
            labels[config] = f'{correct} Correct\n{erroneous} Erroneous'
        else:
            # Use folder name as-is if it doesn't match expected format
            labels[config] = config.replace('_', ' ').title()
    
    return labels


def plot_comparison(base_dir, configs, plot_title=None, output_filename='comparison_plots'):
    """
    Create comparison plots for experiments.
    
    Args:
        base_dir: Path to the base directory containing experiment folders
        configs: List of configuration folder names to compare
        plot_title: Title for the overall plot (default: uses base_dir name)
        output_filename: Base name for output files (default: 'comparison_plots')
    """
    base_dir = Path(base_dir)
    
    # Use base directory name as title if not provided
    if plot_title is None:
        plot_title = base_dir.name.replace('_', ' ').title()
    
    # Load all data dynamically
    print("Loading metrics from all runs...")
    print("="*60)
    all_data = {}
    for config in configs:
        all_data[config] = load_all_metrics(base_dir, config)
    
    print("="*60)
    
    # Calculate statistics
    stats, metrics = calculate_statistics(all_data, configs)
    
    # Generate configuration labels
    config_labels = create_config_labels(configs)
    
    metric_titles = {
        'completeness': 'Completeness',
        'content_similarity': 'Content Similarity',
        'restraint': 'Restraint',
        'average_overall_score': 'Average Overall Score'
    }
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{plot_title} (Mean ± Std Dev)', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Color palette - extend if more configs are needed
    base_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                   '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
    colors = base_colors[:len(configs)]
    
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
    output_path = base_dir / f'{output_filename}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Also save as PDF for better quality
    output_path_pdf = base_dir / f'{output_filename}.pdf'
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


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate comparison plots for experiments with standard deviation.'
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing experiment configuration folders'
    )
    parser.add_argument(
        'configs',
        nargs='+',
        help='Configuration folder names to compare'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title for the plot (default: use base directory name)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_plots',
        help='Output filename (without extension, default: comparison_plots)'
    )
    
    args = parser.parse_args()
    
    # Validate base directory exists
    if not Path(args.base_dir).exists():
        print(f"❌ Error: Directory '{args.base_dir}' does not exist!")
        sys.exit(1)
    
    # Generate plots
    plot_comparison(
        base_dir=args.base_dir,
        configs=args.configs,
        plot_title=args.title,
        output_filename=args.output
    )


if __name__ == '__main__':
    main()
