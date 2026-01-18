#!/usr/bin/env python3
"""
Generate comparison plots for experiments with YAML-based configuration.
Supports dynamic sorting and custom labeling.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from glob import glob
import argparse
import sys
import yaml
import re

def is_timestamp_folder(folder_name: str) -> bool:
    """Check if a folder name matches timestamp format: YYYYMMDDTHHMMSS"""
    timestamp_pattern = r'^\d{8}T\d{6}$'
    return bool(re.match(timestamp_pattern, folder_name))

def load_metrics(base_dir: Path, config: str):
    """Load all aggregate_metrics.json files for a given configuration."""
    pattern = str(base_dir / config / "*" / "aggregate_metrics.json")
    files = glob(pattern)
    
    # Filter to only include timestamp folders
    filtered_files = []
    for file_path in files:
        folder_name = Path(file_path).parent.name
        if is_timestamp_folder(folder_name):
            filtered_files.append(file_path)
    
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
                    'average_overall_score': data['average_overall_score'],
                    'total_tokens': data.get('token_usage', {}).get('total_tokens', 0),
                    'total_cost': data.get('token_usage', {}).get('estimated_cost_usd', 0.0)
                })
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            
    return metrics_list

def calculate_stats(experiments, base_dir):
    """Calculate statistics for all experiments."""
    stats = {}
    
    for exp in experiments:
        config_name = exp['config']
        runs = load_metrics(base_dir, config_name)
        
        if not runs:
            continue
            
        stats[config_name] = {}
        for metric in ['completeness', 'content_similarity', 'restraint', 'average_overall_score', 'total_tokens', 'total_cost']:
            values = [run[metric] for run in runs]
            stats[config_name][metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'n': len(values)
            }
    
    return stats

def get_sorted_experiments(experiments, stats, metric, sorting_config):
    """
    Sort experiments based on configuration.
    Returns: List of (config_name, label) tuples
    """
    sort_by = sorting_config.get('by', 'default')
    order = sorting_config.get('order', 'asc')
    reverse = (order == 'desc')

    valid_experiments = [e for e in experiments if e['config'] in stats]
    
    if sort_by == 'default':
        sorted_exps = valid_experiments
        if reverse:
            sorted_exps = list(reversed(sorted_exps))
            
    elif sort_by == 'alphabetic':
        sorted_exps = sorted(valid_experiments, key=lambda x: x['label'], reverse=reverse)
        
    elif sort_by == 'score':
        # Sort by the specific metric being plotted (independent for each plot)
        sorted_exps = sorted(
            valid_experiments, 
            key=lambda x: stats[x['config']][metric]['mean'], 
            reverse=reverse
        )
        
    elif sort_by == 'overall_score':
        # Sort by average_overall_score
        sorted_exps = sorted(
            valid_experiments, 
            key=lambda x: stats[x['config']]['average_overall_score']['mean'], 
            reverse=reverse
        )
        
    elif sort_by in ['completeness', 'content_similarity', 'restraint', 'total_tokens', 'total_cost']:
        # Sort by a specific fixed metric
        sorted_exps = sorted(
            valid_experiments, 
            key=lambda x: stats[x['config']][sort_by]['mean'], 
            reverse=reverse
        )
        
    else:
        print(f"⚠️  Unknown sorting method '{sort_by}', using default.")
        sorted_exps = valid_experiments

    return [(e['config'], e['label']) for e in sorted_exps]

def plot_all(config_data):
    """Generate plots based on YAML configuration."""
    base_dir = Path(config_data['base_dir'])
    experiments = config_data['experiments']
    sorting = config_data.get('sorting', {'by': 'default', 'order': 'asc'})
    
    print(f"Loading metrics from {base_dir}...")
    stats = calculate_stats(experiments, base_dir)
    
    if not stats:
        print("❌ No data found for any experiment!")
        sys.exit(1)

    metric_titles = {
        'completeness': 'Completeness',
        'content_similarity': 'Content Similarity',
        'restraint': 'Restraint',
        'average_overall_score': 'Average Overall Score',
        'total_tokens': 'Total Tokens',
        'total_cost': 'Total Cost ($)'
    }
    
    metrics = ['completeness', 'content_similarity', 'restraint', 'average_overall_score', 'total_tokens', 'total_cost']
    
    # Plotting setup
    # 3x2 grid to accommodate 6 plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f"{config_data.get('title', 'Experiment Comparison')}\n(Mean ± Std Dev)", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    base_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                   '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400']
    
    # Assign consistent colors to each experiment config
    all_configs = [exp['config'] for exp in experiments]
    import itertools
    config_color_map = {config: color for config, color in zip(all_configs, itertools.cycle(base_colors))}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get sorted data for this subplot
        sorted_items = get_sorted_experiments(experiments, stats, metric, sorting)
        
        configs = [item[0] for item in sorted_items]
        labels = [item[1] for item in sorted_items]
        
        means = [stats[c][metric]['mean'] for c in configs]
        stds = [stats[c][metric]['std'] for c in configs]
        n_runs = [stats[c][metric]['n'] for c in configs]
        
        # Plot bars
        x_pos = np.arange(len(configs))
        bar_colors = [config_color_map[c] for c in configs]
        
        bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                      yerr=stds, capsize=8, error_kw={'linewidth': 2, 'elinewidth': 2})
        
        # Add labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}\n±{std:.3f}' if metric != 'total_tokens' else f'{mean:.0f}\n±{std:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            
        ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold')
        
        # Dynamic Y-limit
        if metric in ['total_tokens', 'total_cost']:
            # Auto scale with 10% padding
            ax.autoscale(axis='y')
            # Ensure 0 is at bottom
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1] * 1.15)
        else:
            ax.set_ylim(0, 1.1)
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save output
    output_base = config_data.get('output_path', 'comparison_plot')
    output_dir = Path(output_base).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight')
    print(f"\n✅ Plots saved to {output_base}.png and .pdf")

def main():
    parser = argparse.ArgumentParser(description='YAML-configured plotting script')
    parser.add_argument('config_file', help='Path to YAML configuration file')
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    plot_all(config)

if __name__ == '__main__':
    main()
