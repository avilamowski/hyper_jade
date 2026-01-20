#!/usr/bin/env python3
"""
Generate TWO separate comparison plots:
1. Evaluation metrics (Completeness, Content Similarity, Restraint, Overall Score)
2. Efficiency metrics (Total Tokens, Total Cost, Latency) - WITHOUT std dev
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
    """Check if a folder name matches timestamp format: YYYYMMDDTHHMMSS"""
    timestamp_pattern = r'^\d{8}T\d{6}$'
    return bool(re.match(timestamp_pattern, folder_name))


def load_all_metrics(base_dir: Path, config: str):
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
                    'total_cost': data.get('token_usage', {}).get('estimated_cost_usd', 0.0),
                    'latency': data.get('latency_seconds', 0.0)  # If available
                })
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
    
    print(f"✓ Loaded {len(metrics_list)} runs for {config}")
    return metrics_list


def calculate_statistics(all_data, configs, metrics):
    """Calculate mean and standard deviation for each metric across all configurations."""
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
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'n': len(values)
            }
    
    return stats


def create_config_labels(configs):
    """Generate readable labels from configuration names."""
    labels = {
        'gpt4o_mini': 'GPT-4o Mini',
        'gemini_2_0_flash': 'Gemini 2.0 Flash',
        'gemini_2_5_pro': 'Gemini 2.5 Pro',
        'gemini_3_flash': 'Gemini 3 Flash Preview',
    }
    
    return {config: labels.get(config, config.replace('_', ' ').title()) for config in configs}


def plot_evaluation_metrics(base_dir, configs, plot_title, output_filename):
    """Create plot for evaluation metrics (2x2 grid)."""
    base_dir = Path(base_dir)
    
    # Load all data
    print("\n" + "="*60)
    print("GENERATING EVALUATION METRICS PLOT")
    print("="*60)
    all_data = {}
    for config in configs:
        all_data[config] = load_all_metrics(base_dir, config)
    
    # Calculate statistics for evaluation metrics only
    eval_metrics = ['completeness', 'content_similarity', 'restraint', 'average_overall_score']
    stats = calculate_statistics(all_data, configs, eval_metrics)
    
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
    fig.suptitle(f'{plot_title}\n(Mean ± Std Dev)', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Color palette
    base_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71']
    config_color_map = {config: base_colors[i] for i, config in enumerate(configs)}
    
    # Create bar plots with error bars
    for idx, metric in enumerate(eval_metrics):
        ax = axes[idx]
        
        # Extract data for this metric
        plot_data = []
        for config in configs:
            if config in stats:
                plot_data.append({
                    'mean': stats[config][metric]['mean'],
                    'std': stats[config][metric]['std'],
                    'label': config_labels[config],
                    'n': stats[config][metric]['n'],
                    'color': config_color_map[config]
                })
        
        means = [d['mean'] for d in plot_data]
        stds = [d['std'] for d in plot_data]
        labels = [d['label'] for d in plot_data]
        n_runs = [d['n'] for d in plot_data]
        bar_colors = [d['color'] for d in plot_data]
        
        # Create bars with error bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                       yerr=stds, capsize=8, error_kw={'linewidth': 2, 'elinewidth': 2})
        
        # Add value labels on top of bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}\n±{std:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Styling
        ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add n= annotation
        n_text = f'n={n_runs[0]} runs per config'
        ax.text(0.02, 0.98, n_text, 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot (PNG only) to plots/ directory
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{output_filename}_evaluation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Evaluation metrics plot saved to: {output_path}")
    
    plt.close()
    
    return stats, all_data


def plot_efficiency_metrics(base_dir, configs, all_data, plot_title, output_filename, latencies=None):
    """Create plot for efficiency metrics (1x3 grid for Tokens, Cost, Latency) WITHOUT std dev."""
    base_dir = Path(base_dir)
    
    print("\n" + "="*60)
    print("GENERATING EFFICIENCY METRICS PLOT")
    print("="*60)
    
    # Calculate statistics for efficiency metrics
    eff_metrics = ['total_tokens', 'total_cost']
    if latencies:
        eff_metrics.append('latency')
    
    stats = calculate_statistics(all_data, configs, eff_metrics)
    
    # If latencies provided externally, add them to stats
    if latencies:
        for config in configs:
            if config in latencies and config in stats:
                lats = latencies[config]
                stats[config]['latency'] = {
                    'mean': np.mean(lats),
                    'std': np.std(lats, ddof=1) if len(lats) > 1 else 0,
                    'n': len(lats)
                }
    
    # Generate configuration labels
    config_labels = create_config_labels(configs)
    
    metric_titles = {
        'total_tokens': 'Average Total Tokens',
        'total_cost': 'Average Cost (USD)',
        'latency': 'Average Time (s)'
    }
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Efficiency Analysis: Cost, Time, and Token Consumption', 
                 fontsize=16, fontweight='bold')
    
    # Color palette
    base_colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71']
    config_color_map = {config: base_colors[i] for i, config in enumerate(configs)}
    
    # Create bar plots WITHOUT error bars
    for idx, metric in enumerate(eff_metrics):
        ax = axes[idx]
        
        # Extract data for this metric
        plot_data = []
        for config in configs:
            if config in stats and metric in stats[config]:
                plot_data.append({
                    'mean': stats[config][metric]['mean'],
                    'std': stats[config][metric]['std'],
                    'label': config_labels[config],
                    'n': stats[config][metric]['n'],
                    'color': config_color_map[config]
                })
        
        if not plot_data:
            continue
            
        means = [d['mean'] for d in plot_data]
        stds = [d['std'] for d in plot_data]
        labels = [d['label'] for d in plot_data]
        n_runs = [d['n'] for d in plot_data]
        bar_colors = [d['color'] for d in plot_data]
        
        # Create bars WITH error bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                       yerr=stds, capsize=8, error_kw={'linewidth': 2, 'elinewidth': 2})
        
        # Add value labels on top of bars (mean on first line, ±std on second line)
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            if metric == 'total_tokens':
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                        f'{mean:.0f}\n±{std:.0f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            elif metric == 'total_cost':
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                        f'${mean:.4f}\n±${std:.4f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            else:  # latency
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                        f'{mean:.1f}s\n±{std:.1f}s',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Styling
        ax.set_title(metric_titles[metric], fontsize=14, fontweight='bold', pad=10)
        
        if metric == 'total_tokens':
            ax.set_ylabel('Tokens', fontsize=11)
        elif metric == 'total_cost':
            ax.set_ylabel('Cost (USD)', fontsize=11)
        else:
            ax.set_ylabel('Time (seconds)', fontsize=11)
        
        # Auto scale with padding
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1] * 1.15)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add n= annotation
        n_text = f'n={n_runs[0]} runs per config'
        ax.text(0.02, 0.98, n_text, 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot (PNG only) to plots/ directory
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{output_filename}_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Efficiency metrics plot saved to: {output_path}")
    
    plt.close()


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate separate evaluation and efficiency comparison plots.'
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
        help='Title for the plots (default: use base directory name)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison',
        help='Output filename base (without extension, default: comparison)'
    )
    parser.add_argument(
        '--latencies',
        type=str,
        default=None,
        help='JSON file with latency data (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate base directory exists
    if not Path(args.base_dir).exists():
        print(f"❌ Error: Directory '{args.base_dir}' does not exist!")
        sys.exit(1)
    
    # Use base directory name as title if not provided
    if args.title is None:
        plot_title = Path(args.base_dir).name.replace('_', ' ').title()
    else:
        plot_title = args.title
    
    # Load latencies if provided
    latencies = None
    if args.latencies:
        with open(args.latencies, 'r') as f:
            latencies = json.load(f)
    
    # Generate evaluation metrics plot
    stats, all_data = plot_evaluation_metrics(
        base_dir=args.base_dir,
        configs=args.configs,
        plot_title=plot_title,
        output_filename=args.output
    )
    
    # Generate efficiency metrics plot
    plot_efficiency_metrics(
        base_dir=args.base_dir,
        configs=args.configs,
        all_data=all_data,
        plot_title=plot_title,
        output_filename=args.output,
        latencies=latencies
    )
    
    print("\n" + "="*60)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60)


if __name__ == '__main__':
    main()
