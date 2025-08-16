#!/usr/bin/env python3
"""
MLflow Metrics Viewer

This script provides utilities to view and analyze MLflow metrics and artifacts
from the assignment evaluation pipeline.
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Any

def load_mlflow_config(config_path: str = "src/config/mlflow_config.yaml") -> Dict[str, Any]:
    """Load MLflow configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('mlflow', {})
    except Exception as e:
        print(f"Warning: Could not load MLflow config: {e}")
        return {}

def setup_mlflow(config: Dict[str, Any]):
    """Setup MLflow with configuration"""
    tracking_uri = config.get('tracking_uri')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Set MLflow tracking URI: {tracking_uri}")
    
    experiment_name = config.get('experiment_name', 'assignment_evaluation')
    mlflow.set_experiment(experiment_name)
    print(f"Set MLflow experiment: {experiment_name}")

def list_runs():
    """List all runs in the current experiment"""
    experiment = mlflow.get_experiment_by_name(mlflow.get_experiment().name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    print(f"\n📊 MLflow Runs Summary")
    print("=" * 50)
    print(f"Total runs: {len(runs)}")
    
    if len(runs) > 0:
        print("\nRecent runs:")
        for i, (_, run) in enumerate(runs.head(10).iterrows()):
            run_id = run['run_id']
            run_name = run.get('tags.mlflow.runName', 'Unnamed')
            status = run.get('status', 'Unknown')
            start_time = pd.to_datetime(run.get('start_time', 0), unit='ms')
            
            print(f"  {i+1}. {run_name} ({run_id[:8]}...) - {status} - {start_time}")

def show_run_details(run_id: str):
    """Show detailed information about a specific run"""
    try:
        run = mlflow.get_run(run_id)
        
        print(f"\n🔍 Run Details: {run.info.run_name}")
        print("=" * 50)
        print(f"Run ID: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print(f"Start Time: {pd.to_datetime(run.info.start_time, unit='ms')}")
        print(f"End Time: {pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else 'Running'}")
        
        # Show parameters
        if run.data.params:
            print(f"\n📋 Parameters:")
            for key, value in run.data.params.items():
                print(f"  {key}: {value}")
        
        # Show metrics
        if run.data.metrics:
            print(f"\n📈 Metrics:")
            for key, value in run.data.metrics.items():
                print(f"  {key}: {value}")
        
        # Show tags
        if run.data.tags:
            print(f"\n🏷️  Tags:")
            for key, value in run.data.tags.items():
                if not key.startswith('mlflow.'):
                    print(f"  {key}: {value}")
        
        # List artifacts
        artifacts = mlflow.list_artifacts(run_id)
        if artifacts:
            print(f"\n📁 Artifacts:")
            for artifact in artifacts:
                print(f"  {artifact.path}")
        
    except Exception as e:
        print(f"Error getting run details: {e}")

def compare_runs(run_ids: List[str]):
    """Compare multiple runs"""
    if len(run_ids) < 2:
        print("Need at least 2 run IDs to compare")
        return
    
    runs_data = []
    for run_id in run_ids:
        try:
            run = mlflow.get_run(run_id)
            run_data = {
                'run_id': run_id,
                'run_name': run.info.run_name,
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                **run.data.params,
                **run.data.metrics
            }
            runs_data.append(run_data)
        except Exception as e:
            print(f"Error getting run {run_id}: {e}")
    
    if runs_data:
        df = pd.DataFrame(runs_data)
        print(f"\n📊 Run Comparison")
        print("=" * 50)
        print(df.to_string(index=False))

def plot_metrics(metric_name: str, experiment_name: str = None):
    """Plot a specific metric across runs"""
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    experiment = mlflow.get_experiment_by_name(mlflow.get_experiment().name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if metric_name not in runs.columns:
        print(f"Metric '{metric_name}' not found in runs")
        print(f"Available metrics: {[col for col in runs.columns if col.startswith('metrics.')]}")
        return
    
    # Filter runs that have this metric
    metric_runs = runs[runs[metric_name].notna()]
    
    if len(metric_runs) == 0:
        print(f"No runs found with metric '{metric_name}'")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(metric_runs.index, metric_runs[metric_name], 'o-')
    plt.title(f'{metric_name} Across Runs')
    plt.xlabel('Run Index')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    output_path = Path('mlflow_plots')
    output_path.mkdir(exist_ok=True)
    plot_file = output_path / f"{metric_name.replace('.', '_')}_plot.png"
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")
    plt.show()

def export_run_data(run_id: str, output_dir: str = "mlflow_exports"):
    """Export all data from a run to local files"""
    try:
        run = mlflow.get_run(run_id)
        output_path = Path(output_dir) / run_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export run info
        run_info = {
            'run_id': run.info.run_id,
            'run_name': run.info.run_name,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'parameters': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'tags': dict(run.data.tags)
        }
        
        import json
        with open(output_path / 'run_info.json', 'w') as f:
            json.dump(run_info, f, indent=2, default=str)
        
        # Download artifacts
        artifacts = mlflow.list_artifacts(run_id)
        for artifact in artifacts:
            local_path = mlflow.artifacts.download_artifacts(run_id, artifact.path)
            print(f"Downloaded artifact: {artifact.path}")
        
        print(f"Run data exported to: {output_path}")
        
    except Exception as e:
        print(f"Error exporting run data: {e}")

def main():
    parser = argparse.ArgumentParser(description="MLflow Metrics Viewer")
    parser.add_argument("--config", default="src/config/mlflow_config.yaml", help="MLflow config file")
    parser.add_argument("--list-runs", action="store_true", help="List all runs")
    parser.add_argument("--run-id", help="Show details for specific run")
    parser.add_argument("--compare", nargs='+', help="Compare multiple runs")
    parser.add_argument("--plot-metric", help="Plot a specific metric")
    parser.add_argument("--export", help="Export run data")
    parser.add_argument("--output-dir", default="mlflow_exports", help="Output directory for exports")
    
    args = parser.parse_args()
    
    # Load and setup MLflow
    config = load_mlflow_config(args.config)
    setup_mlflow(config)
    
    if args.list_runs:
        list_runs()
    elif args.run_id:
        show_run_details(args.run_id)
    elif args.compare:
        compare_runs(args.compare)
    elif args.plot_metric:
        plot_metrics(args.plot_metric)
    elif args.export:
        export_run_data(args.export, args.output_dir)
    else:
        # Default: list runs
        list_runs()

if __name__ == "__main__":
    main()
