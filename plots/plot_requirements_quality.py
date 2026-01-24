"""
Script para generar gráficos de barras con los resultados de la evaluación de calidad.
Versión 3: Con promedios de múltiples corridas y colores por modelo
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Colores por modelo
MODEL_COLORS = {
    "gpt-4o-mini": "#e74c3c",      # Rojo
    "gemini-2.0-flash": "#3498db",  # Azul
    "gemini-3-pro": "#27ae60"       # Verde
}

MODEL_EDGE_COLORS = {
    "gpt-4o-mini": "#c0392b",
    "gemini-2.0-flash": "#2980b9", 
    "gemini-3-pro": "#1e8449"
}


def load_results_multi_run():
    """Carga los resultados de múltiples corridas y calcula promedios."""
    all_runs = []
    
    # Buscar todos los archivos JSON de evaluación
    for json_file in OUTPUT_BASE.glob("requirements_quality_evaluation_run*.json"):
        with open(json_file, 'r') as f:
            all_runs.append(json.load(f))
    
    # Si no hay múltiples corridas, buscar el archivo único
    if not all_runs:
        results_file = OUTPUT_BASE / "requirements_quality_evaluation.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_runs.append(json.load(f))
    
    return all_runs


def create_quantity_chart(all_runs: list):
    """Crea gráfico de PRECISIÓN con promedios de múltiples corridas - SEPARADOS POR EJERCICIO."""
    # Organizar datos por dataset y modelo
    stats_by_dataset = defaultdict(lambda: defaultdict(list))
    
    for run_data in all_runs:
        datasets = {}
        for detail in run_data.get("detailed", []):
            dataset = detail["dataset"]
            if dataset not in datasets:
                datasets[dataset] = {"gpt-4o-mini": [], "gemini-2.0-flash": [], "gemini-3-pro": []}
            datasets[dataset][detail["model"]].append(detail)
        
        # Calcular métricas para cada dataset y modelo en esta corrida
        for dataset_name, model_data in datasets.items():
            for model, details in model_data.items():
                # Contar requerimientos únicos relacionados
                seen_teacher_reqs = set()
                unique_related = 0
                
                for d in details:
                    if d.get("related_to_teacher"):
                        teacher_req_num = d.get("related_teacher_req_number")
                        if teacher_req_num is not None and teacher_req_num not in seen_teacher_reqs:
                            unique_related += 1
                            seen_teacher_reqs.add(teacher_req_num)
                
                total = len(details)
                pct = (unique_related / total * 100) if total > 0 else 0
                
                stats_by_dataset[dataset_name][model].append({
                    'pct': pct,
                    'related': unique_related,
                    'total': total
                })
    
    # Crear gráficos para cada dataset
    for dataset_name in sorted(stats_by_dataset.keys()):
        display_name = dataset_name.replace('ej1-', '')
        
        # Cargar criterios del docente
        teacher_reqs_path = OUTPUT_BASE / dataset_name / "teacher_requirements.txt"
        max_criteria = 0
        if teacher_reqs_path.exists():
            with open(teacher_reqs_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('- Requirement'):
                        max_criteria += 1
        
        models = sorted(stats_by_dataset[dataset_name].keys())
        
        print(f"\n=== PRECISION - Dataset: {dataset_name} ===")
        
        avg_percentages = []
        std_percentages = []
        avg_related = []
        avg_total = []
        
        for model in models:
            runs = stats_by_dataset[dataset_name][model]
            avg_pct = np.mean([r['pct'] for r in runs])
            std_pct = np.std([r['pct'] for r in runs])
            avg_rel = np.mean([r['related'] for r in runs])
            avg_tot = np.mean([r['total'] for r in runs])
            
            print(f"  {model}:")
            print(f"    related: {[r['related'] for r in runs]} → avg={avg_rel:.1f}")
            print(f"    total: {[r['total'] for r in runs]} → avg={avg_tot:.1f}")
            
            avg_percentages.append(avg_pct)
            std_percentages.append(std_pct)
            avg_related.append(avg_rel)
            avg_total.append(avg_tot)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        colors = [MODEL_COLORS[m] for m in models]
        edge_colors = [MODEL_EDGE_COLORS[m] for m in models]
        
        bars = ax.bar(x, avg_percentages, width=0.6, color=colors, linewidth=2, 
                     edgecolor=edge_colors, yerr=std_percentages, capsize=5, 
                     error_kw={'linewidth': 2})
        
        ax.set_xlabel('Modelo', fontsize=14, fontweight='bold')
        ax.set_ylabel('Porcentaje de requerimientos relacionados (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Precisión: Requerimientos Relacionados con Docente - {display_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{avg_related[i]:.1f}/{avg_total[i]:.0f}\n({height:.0f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                       xytext=(0, 0),
                       textcoords="offset points",
                       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        plt.tight_layout()
        safe_name = display_name.replace('/', '_')
        output_path = PLOTS_DIR / f"precision_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Precision chart for {display_name} saved to: {output_path}")
        plt.close()


def create_diversity_chart(all_runs: list):
    """Crea gráfico de DIVERSIDAD con promedios - SEPARADOS POR EJERCICIO."""
    # Organizar datos por dataset y modelo
    stats_by_dataset = defaultdict(lambda: defaultdict(list))
    
    for run_data in all_runs:
        datasets = {}
        for detail in run_data.get("detailed", []):
            dataset = detail["dataset"]
            if dataset not in datasets:
                datasets[dataset] = {"gpt-4o-mini": [], "gemini-2.0-flash": [], "gemini-3-pro": []}
            datasets[dataset][detail["model"]].append(detail)
        
        # Calcular diversidad para cada dataset y modelo en esta corrida
        for dataset_name, model_data in datasets.items():
            for model, details in model_data.items():
                unique_criteria = len(set(d.get("related_teacher_req_number") for d in details 
                                         if d.get("related_to_teacher")))
                stats_by_dataset[dataset_name][model].append(unique_criteria)
    
    # Crear gráficos para cada dataset
    for dataset_name in sorted(stats_by_dataset.keys()):
        display_name = dataset_name.replace('ej1-', '')
        
        # Cargar criterios del docente
        teacher_reqs_path = OUTPUT_BASE / dataset_name / "teacher_requirements.txt"
        max_criteria = 0
        if teacher_reqs_path.exists():
            with open(teacher_reqs_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('- Requirement'):
                        max_criteria += 1
        
        print(f"Dataset: {dataset_name}, max_criteria (teacher): {max_criteria}")
        
        models = sorted(stats_by_dataset[dataset_name].keys())
        
        avg_unique = []
        std_unique = []
        
        for model in models:
            runs = stats_by_dataset[dataset_name][model]
            avg = np.mean(runs)
            avg_unique.append(avg)
            std_unique.append(np.std(runs))
            print(f"  {model}: runs={runs}, avg={avg:.2f}")

        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        colors = [MODEL_COLORS[m] for m in models]
        edge_colors = [MODEL_EDGE_COLORS[m] for m in models]
        
        bars = ax.bar(x, avg_unique, width=0.6, color=colors, linewidth=2,
                     edgecolor=edge_colors, yerr=std_unique, capsize=5,
                     error_kw={'linewidth': 2})
        
        ax.set_xlabel('Modelo', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cantidad de criterios', fontsize=14, fontweight='bold')
        ax.set_title(f'Diversidad: Criterios Únicos Cubiertos - {display_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = (height / max_criteria * 100) if max_criteria > 0 else 0
            ax.annotate(f'{height:.1f}/{max_criteria}\n({pct:.0f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, -100),
                       textcoords="offset points",
                       ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        
        plt.tight_layout()
        safe_name = display_name.replace('/', '_')
        output_path = PLOTS_DIR / f"diversidad_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Diversity chart for {display_name} saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    print("Loading evaluation results from multiple runs...")
    all_runs = load_results_multi_run()
    print(f"Found {len(all_runs)} run(s)")
    
    if not all_runs:
        print("ERROR: No evaluation results found!")
        exit(1)
    
    print("\nCreating quantity charts (with averages)...")
    create_quantity_chart(all_runs)
    
    print("Creating diversity charts (with averages)...")
    create_diversity_chart(all_runs)
    
    print("\nDone!")
