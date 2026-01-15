"""
Script para generar gráficos de barras con los resultados de la evaluación de calidad.
Versión 2: Gráficos SEPARADOS (Cantidad y Diversidad)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def load_results():
    """Carga los resultados de la evaluación."""
    results_file = OUTPUT_BASE / "requirements_quality_evaluation.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def create_quantity_chart(results: dict):
    """Crea gráfico de PRECISIÓN: Porcentaje de requerimientos relacionados - SEPARADOS POR EJERCICIO."""
    datasets = {}
    for detail in results.get("detailed", []):
        dataset = detail["dataset"]
        if dataset not in datasets:
            datasets[dataset] = {"gpt-4o-mini": [], "gemini-2.0-flash": [], "gemini-3-pro": []}
        datasets[dataset][detail["model"]].append(detail)
    
    for dataset_name, model_data in sorted(datasets.items()):
        models = list(model_data.keys())
        
        percentages = []
        totals = []
        relateds = []
        
        for model in models:
            details = model_data[model]
            total = len(details)
            rel_count = sum(1 for d in details if d.get("related_to_teacher"))
            pct = (rel_count / total * 100) if total > 0 else 0
            percentages.append(pct)
            totals.append(total)
            relateds.append(rel_count)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(models))
        
        bars = ax.bar(x, percentages, width=0.6, color='#3498db', linewidth=2, edgecolor='#2980b9')
        
        ax.set_xlabel('Modelo', fontsize=14, fontweight='bold')
        ax.set_ylabel('Porcentaje de requerimientos relacionados (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Precisión: Requerimientos Relacionados con Docente - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{relateds[i]}/{totals[i]}\n({height:.0f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                       xytext=(0, 0),
                       textcoords="offset points",
                       ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        plt.tight_layout()
        safe_name = dataset_name.replace('/', '_')
        output_path = PLOTS_DIR / f"precision_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Precision chart for {dataset_name} saved to: {output_path}")
        plt.close()


def create_diversity_chart(results: dict):
    """Crea gráfico de DIVERSIDAD: Criterios Únicos Cubiertos - SEPARADOS POR EJERCICIO."""
    datasets = {}
    for detail in results.get("detailed", []):
        dataset = detail["dataset"]
        if dataset not in datasets:
            datasets[dataset] = {"gpt-4o-mini": [], "gemini-2.0-flash": [], "gemini-3-pro": []}
        datasets[dataset][detail["model"]].append(detail)
    
    for dataset_name, model_data in sorted(datasets.items()):
        # Cargar criterios del docente
        teacher_reqs_path = OUTPUT_BASE / dataset_name / "teacher_requirements.txt"
        max_criteria = 0
        if teacher_reqs_path.exists():
            with open(teacher_reqs_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('- Requirement'):
                        max_criteria += 1
        
        models = list(model_data.keys())
        unique = []
        
        for model in models:
            details = model_data[model]
            unique_criteria = len(set(d.get("related_teacher_req_number") for d in details 
                                     if d.get("related_to_teacher")))
            unique.append(unique_criteria)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        bars = ax.bar(x, unique, width=0.6, color='#9b59b6', linewidth=2, edgecolor='#8e44ad')
        
        ax.set_xlabel('Modelo', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cantidad de criterios', fontsize=14, fontweight='bold')
        ax.set_title(f'Diversidad: Criterios Únicos Cubiertos - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            pct = (height / max_criteria * 100) if max_criteria > 0 else 0
            ax.annotate(f'{int(height)}/{max_criteria}\n({pct:.0f}%)',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, -25),
                       textcoords="offset points",
                       ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        
        plt.tight_layout()
        safe_name = dataset_name.replace('/', '_')
        output_path = PLOTS_DIR / f"diversidad_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Diversity chart for {dataset_name} saved to: {output_path}")
        plt.close()


def create_stacked_bar_chart(results: dict):
    """Crea gráfico STACKED con desglose: No relacionado, Duplicado, Único."""
    datasets = {}
    for detail in results.get("detailed", []):
        dataset = detail["dataset"]
        if dataset not in datasets:
            datasets[dataset] = {"gpt-4o-mini": [], "gemini-2.0-flash": [], "gemini-3-pro": []}
        datasets[dataset][detail["model"]].append(detail)
    
    for dataset_name, model_data in sorted(datasets.items()):
        models = list(model_data.keys())
        not_related_counts = []
        duplicated_counts = []
        unique_counts = []
        
        for model in models:
            details = model_data[model]
            
            # Calculate duplicates on-the-fly if not in data
            seen_teacher_reqs = set()
            not_related = 0
            duplicated = 0
            unique = 0
            
            for d in details:
                if not d.get("related_to_teacher"):
                    not_related += 1
                else:
                    # Check if we've seen this teacher requirement before
                    teacher_req_num = d.get("related_teacher_req_number")
                    if teacher_req_num is not None:
                        if teacher_req_num in seen_teacher_reqs:
                            duplicated += 1
                        else:
                            unique += 1
                            seen_teacher_reqs.add(teacher_req_num)
                    else:
                        # Fallback: use is_duplicated field if present
                        if d.get("is_duplicated", False):
                            duplicated += 1
                        else:
                            unique += 1
            
            not_related_counts.append(not_related)
            duplicated_counts.append(duplicated)
            unique_counts.append(unique)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.6
        
        bars1 = ax.bar(x, not_related_counts, width, label='No relacionado', color='#e74c3c', linewidth=2, edgecolor='#c0392b')
        bars2 = ax.bar(x, duplicated_counts, width, bottom=not_related_counts, label='Duplicado', color='#f39c12', linewidth=2, edgecolor='#d68910')
        
        bottom_for_unique = np.array(not_related_counts) + np.array(duplicated_counts)
        bars3 = ax.bar(x, unique_counts, width, bottom=bottom_for_unique, label='Único', color='#27ae60', linewidth=2, edgecolor='#1e8449')
        
        ax.set_xlabel('Modelo', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cantidad de requerimientos', fontsize=14, fontweight='bold')
        ax.set_title(f'Desglose de Requerimientos - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add numbers inside bars
        for i, (n, d, u) in enumerate(zip(not_related_counts, duplicated_counts, unique_counts)):
            if n > 0:
                ax.text(i, n/2, str(int(n)), ha='center', va='center', fontweight='bold', fontsize=10)
            if d > 0:
                ax.text(i, n + d/2, str(int(d)), ha='center', va='center', fontweight='bold', fontsize=10)
            if u > 0:
                ax.text(i, n + d + u/2, str(int(u)), ha='center', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        safe_name = dataset_name.replace('/', '_')
        output_path = PLOTS_DIR / f"requirements_quality_stacked_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Stacked chart for {dataset_name} saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    print("Loading evaluation results...")
    results = load_results()
    
    print("\nCreating quantity charts...")
    create_quantity_chart(results)
    
    print("Creating diversity charts...")
    create_diversity_chart(results)
    
    print("Creating stacked charts...")
    create_stacked_bar_chart(results)
    
    print("\nDone!")
