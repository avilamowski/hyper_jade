"""
Script para generar un reporte legible de la evaluación de requerimientos.
Muestra cada requerimiento generado y su relación con los criterios del docente.
"""
import json
from pathlib import Path
from typing import Dict, List

# Paths
OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"


def load_results():
    """Carga los resultados de la evaluación."""
    results_file = OUTPUT_BASE / "requirements_quality_evaluation.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def generate_text_report(results: dict):
    """Genera un reporte de texto legible."""
    report_path = OUTPUT_BASE / "requirements_evaluation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE EVALUACIÓN DE REQUERIMIENTOS\n")
        f.write("="*80 + "\n\n")
        
        # Group by dataset and student
        by_dataset = {}
        for detail in results["detailed"]:
            dataset = detail["dataset"]
            student = detail["student"]
            key = (dataset, student)
            
            if key not in by_dataset:
                by_dataset[key] = []
            by_dataset[key].append(detail)
        
        # Generate report for each dataset/student/model
        for (dataset, student), details in sorted(by_dataset.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"EJERCICIO: {dataset}\n")
            f.write(f"ALUMNO: {student}\n")
            f.write(f"{'='*80}\n\n")
            
            # Group by model
            by_model = {}
            for detail in details:
                model = detail["model"]
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(detail)
            
            # Show each model's requirements
            for model in sorted(by_model.keys()):
                model_details = sorted(by_model[model], key=lambda x: x["requirement_id"])
                
                f.write(f"\n{'-'*80}\n")
                f.write(f"MODELO: {model}\n")
                f.write(f"{'-'*80}\n\n")
                
                for detail in model_details:
                    req_id = detail["requirement_id"]
                    req_type = detail["requirement_type"]
                    req_func = detail["requirement_function"]
                    req_desc = detail["requirement_description"]
                    related = detail["related_to_teacher"]
                    teacher_num = detail.get("related_teacher_req_number")
                    teacher_text = detail.get("related_teacher_req_text")
                    reasoning = detail["reasoning"]
                    
                    f.write(f"Requerimiento {req_id}:\n")
                    f.write(f"  Tipo: {req_type}\n")
                    f.write(f"  Función: {req_func}\n")
                    f.write(f"  Descripción: {req_desc}\n")
                    f.write(f"\n  Evaluación:\n")
                    f.write(f"    ✓ Relacionado con docente: {'SÍ' if related else 'NO'}\n")
                    
                    if related and teacher_num and teacher_text:
                        f.write(f"    → Requerimiento del docente #{teacher_num}:\n")
                        f.write(f"      \"{teacher_text}\"\n")
                    
                    f.write(f"    Razonamiento: {reasoning}\n")
                    f.write(f"\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")
    
    print(f"Reporte generado: {report_path}")
    return report_path


def generate_html_report(results: dict):
    """Genera un reporte HTML interactivo."""
    report_path = OUTPUT_BASE / "requirements_evaluation_report.html"
    
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluación de Requerimientos</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            background-color: #ecf0f1;
            padding: 10px;
            border-left: 5px solid #3498db;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }
        .requirement {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .requirement-header {
            font-weight: bold;
            color: #2980b9;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .requirement-content {
            margin-left: 20px;
        }
        .label {
            display: inline-block;
            font-weight: bold;
            color: #555;
            min-width: 100px;
        }
        .evaluation {
            background-color: #f8f9fa;
            border-left: 4px solid #95a5a6;
            padding: 10px;
            margin-top: 10px;
        }
        .evaluation.related {
            border-left-color: #27ae60;
        }
        .evaluation.not-related {
            border-left-color: #e74c3c;
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }
        .badge-yes {
            background-color: #d4edda;
            color: #155724;
        }
        .badge-no {
            background-color: #f8d7da;
            color: #721c24;
        }
        .teacher-req {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 10px;
            margin-top: 8px;
            font-style: italic;
        }
        .teacher-req-header {
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }
        .reasoning {
            color: #666;
            font-size: 0.95em;
            margin-top: 8px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .summary {
            background-color: #e8f4f8;
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 30px;
        }
        .summary table {
            width: 100%;
            border-collapse: collapse;
        }
        .summary th, .summary td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .summary th {
            background-color: #3498db;
            color: white;
        }
    </style>
</head>
<body>
    <h1>📊 Reporte de Evaluación de Requerimientos</h1>
    
    <div class="summary">
        <h3>Resumen por Modelo</h3>
        <table>
            <thead>
                <tr>
                    <th>Modelo</th>
                    <th>Total</th>
                    <th>Relacionado con Docente</th>
                    <th>Criterios Únicos Cubiertos</th>
                    <th>Redundancia</th>
                    <th>% Calidad</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add summary table
    for model, stats in results["by_model"].items():
        total = stats["total"]
        related = stats["related_to_teacher"]
        unique = stats["unique_teacher_requirements"]
        redundancy = stats.get("redundancy_ratio", 0)
        quality_pct = (related / total * 100) if total > 0 else 0
        
        html += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{total}</td>
                    <td>{related} ({related/total*100:.0f}%)</td>
                    <td>{unique}</td>
                    <td>{redundancy:.2f}x</td>
                    <td>{quality_pct:.1f}%</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
"""
    
    # Group by dataset and student
    by_dataset = {}
    for detail in results["detailed"]:
        dataset = detail["dataset"]
        student = detail["student"]
        key = (dataset, student)
        
        if key not in by_dataset:
            by_dataset[key] = []
        by_dataset[key].append(detail)
    
    # Generate report for each dataset/student/model
    for (dataset, student), details in sorted(by_dataset.items()):
        html += f"""
    <h2>📝 {dataset} - {student}</h2>
"""
        
        # Group by model
        by_model = {}
        for detail in details:
            model = detail["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(detail)
        
        # Show each model's requirements
        for model in sorted(by_model.keys()):
            model_details = sorted(by_model[model], key=lambda x: x["requirement_id"])
            
            html += f"""
    <h3>🤖 {model}</h3>
"""
            
            for detail in model_details:
                req_id = detail["requirement_id"]
                req_type = detail["requirement_type"]
                req_func = detail["requirement_function"]
                req_desc = detail["requirement_description"]
                related = detail["related_to_teacher"]
                teacher_num = detail.get("related_teacher_req_number")
                teacher_text = detail.get("related_teacher_req_text")
                reasoning = detail["reasoning"]
                
                eval_class = "related" if related else "not-related"
                related_badge = "badge-yes" if related else "badge-no"
                
                html += f"""
    <div class="requirement">
        <div class="requirement-header">Requerimiento {req_id}</div>
        <div class="requirement-content">
            <div><span class="label">Tipo:</span> {req_type}</div>
            <div><span class="label">Función:</span> {req_func}</div>
            <div><span class="label">Descripción:</span> {req_desc}</div>
            
            <div class="evaluation {eval_class}">
                <div>
                    <strong>Relacionado con docente:</strong> 
                    <span class="badge {related_badge}">{'SÍ' if related else 'NO'}</span>
                </div>
"""
                
                if related and teacher_num and teacher_text:
                    html += f"""
                <div class="teacher-req">
                    <div class="teacher-req-header">→ Requerimiento del docente #{teacher_num}:</div>
                    <div>"{teacher_text}"</div>
                </div>
"""
                
                html += f"""
                <div class="reasoning"><strong>Razonamiento:</strong> {reasoning}</div>
            </div>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Reporte HTML generado: {report_path}")
    return report_path


if __name__ == "__main__":
    print("Cargando resultados de evaluación...")
    results = load_results()
    
    print("\nGenerando reportes...")
    text_report = generate_text_report(results)
    html_report = generate_html_report(results)
    
    print(f"\n✓ Reporte de texto: {text_report}")
    print(f"✓ Reporte HTML: {html_report}")
    print("\nPuedes abrir el reporte HTML en tu navegador para una mejor visualización.")
