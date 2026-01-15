import json
from pathlib import Path

OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"

with open(OUTPUT_BASE / "requirements_quality_evaluation.json", 'r') as f:
    data = json.load(f)

print('=== EJEMPLOS DE REQUERIMIENTOS RELACIONADOS ===\n')
count = 0
for d in data['detailed']:
    if d.get('related_to_teacher') and count < 4:
        print(f'Modelo: {d["model"]}')
        print(f'Dataset: {d["dataset"]}')
        print(f'Requerimiento generado: {d["requirement_description"]}')
        print(f'Criterio docente (#): {d.get("related_teacher_req_number")}')
        teacher_text = d.get("related_teacher_req_text", "N/A")
        if teacher_text:
            print(f'Criterio docente: {teacher_text[:300]}')
        print('-' * 80)
        count += 1

print('\n=== EJEMPLOS DE REQUERIMIENTOS NO RELACIONADOS ===\n')
count = 0
for d in data['detailed']:
    if not d.get('related_to_teacher') and count < 4:
        print(f'Modelo: {d["model"]}')
        print(f'Dataset: {d["dataset"]}')
        print(f'Requerimiento generado: {d["requirement_description"]}')
        reason = d.get("reasoning", "N/A")
        if reason:
            print(f'Razon: {reason[:350]}')
        print('-' * 80)
        count += 1

# Cargar criterios del docente
print('\n=== CRITERIOS DEL DOCENTE (ej1) ===\n')
teacher_path = OUTPUT_BASE / "ej1-2025-s2-p2-ej1" / "teacher_requirements.txt"
if teacher_path.exists():
    with open(teacher_path, 'r') as f:
        print(f.read()[:1500])

print('\n=== CRITERIOS DEL DOCENTE (ej2) ===\n')
teacher_path = OUTPUT_BASE / "ej1-2025-s2-p2-ej2" / "teacher_requirements.txt"
if teacher_path.exists():
    with open(teacher_path, 'r') as f:
        print(f.read()[:1500])
