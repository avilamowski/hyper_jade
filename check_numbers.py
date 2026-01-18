import json
from pathlib import Path

results_file = Path('outputs/model_requirements_comparison/requirements_quality_evaluation.json')
with open(results_file, 'r') as f:
    data = json.load(f)

# Por ejercicio
print('=== POR EJERCICIO ===')
for dataset in ['ej1-2025-s2-p2-ej1', 'ej1-2025-s2-p2-ej2']:
    print(f'\n{dataset}:')
    models_data = {}
    for detail in data['detailed']:
        if detail['dataset'] == dataset:
            model = detail['model']
            if model not in models_data:
                models_data[model] = {'total': 0, 'unicos': set(), 'relacionados': 0, 'no_relacionados': 0, 'duplicados': 0}
            
            models_data[model]['total'] += 1
            if detail.get('related_to_teacher'):
                models_data[model]['relacionados'] += 1
                req_num = detail.get('related_teacher_req_number')
                if req_num not in models_data[model]['unicos']:
                    models_data[model]['unicos'].add(req_num)
                else:
                    models_data[model]['duplicados'] += 1
            else:
                models_data[model]['no_relacionados'] += 1
    
    for model, stats in sorted(models_data.items()):
        unicos = len(stats['unicos'])
        print(f'  {model}:')
        print(f'    Total: {stats["total"]}')
        print(f'    Únicos: {unicos}')
        print(f'    Duplicados: {stats["duplicados"]}')
        print(f'    No relacionados: {stats["no_relacionados"]}')
        print(f'    Precisión (únicos/total): {unicos}/{stats["total"]} = {unicos/stats["total"]*100:.0f}%')

print('\n=== TOTALES (ambos ejercicios) ===')
totales = {}
for detail in data['detailed']:
    model = detail['model']
    if model not in totales:
        totales[model] = {'total': 0, 'unicos': set(), 'relacionados': 0, 'no_relacionados': 0, 'duplicados': 0}
    
    totales[model]['total'] += 1
    if detail.get('related_to_teacher'):
        totales[model]['relacionados'] += 1
        req_num = detail.get('related_teacher_req_number')
        if req_num not in totales[model]['unicos']:
            totales[model]['unicos'].add(req_num)
        else:
            totales[model]['duplicados'] += 1
    else:
        totales[model]['no_relacionados'] += 1

for model, stats in sorted(totales.items()):
    unicos = len(stats['unicos'])
    print(f'\n{model}:')
    print(f'  Total: {stats["total"]}')
    print(f'  Únicos: {unicos} ({unicos/stats["total"]*100:.0f}%)')
    print(f'  Duplicados: {stats["duplicados"]}')
    print(f'  No relacionados: {stats["no_relacionados"]}')
    print(f'  Relacionados (únicos+duplicados): {unicos + stats["duplicados"]} ({(unicos + stats["duplicados"])/stats["total"]*100:.0f}%)')
    print(f'  Ratio redundancia: {(unicos + stats["duplicados"])/unicos if unicos > 0 else 0:.2f}')
