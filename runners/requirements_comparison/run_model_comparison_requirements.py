"""
Script para comparar diferentes modelos en la generación de requerimientos.
Genera requerimientos múltiples veces por ejercicio (no por alumno) con múltiples modelos para análisis estadístico.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent
from src.config import load_config, load_langsmith_config


# Número de corridas independientes para cada modelo
NUM_RUNS = 5

# Modelos a comparar (configuración: provider, model_name)
# Solo modelos estables y con cuotas suficientes
MODELS = [
    {"provider": "openai", "model_name": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"provider": "google-genai", "model_name": "gemini-2.0-flash", "name": "gemini-2.0-flash"},
    {"provider": "google-genai", "model_name": "gemini-3-pro-preview", "name": "gemini-3-pro"},
]

# Datasets a usar
DATASETS = [
    "ej1-2025-s2-p2-ej1",
    "ej1-2025-s2-p2-ej2",
]


def load_consigna(dataset_path: Path):
    """Carga la consigna del ejercicio."""
    consigna_file = dataset_path / "consigna.txt"
    with open(consigna_file, 'r', encoding='utf-8') as f:
        return f.read()


def generate_requirements_with_model(model_config: dict, consigna: str, base_config: dict):
    """Genera requerimientos usando un modelo específico."""
    model_name = model_config["name"]
    print(f"\n  Generando con {model_name}...")
    
    # Crear configuración temporal para este modelo
    config = base_config.copy()
    config["agents"]["requirement_generator"]["provider"] = model_config["provider"]
    config["agents"]["requirement_generator"]["model_name"] = model_config["model_name"]
    
    try:
        agent = RequirementGeneratorAgent(config)
        requirements = agent.generate_requirements(consigna)
        
        # Convertir a formato legible
        requirements_text = ""
        for i, req in enumerate(requirements, 1):
            requirements_text += f"\nRequirement {i}:\n"
            requirements_text += f"  Type: {req['type'].value if hasattr(req['type'], 'value') else req['type']}\n"
            if req.get('function'):
                requirements_text += f"  Function: {req['function']}\n"
            requirements_text += f"  Description: {req['requirement']}\n"
        
        return requirements_text
    except Exception as e:
        print(f"    Error con {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(output_dir: Path, dataset_name: str, model_name: str, requirements: str, run_number: int):
    """Guarda los requerimientos generados (una vez por ejercicio por corrida)."""
    # Crear directorio por dataset
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar requirements por modelo y run directamente en el directorio del dataset
    model_filename = model_name.replace("/", "_").replace(" ", "_")
    output_file = dataset_dir / f"requirements_{model_filename}_run{run_number}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Run: {run_number}\n")
        f.write("="*80 + "\n\n")
        f.write(requirements)
    
    print(f"    Guardado en: {output_file}")


def copy_teacher_requirements(output_dir: Path, dataset_path: Path, dataset_name: str):
    """Copia los teacher requirements para referencia."""
    teacher_req_file = dataset_path / "teacher_requirements.txt"
    if not teacher_req_file.exists():
        return
    
    with open(teacher_req_file, 'r', encoding='utf-8') as f:
        teacher_reqs = f.read()
    
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = dataset_dir / "teacher_requirements.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(teacher_reqs)
    
    print(f"  Teacher requirements copiados a: {output_file}")


def main():
    base_path = Path(__file__).parent.parent / "ejemplos"
    output_dir = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"
    
    # Load base configuration
    config_path = Path(__file__).parent.parent / "src" / "config" / "assignment_config.yaml"
    base_config = load_config(str(config_path))
    if not base_config:
        print("Error: No se pudo cargar la configuración")
        sys.exit(1)
    
    # Load LangSmith config (optional)
    try:
        load_langsmith_config()
    except:
        pass
    
    print("="*80)
    print("COMPARACIÓN DE MODELOS PARA GENERACIÓN DE REQUERIMIENTOS")
    print("="*80)
    print(f"\nNúmero de corridas independientes: {NUM_RUNS}")
    print(f"\nModelos a comparar:")
    for model in MODELS:
        print(f"  - {model['name']} ({model['provider']}: {model['model_name']})")
    print(f"\nDatasets: {DATASETS}")
    print("\nNOTA: Los requerimientos se generan múltiples veces por ejercicio para análisis estadístico")
    
    # Procesar cada dataset
    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        dataset_path = base_path / dataset_name
        
        # Copiar teacher requirements
        copy_teacher_requirements(output_dir, dataset_path, dataset_name)
        
        # Cargar consigna
        consigna = load_consigna(dataset_path)
        
        # Generar con cada modelo (múltiples corridas)
        for model_config in MODELS:
            print(f"\n  Modelo: {model_config['name']}")
            for run in range(1, NUM_RUNS + 1):
                print(f"    Corrida {run}/{NUM_RUNS}...", end=" ")
                requirements = generate_requirements_with_model(
                    model_config=model_config,
                    consigna=consigna,
                    base_config=base_config
                )
                
                if requirements:
                    save_results(output_dir, dataset_name, model_config["name"], requirements, run)
                    print("✓")
                else:
                    print("✗")
    
    print(f"\n{'='*80}")
    print("PROCESO COMPLETADO")
    print(f"{'='*80}")
    print(f"\nResultados guardados en: {output_dir}")
    print("\nEstructura generada:")
    print("  outputs/model_requirements_comparison/")
    print("    ├── ej1-2025-s2-p2-ej1/")
    print("    │   ├── teacher_requirements.txt")
    print(f"    │   ├── requirements_gpt-4o-mini_run1.txt ... run{NUM_RUNS}.txt")
    print(f"    │   ├── requirements_gemini-2.0-flash_run1.txt ... run{NUM_RUNS}.txt")
    print(f"    │   └── requirements_gemini-3-pro_run1.txt ... run{NUM_RUNS}.txt")
    print("    └── ej1-2025-s2-p2-ej2/")
    print("        ├── teacher_requirements.txt")
    print(f"        ├── requirements_gpt-4o-mini_run1.txt ... run{NUM_RUNS}.txt")
    print(f"        ├── requirements_gemini-2.0-flash_run1.txt ... run{NUM_RUNS}.txt")
    print(f"        └── requirements_gemini-3-pro_run1.txt ... run{NUM_RUNS}.txt")
    print(f"\nTotal de archivos generados: {len(MODELS) * len(DATASETS) * NUM_RUNS} + {len(DATASETS)} teacher requirements")
    print("\nAhora puedes revisar manualmente los requerimientos y compararlos con los del docente.")
    print("Para análisis estadístico, procesa todos los archivos run1...runN de cada modelo.")


if __name__ == "__main__":
    main()
