"""
Script para evaluar todas las corridas de requerimientos.
Procesa cada run independientemente y genera archivos JSON separados.
"""
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "model_requirements_comparison"

# Number of runs
NUM_RUNS = 5


def evaluate_single_run(run_number: int):
    """Evalúa una sola corrida."""
    print(f"\n{'='*80}")
    print(f"EVALUANDO RUN {run_number}/{NUM_RUNS}")
    print(f"{'='*80}\n")
    
    # Ejecutar el script de evaluación pasándole el número de run
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate_requirements_quality.py"),
        f"--run={run_number}"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"ERROR: Falló la evaluación del run {run_number}")
        return False
    
    return True


def main():
    print("="*80)
    print("EVALUACIÓN DE TODAS LAS CORRIDAS")
    print("="*80)
    print(f"\nSe evaluarán {NUM_RUNS} corridas independientes")
    print("Cada corrida generará un archivo JSON separado\n")
    
    successful_runs = 0
    
    for run_num in range(1, NUM_RUNS + 1):
        if evaluate_single_run(run_num):
            successful_runs += 1
    
    print(f"\n{'='*80}")
    print(f"PROCESO COMPLETADO: {successful_runs}/{NUM_RUNS} corridas exitosas")
    print(f"{'='*80}")
    
    if successful_runs == NUM_RUNS:
        print("\n✅ Todas las corridas evaluadas exitosamente")
        print(f"\nArchivos generados en: {OUTPUT_BASE}")
        print("  - requirements_quality_evaluation_run1.json")
        print("  - requirements_quality_evaluation_run2.json")
        print("  - ...")
        print(f"  - requirements_quality_evaluation_run{NUM_RUNS}.json")
    else:
        print(f"\n⚠️ Solo {successful_runs} de {NUM_RUNS} corridas fueron exitosas")


if __name__ == "__main__":
    main()
