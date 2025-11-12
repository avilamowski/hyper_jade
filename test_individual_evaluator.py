#!/usr/bin/env python3
"""
Test script for individual evaluator implementation.

Tests the auxiliary metric (MATCH) and evaluation metric (completeness).
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from src.evaluators.supervised_evaluator_aux import AuxiliaryMetricsEvaluator
from src.evaluators.supervised_evaluator_indivudal import IndividualMetricsEvaluator
from src.models import Submission


def test_match_and_completeness():
    """Test MATCH auxiliary metric and completeness evaluation metric."""
    
    # Sample data
    student_code = """
def calcular_promedio(numeros):
    suma = 0
    for num in numeros:
        suma += num
    return suma / len(numeros)

resultado = calcular_promedio([10, 20, 30])
print(resultado)
"""
    
    reference_correction = """
El código tiene los siguientes problemas:
1. No maneja el caso cuando la lista está vacía (división por cero)
2. No valida que los elementos sean números
3. Falta documentación de la función
"""
    
    generated_correction = """
Problemas encontrados:
1. La función no maneja el caso de lista vacía, lo que causaría división por cero
2. No hay validación de tipos para los elementos de la lista
"""
    
    assignment = "Implementar una función que calcule el promedio de una lista de números"
    requirements = """
- La función debe manejar listas vacías
- Debe validar los tipos de datos
- Debe incluir documentación
"""
    
    submission: Submission = {"code": student_code}
    
    print("=" * 80)
    print("Testing Individual Evaluator Implementation")
    print("=" * 80)
    
    # Step 1: Compute auxiliary metrics
    print("\n[STEP 1] Computing auxiliary metrics (MATCH and MISSING)...")
    print("-" * 80)
    
    aux_evaluator = AuxiliaryMetricsEvaluator()
    
    aux_metrics = aux_evaluator.compute_all_auxiliary_metrics(
        generated_text=generated_correction,
        reference_text=reference_correction,
        submission=submission,
        assignment=assignment,
        requirements=requirements,
        metrics_to_compute=["match", "missing"]  # Compute both MATCH and MISSING
    )
    
    print("\nMATCH auxiliary metric result:")
    print(aux_metrics.get("match", "NOT COMPUTED"))
    
    print("\n" + "-" * 80)
    print("\nMISSING auxiliary metric result:")
    print(aux_metrics.get("missing", "NOT COMPUTED"))
    
    # Step 2: Evaluate completeness metric
    print("\n" + "=" * 80)
    print("[STEP 2] Evaluating COMPLETENESS metric...")
    print("-" * 80)
    
    eval_evaluator = IndividualMetricsEvaluator()
    
    result = eval_evaluator.evaluate_metric(
        metric_name="completeness",
        aux_metrics=aux_metrics,
        generated_text=generated_correction,
        reference_text=reference_correction,
        submission=submission,
        assignment=assignment,
        requirements=requirements
    )
    
    print(f"\nCompleteness Score: {result['score']}/5")
    print(f"Explanation: {result['explanation']}")
    print(f"Timing: {result['timing']:.2f}s")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_match_and_completeness()
