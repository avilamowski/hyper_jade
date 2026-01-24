"""
Script para evaluar la calidad de los requerimientos generados por cada modelo.
Usa Gemini 2.0 Flash como evaluador "humano" para determinar:
1. Si cada requerimiento tiene sentido
2. Si está relacionado con los criterios del docente

Puede evaluar una corrida específica con --run=N
"""
import sys
import os
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Configurar Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelos evaluados
MODELS = ["gpt-4o-mini", "gemini-2.0-flash", "gemini-3-pro"]

# Datasets
DATASETS = ["ej1-2025-s2-p2-ej1", "ej1-2025-s2-p2-ej2"]

# Output path
OUTPUT_BASE = Path(__file__).parent.parent.parent / "outputs" / "model_requirements_comparison"

# Run number (global variable set by command line)
RUN_NUMBER = None


def parse_requirements_file(filepath: Path, run_number: int) -> List[Dict[str, str]]:
    """Parsea un archivo de requerimientos y extrae cada uno."""
    # Construct filename with run number
    filename = filepath.stem + f"_run{run_number}" + filepath.suffix
    filepath_with_run = filepath.parent / filename
    
    content = filepath_with_run.read_text()
    requirements = []
    
    # Split by "Requirement N:"
    req_blocks = re.split(r'(?=Requirement \d+:)', content)
    
    for block in req_blocks:
        block = block.strip()
        if not block or not block.startswith('Requirement'):
            continue
        
        # Extract requirement number
        req_num_match = re.search(r'Requirement (\d+):', block)
        if not req_num_match:
            continue
        req_id = int(req_num_match.group(1))
        
        # Extract type
        type_match = re.search(r'Type:\s*(\w+)', block)
        req_type = type_match.group(1).strip() if type_match else "unknown"
        
        # Extract function (optional)
        function_match = re.search(r'Function:\s*([^\n]+)', block)
        function = function_match.group(1).strip() if function_match else ""
        
        # Extract description
        desc_match = re.search(r'Description:\s*(.+?)(?=\n\s*\n|$)', block, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        requirements.append({
            "id": req_id,
            "type": req_type,
            "function": function,
            "description": description
        })
    
    return requirements


def parse_requirements_file_direct(filepath: Path) -> List[Dict[str, str]]:
    """Parsea un archivo de requerimientos directamente (ya tiene el nombre completo)."""
    content = filepath.read_text()
    requirements = []
    
    # Split by "Requirement N:"
    req_blocks = re.split(r'(?=Requirement \d+:)', content)
    
    for block in req_blocks:
        block = block.strip()
        if not block or not block.startswith('Requirement'):
            continue
        
        # Extract requirement number
        req_num_match = re.search(r'Requirement (\d+):', block)
        if not req_num_match:
            continue
        req_id = int(req_num_match.group(1))
        
        # Extract type
        type_match = re.search(r'Type:\s*(\w+)', block)
        req_type = type_match.group(1).strip() if type_match else "unknown"
        
        # Extract function (optional)
        function_match = re.search(r'Function:\s*([^\n]+)', block)
        function = function_match.group(1).strip() if function_match else ""
        
        # Extract description
        desc_match = re.search(r'Description:\s*(.+?)(?=\n\s*\n|$)', block, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        
        requirements.append({
            "id": req_id,
            "type": req_type,
            "function": function,
            "description": description
        })
    
    return requirements


def parse_teacher_requirements(filepath: Path) -> List[str]:
    """Parsea los requerimientos del docente."""
    content = filepath.read_text()
    requirements = []
    for line in content.strip().split('\n'):
        if line.startswith('- Requirement'):
            # Extract the requirement text after the colon
            parts = line.split(':', 1)
            if len(parts) > 1:
                requirements.append(parts[1].strip())
    return requirements


def evaluate_requirement(requirement: Dict[str, str], teacher_requirements: List[str], exercise_name: str) -> Dict[str, Any]:
    """
    Evalúa un requerimiento individual usando Gemini 2.0 Flash.
    Retorna:
    - makes_sense: bool - si el requerimiento tiene sentido
    - related_to_teacher: bool - si está relacionado con algún criterio docente
    - related_teacher_req_number: int - número del requerimiento del docente (1-indexed)
    - related_teacher_req_text: str - texto del requerimiento del docente
    - reasoning: str - explicación del evaluador
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    teacher_list = "\n".join([f"  {i+1}. {req}" for i, req in enumerate(teacher_requirements)])
    
    prompt = f"""You are an expert evaluator assessing the quality of automatically generated requirements for a programming exercise.

EXERCISE: {exercise_name}

TEACHER'S GRADING CRITERIA (reference):
{teacher_list}

GENERATED REQUIREMENT TO EVALUATE:
- Type: {requirement['type']}
- Function: {requirement['function']}  
- Description: {requirement['description']}

Please evaluate: Is this requirement DIRECTLY and SPECIFICALLY related to one of the teacher's grading criteria?
   
   BE VERY STRICT. Only mark as related if:
   - The requirement addresses the EXACT SAME specific aspect mentioned in the teacher's criterion
   - It checks for the SAME calculation, output format, condition, or behavior
   - It validates the SAME functional requirement
   
   DO NOT mark as related if:
   - The requirement is generic or vague (e.g., "solve assignment", "implement function")
   - It's a general good practice (e.g., modularity, naming, documentation) not specifically mentioned by teacher
   - It's a stylistic or error detection that doesn't match a specific teacher criterion
   - The connection is indirect or tangential
   
   Examples of VALID relationships:
   - Teacher: "Does not calculate average correctly" → Generated: "Calculate average of grades"
   - Teacher: "Does not round to 2 decimals" → Generated: "Round result to two decimal places"
   - Teacher: "Does not print separator every 3 students" → Generated: "Print separator line every three students"
   
   Examples of INVALID relationships (DO NOT mark as related):
   - Teacher: "Does not solve assignment requirements" → Generated: "Maintain modularity" (too vague)
   - Teacher: "Does not calculate average" → Generated: "Handle empty lists" (different aspect)
   - Teacher: "Output format wrong" → Generated: "Use meaningful variable names" (unrelated)
   
   If related, identify which teacher requirement number (1-{len(teacher_requirements)})

Respond in this exact JSON format:
{{
    "related_to_teacher": true/false,
    "related_teacher_req_number": <number 1-{len(teacher_requirements)} or null if not related>,
    "reasoning": "Brief explanation of your evaluation"
}}

Respond ONLY with the JSON, no other text."""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Try to parse JSON from response
        # Sometimes the model wraps it in ```json ... ```
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        
        result = json.loads(text)
        
        # Get teacher req text if number provided
        teacher_req_number = result.get("related_teacher_req_number")
        teacher_req_text = None
        if teacher_req_number and 1 <= teacher_req_number <= len(teacher_requirements):
            teacher_req_text = teacher_requirements[teacher_req_number - 1]
        
        return {
            "related_to_teacher": result.get("related_to_teacher", False),
            "related_teacher_req_number": teacher_req_number,
            "related_teacher_req_text": teacher_req_text,
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"  Error evaluating requirement: {e}")
        return {
            "related_to_teacher": None,
            "related_teacher_req_number": None,
            "related_teacher_req_text": None,
            "reasoning": f"Error: {str(e)}"
        }


def normalize_requirement_text(text: str) -> str:
    """Normaliza el texto de un requerimiento para comparación."""
    # Convertir a minúsculas y eliminar espacios extras
    text = text.lower().strip()
    # Eliminar puntuación al final
    text = text.rstrip('.,;:!?')
    return text


def are_requirements_similar(req1_desc: str, req2_desc: str, threshold: float = 0.85) -> bool:
    """
    Determina si dos requerimientos son similares usando similitud de texto simple.
    Retorna True si son esencialmente el mismo requerimiento.
    """
    norm1 = normalize_requirement_text(req1_desc)
    norm2 = normalize_requirement_text(req2_desc)
    
    # Si son exactamente iguales después de normalizar
    if norm1 == norm2:
        return True
    
    # Calcular similitud simple basada en palabras en común
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return False
    
    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    
    return similarity >= threshold


def evaluate_all_requirements():
    """Evalúa todos los requerimientos generados por todos los modelos."""
    global RUN_NUMBER
    
    all_results = {
        "by_model": {},
        "by_dataset": {},
        "detailed": []
    }
    
    for model in MODELS:
        all_results["by_model"][model] = {
            "total": 0,
            "related_to_teacher": 0,
            "unique_teacher_requirements": set(),  # Para trackear criterios únicos del docente
            "teacher_req_counts": {},  # Para contar cuántas veces se usa cada req del docente
            "unique_requirements": [],  # Lista de descripciones únicas de requerimientos
            "unique_count": 0  # Contador de requerimientos únicos
        }
    
    for dataset in DATASETS:
        dataset_path = OUTPUT_BASE / dataset
        teacher_file = dataset_path / "teacher_requirements.txt"
        
        if not teacher_file.exists():
            print(f"Warning: Teacher requirements not found for {dataset}")
            continue
            
        teacher_requirements = parse_teacher_requirements(teacher_file)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Teacher requirements: {len(teacher_requirements)}")
        
        # Procesar cada modelo para este dataset
        for model in MODELS:
            # Construct filename with run number
            req_file = dataset_path / f"requirements_{model}_run{RUN_NUMBER}.txt"
            
            if not req_file.exists():
                print(f"  {model}: File not found at {req_file}")
                continue
            
            requirements = parse_requirements_file_direct(req_file)
            print(f"\n  {model}: {len(requirements)} requirements")
            
            # Lista de requerimientos únicos PARA ESTE DATASET
            dataset_unique_reqs = []
            
            for req in requirements:
                print(f"    Evaluating requirement {req['id']}...", end=" ", flush=True)
                
                eval_result = evaluate_requirement(req, teacher_requirements, dataset)
                
                # Update counters
                all_results["by_model"][model]["total"] += 1
                
                # Determinar si este requerimiento es único DENTRO DE ESTE DATASET
                req_desc = req["description"]
                is_unique = True
                
                for unique_desc in dataset_unique_reqs:
                    if are_requirements_similar(req_desc, unique_desc):
                        is_unique = False
                        break
                
                if is_unique:
                    dataset_unique_reqs.append(req_desc)
                    all_results["by_model"][model]["unique_count"] += 1
                
                is_duplicated = False
                if eval_result["related_to_teacher"]:
                    all_results["by_model"][model]["related_to_teacher"] += 1
                    # Track unique teacher requirements
                    teacher_req_key = (dataset, eval_result["related_teacher_req_number"])
                    
                    # Check if this teacher requirement was already seen (= duplicated)
                    if teacher_req_key in all_results["by_model"][model]["unique_teacher_requirements"]:
                        is_duplicated = True
                    else:
                        all_results["by_model"][model]["unique_teacher_requirements"].add(teacher_req_key)
                    
                    # Count occurrences
                    if teacher_req_key not in all_results["by_model"][model]["teacher_req_counts"]:
                        all_results["by_model"][model]["teacher_req_counts"][teacher_req_key] = 0
                    all_results["by_model"][model]["teacher_req_counts"][teacher_req_key] += 1
                
                # Store detailed result
                all_results["detailed"].append({
                    "dataset": dataset,
                    "model": model,
                    "requirement_id": req["id"],
                    "requirement_type": req["type"],
                    "requirement_function": req["function"],
                    "requirement_description": req["description"],
                    "related_to_teacher": eval_result["related_to_teacher"],
                    "is_duplicated": is_duplicated,
                    "is_unique": is_unique,  # Nueva métrica
                    "related_teacher_req_number": eval_result["related_teacher_req_number"],
                    "related_teacher_req_text": eval_result["related_teacher_req_text"],
                    "reasoning": eval_result["reasoning"]
                })
                
                status = "✓" if eval_result["related_to_teacher"] else "○"
                print(status)
    
    return all_results


def save_results(results: Dict[str, Any], run_number: int):
    """Guarda los resultados en un archivo JSON."""
    # Convert sets to counts and calculate metrics
    for model in results["by_model"]:
        unique_teacher_count = len(results["by_model"][model]["unique_teacher_requirements"])
        results["by_model"][model]["unique_teacher_requirements"] = unique_teacher_count
        
        # Get unique requirements count
        unique_req_count = results["by_model"][model]["unique_count"]
        
        # Remove non-serializable fields
        if "teacher_req_counts" in results["by_model"][model]:
            del results["by_model"][model]["teacher_req_counts"]
        if "unique_requirements" in results["by_model"][model]:
            del results["by_model"][model]["unique_requirements"]
        
        # Calculate redundancy ratio: related requirements / unique teacher criteria
        related = results["by_model"][model]["related_to_teacher"]
        if unique_teacher_count > 0:
            results["by_model"][model]["redundancy_ratio"] = related / unique_teacher_count
        else:
            results["by_model"][model]["redundancy_ratio"] = 0
        
        # Calculate NEW diversity ratio: unique requirements / total requirements
        # Esta es la métrica que mide: de los requerimientos generados, cuántos son únicos
        total = results["by_model"][model]["total"]
        if total > 0:
            results["by_model"][model]["diversity_ratio"] = unique_req_count / total
        else:
            results["by_model"][model]["diversity_ratio"] = 0
    
    output_file = OUTPUT_BASE / f"requirements_quality_evaluation_run{run_number}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def print_summary(results: Dict[str, Any]):
    """Imprime un resumen de los resultados."""
    print("\n" + "="*60)
    print("SUMMARY BY MODEL")
    print("="*60)
    
    for model, stats in results["by_model"].items():
        total = stats["total"]
        if total == 0:
            print(f"\n{model}: No requirements found")
            continue
            
        related_pct = (stats["related_to_teacher"] / total) * 100
        unique_teacher = stats["unique_teacher_requirements"]
        unique_reqs = stats["unique_count"]
        redundancy = stats["redundancy_ratio"]
        diversity = stats.get("diversity_ratio", 0)
        
        print(f"\n{model}:")
        print(f"  Total requirements: {total}")
        print(f"  Related to teacher: {stats['related_to_teacher']}/{total} ({related_pct:.1f}%)")
        print(f"  Unique teacher criteria covered: {unique_teacher}")
        print(f"  Redundancy ratio: {redundancy:.2f} (avg requirements per criterion)")
        print(f"  Unique requirements: {unique_reqs}/{total}")
        print(f"  Diversity ratio: {diversity:.2%} (unique requirements / total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate requirements quality with Gemini 2.0 Flash")
    parser.add_argument("--run", type=int, required=True, help="Run number (1-5)")
    args = parser.parse_args()
    
    RUN_NUMBER = args.run
    
    print(f"Evaluating requirements quality for RUN {RUN_NUMBER} with Gemini 2.0 Flash...")
    print("This may take a while as each requirement is evaluated individually.\n")
    
    results = evaluate_all_requirements()
    save_results(results, RUN_NUMBER)
    print_summary(results)

