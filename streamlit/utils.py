"""
Utility functions for the Streamlit Comparator app.
Handles loading configuration, submissions, and run data.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml


def get_project_root() -> Path:
    """Get the project root directory (parent of streamlit folder)."""
    return Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """Load streamlit.yaml configuration file."""
    config_path = Path(__file__).parent / "streamlit.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_submissions(config: Dict[str, Any]) -> List[str]:
    """
    Get list of available submissions from the assignment folder.
    Returns list of submission names (without extension).
    """
    project_root = get_project_root()
    assignment_path = project_root / config['assignment_path']
    
    if not assignment_path.exists():
        return []
    
    submissions = []
    for f in assignment_path.iterdir():
        if f.suffix == '.py' and f.stem != '__init__':
            submissions.append(f.stem)
    
    return sorted(submissions)


def get_submission_data(config: Dict[str, Any], submission_name: str) -> Dict[str, Any]:
    """
    Load submission code and reference correction.
    
    Returns dict with:
        - code: The student's Python code
        - reference_correction: The reference correction (list or text)
        - assignment: The assignment description (consigna)
    """
    project_root = get_project_root()
    assignment_path = project_root / config['assignment_path']
    
    result = {
        'code': '',
        'reference_correction': None,
        'assignment': ''
    }
    
    # Load submission code
    code_path = assignment_path / f"{submission_name}.py"
    if code_path.exists():
        with open(code_path, 'r', encoding='utf-8') as f:
            result['code'] = f.read()
    
    # Load reference correction (try .json first, then .txt)
    json_path = assignment_path / f"{submission_name}.json"
    txt_path = assignment_path / f"{submission_name}.txt"
    
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            result['reference_correction'] = json.load(f)
    elif txt_path.exists():
        with open(txt_path, 'r', encoding='utf-8') as f:
            result['reference_correction'] = f.read()
    
    # Load assignment description
    consigna_path = assignment_path / "consigna.txt"
    if consigna_path.exists():
        with open(consigna_path, 'r', encoding='utf-8') as f:
            result['assignment'] = f.read()
    
    return result


def get_experiment_names(config: Dict[str, Any]) -> List[str]:
    """
    Get list of experiment names by scanning the outputs directory.
    
    Structure: outputs_path/{experiment_name}/{timestamp}/{submission}/
    """
    project_root = get_project_root()
    outputs_path = project_root / config['outputs_path']
    
    if not outputs_path.exists():
        return []
    
    experiments = []
    for entry in outputs_path.iterdir():
        if entry.is_dir() and not entry.name.startswith('.'):
            experiments.append(entry.name)
    
    return sorted(experiments)


def get_available_runs(config: Dict[str, Any], submission_name: str) -> Dict[str, List[str]]:
    """
    Scan outputs for runs that have data for a given submission.
    
    Structure: outputs_path/{experiment_name}/{timestamp}/{submission}/
    
    Returns dict mapping experiment_name -> list of timestamps
    """
    project_root = get_project_root()
    outputs_path = project_root / config['outputs_path']
    
    if not outputs_path.exists():
        return {}
    
    available = {}
    
    # Scan all experiment directories
    for exp_dir in outputs_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        exp_name = exp_dir.name
        timestamps = []
        
        for timestamp_dir in sorted(exp_dir.iterdir(), reverse=True):
            if not timestamp_dir.is_dir():
                continue
            
            # Skip nested directories like "archive"
            # Timestamps follow pattern YYYYMMDDTHHMMSS
            if not (len(timestamp_dir.name) >= 15 and 'T' in timestamp_dir.name):
                continue
            
            # Check if this timestamp has the submission
            submission_dir = timestamp_dir / submission_name
            if submission_dir.exists() and submission_dir.is_dir():
                timestamps.append(timestamp_dir.name)
        
        if timestamps:
            available[exp_name] = timestamps
    
    return available


def get_run_data(
    config: Dict[str, Any], 
    experiment_name: str, 
    timestamp: str, 
    submission_name: str
) -> Optional[Dict[str, Any]]:
    """
    Load run data for a specific experiment/timestamp/submission.
    
    Returns dict with:
        - generated_corrections: List of corrections
        - auxiliary_metrics: Dict of auxiliary metrics
        - evaluation_results: Dict with scores, explanations, overall_score
    """
    project_root = get_project_root()
    run_path = project_root / config['outputs_path'] / experiment_name / timestamp / submission_name
    
    if not run_path.exists():
        return None
    
    result = {
        'generated_corrections': None,
        'auxiliary_metrics': None,
        'evaluation_results': None,
        'path': str(run_path)
    }
    
    # Load generated corrections
    corrections_path = run_path / "generated_corrections.json"
    if corrections_path.exists():
        with open(corrections_path, 'r', encoding='utf-8') as f:
            result['generated_corrections'] = json.load(f)
    
    # Load auxiliary metrics
    aux_path = run_path / "auxiliary_metrics.json"
    if aux_path.exists():
        with open(aux_path, 'r', encoding='utf-8') as f:
            result['auxiliary_metrics'] = json.load(f)
    
    # Load evaluation results
    eval_path = run_path / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, 'r', encoding='utf-8') as f:
            result['evaluation_results'] = json.load(f)
    
    return result


def format_timestamp(timestamp: str) -> str:
    """Format a timestamp string for display (YYYYMMDDTHHMMSS -> YYYY-MM-DD HH:MM:SS)."""
    if len(timestamp) >= 15 and 'T' in timestamp:
        date_part = timestamp[:8]
        time_part = timestamp[9:15]
        return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    return timestamp


def get_prompts_data(
    config: Dict[str, Any], 
    experiment_name: str, 
    timestamp: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Load prompts data for a specific experiment/timestamp.
    Prompts are stored at the timestamp level (not per submission).
    
    Returns list of prompt dicts, each with:
        - requirement: Dict with requirement, function, type
        - jinja_template: The generated prompt template
        - examples: The examples used
        - index: The prompt index
    """
    project_root = get_project_root()
    prompts_path = project_root / config['outputs_path'] / experiment_name / timestamp / "prompts"
    
    if not prompts_path.exists():
        return None
    
    prompts = []
    for prompt_file in sorted(prompts_path.glob("*.json")):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                prompt_data['filename'] = prompt_file.name
                prompts.append(prompt_data)
        except Exception as e:
            continue
    
    return prompts if prompts else None

