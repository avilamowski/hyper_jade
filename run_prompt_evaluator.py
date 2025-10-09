#!/usr/bin/env python3
"""
Prompt Evaluator Runner - Standalone

Runs the prompt evaluator over generated Jinja2 templates and writes
metrics/results to a JSON file. Accepts the same arguments as
`run_prompt_generator.py` plus `--metrics-output` to specify where to
save evaluation metrics (default: <output-dir>/prompt_evaluator_metrics.json).
"""

import sys
import os
import json
import argparse
import time
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluators.prompt_evaluator import PromptEvaluator, evaluate_prompt
from src.config import get_agent_config, load_config, load_langsmith_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_requirement_from_json(json_file_path: str) -> str:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # support simple schema used in run_prompt_generator
        if isinstance(data, dict) and 'requirement' in data:
            return data['requirement']
        # fallback: if file contains raw string
        return str(data)
    except Exception as e:
        raise ValueError(f"Error loading requirement from {json_file_path}: {e}")


def find_prompt_files(output_dir: Path) -> list[Path]:
    # look for prompt_*.jinja files (sorted)
    if not output_dir.exists():
        return []
    files = sorted(output_dir.glob('prompt_*.jinja'))
    return files


def main():
    parser = argparse.ArgumentParser(description="Prompt Evaluator Runner")
    parser.add_argument("--requirement", "-r", nargs="+", required=True, help="Path(s) to requirement file(s) (.json)")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output-dir", required=True, help="Directory where generated templates are located")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--metrics-output", "-m", help="Path to write metrics JSON (default: <output-dir>/prompt_evaluator_metrics.json)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()

    agent_config = get_agent_config(config, 'prompt_generator')
    print(f"🤖 Using model: {agent_config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {agent_config.get('provider', 'Unknown')}")

    # Validate files
    for req_file in args.requirement:
        if not os.path.exists(req_file):
            print(f"Error: Requirement file not found: {req_file}")
            sys.exit(1)

    if not os.path.exists(args.assignment):
        print(f"Error: Assignment file not found: {args.assignment}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_output = Path(args.metrics_output) if args.metrics_output else (output_dir / 'prompt_evaluator_metrics.json')

    # Load requirements
    requirements = []
    try:
        for req_file in args.requirement:
            requirements.append(load_requirement_from_json(req_file))
    except Exception as e:
        print(f"Error loading requirement files: {e}")
        sys.exit(1)

    # Load assignment text
    try:
        with open(args.assignment, 'r', encoding='utf-8') as f:
            assignment_text = f.read().strip()
    except Exception as e:
        print(f"Error reading assignment file: {e}")
        sys.exit(1)

    # Find generated prompt templates
    prompt_files = find_prompt_files(output_dir)
    if not prompt_files:
        print(f"No prompt templates found in {output_dir}")
        sys.exit(1)

    # Initialize evaluator
    try:
        pe = PromptEvaluator(config_path=args.config)
    except Exception as e:
        print(f"Error initializing PromptEvaluator: {e}")
        sys.exit(1)

    results = []

    for i, prompt_path in enumerate(prompt_files):
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                generated_prompt = f.read()

            # Try to map to a requirement: prefer state file if present
            state_path = prompt_path.with_suffix('.json')
            requirement_text = requirements[i] if i < len(requirements) else ''
            if state_path.exists():
                try:
                    with open(state_path, 'r', encoding='utf-8') as sf:
                        state = json.load(sf)
                        req = state.get('requirement') or {}
                        if isinstance(req, dict) and 'requirement' in req:
                            requirement_text = req['requirement']
                except Exception:
                    pass

            # Evaluate using the PromptEvaluator (which uses LangChain LLMs internally)
            start = time.time()
            evaluation = pe.evaluate_prompt(requirement_text, assignment_text, generated_prompt)
            elapsed = time.time() - start

            # normalize evaluation into list of dicts with name/score/rationale/max_score
            normalized = []
            # evaluation may already be a list of dicts
            if isinstance(evaluation, list):
                for entry in evaluation:
                    # try to support both our internal keys and LangSmith style keys
                    name = entry.get('key') or entry.get('name') or entry.get('criterion')
                    score = float(entry.get('score', entry.get('value', 0))) if entry else 0.0
                    rationale = entry.get('rationale') or entry.get('justification') or entry.get('explanation', '')
                    normalized.append({
                        'name': name,
                        'score': score,
                        'max_score': 5.0,
                        'rationale': rationale,
                    })
            else:
                # unexpected format; wrap it
                normalized.append({'name': prompt_path.name, 'score': 0.0, 'max_score': 5.0, 'rationale': str(evaluation)})

            # compute averages
            avg = sum(e['score'] for e in normalized) / len(normalized) if normalized else 0.0

            result = {
                'prompt_file': str(prompt_path),
                'index': i,
                'requirement_text': requirement_text,
                'evaluation': normalized,
                'average_score': avg,
                'evaluation_time': elapsed,
            }

            results.append(result)

            if args.verbose:
                print(f"Evaluated {prompt_path.name}: avg={avg:.2f}, time={elapsed:.2f}s")

        except Exception as e:
            logger.exception(f"Error evaluating prompt {prompt_path}: {e}")
            results.append({'prompt_file': str(prompt_path), 'error': str(e)})

    # Save metrics
    try:
        with open(metrics_output, 'w', encoding='utf-8') as mf:
            json.dump({'metrics': results, 'generated_at': time.time()}, mf, indent=2, ensure_ascii=False)
        print(f"✅ Metrics saved to: {metrics_output}")
    except Exception as e:
        print(f"Error saving metrics file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
