#!/usr/bin/env python3
"""Supervised evaluator runner."""

import sys
import os
import json
import argparse
import time
import logging
import mlflow
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.evaluators.supervised_evaluator_2step import SupervisedEvaluator2Step
from src.config import load_config, get_agent_config, load_langsmith_config
from src.core.mlflow_utils import mlflow_logger
from src.models import Requirement, GeneratedPrompt, Submission, Correction, PromptType
from src.agents.utils.composite_llm import CompositeLLM

# LangSmith tracing: optional, provide no-op fallbacks when not installed
try:
    from langsmith import trace, traceable
    LANGSMITH_AVAILABLE = True
except Exception:
    LANGSMITH_AVAILABLE = False
    def traceable(name: str = None, run_type: str = None):
        def decorator(func):
            return func
        return decorator
    class NoOpContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def add_tags(self, tags):
            pass
    def trace(name: str = None, run_type: str = None):
        return NoOpContext()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


@traceable(name="generate_prompts", run_type="chain")
def generate_prompts_traced(prompt_generator: PromptGeneratorAgent, requirements: List[Requirement], assignment_text: str) -> List[GeneratedPrompt]:
    return prompt_generator.generate_prompts_batch(requirements, assignment_text)


@traceable(name="submission_correction_and_evaluation", run_type="chain")
def process_submission_traced(
    code_corrector: CodeCorrectorAgent,
    supervised_evaluator: Any,
    generated_prompts: List[GeneratedPrompt],
    submission: Submission,
    reference_correction: str,
    requirements: List[Requirement],
    assignment_text: str,
    submission_index: int
) -> tuple[List[Correction], Dict[str, Any]]:
    # Group correction and evaluation in one LangSmith trace
    with trace(name=f"submission_{submission_index+1}_processing", run_type="chain") as run_context:
        run_context.add_tags([f"submission_index:{submission_index}", "correction_and_evaluation"])
        
        with trace(name="generate_corrections", run_type="llm") as correction_context:
            correction_context.add_tags(["code_correction", f"prompts_count:{len(generated_prompts)}"])
            submission_corrections = code_corrector.correct_code_batch(generated_prompts, submission, assignment_text)
        with trace(name="evaluate_corrections", run_type="llm") as evaluation_context:
            evaluation_context.add_tags(["evaluation", f"corrections_count:{len(submission_corrections)}"])
            evaluation_results = supervised_evaluator.evaluate_corrections(
                generated_corrections=[submission_corrections],
                reference_corrections=[reference_correction],
                submissions=[submission],
                requirements=requirements
            )
    
    return submission_corrections, evaluation_results


def load_assignment(assignment_path: str) -> str:
    """Load assignment description from file"""
    with open(assignment_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_requirements(requirement_paths: List[str]) -> List[Requirement]:
    requirements = []
    for req_path in requirement_paths:
        with open(req_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        prompt_type = PromptType(data["type"]) if isinstance(data.get("type"), str) else data.get("type")
        requirements.append({
            "requirement": data["requirement"],
            "function": data["function"],
            "type": prompt_type
        })
    return requirements


def load_submissions(submission_paths: List[str]) -> List[Submission]:
    submissions = []
    for sub_path in submission_paths:
        with open(sub_path, 'r', encoding='utf-8') as f:
            code = f.read()
        submissions.append({"code": code})
    return submissions


def load_reference_corrections(correction_paths: List[str]) -> List[str]:
    corrections = []
    for corr_path in correction_paths:
        with open(corr_path, 'r', encoding='utf-8') as f:
            corrections.append(f.read().strip())
    return corrections


def save_results(output_path: str, corrections: List[Correction], evaluation_results: Dict[str, Any], extra: Dict[str, Any] = None):
    """Save corrections, evaluation results, and extra data to JSON files"""
    # Save corrections
    corrections_path = Path(output_path) / "generated_corrections.json"
    corrections_data = {
        "corrections": [
            {
                "requirement": {
                    "requirement": corr["requirement"]["requirement"],
                    "function": corr["requirement"]["function"],
                    "type": corr["requirement"]["type"].value
                },
                "result": corr["result"]
            }
            for corr in corrections
        ],
        "timestamp": time.time(),
        "metadata": {
            "total_corrections": len(corrections),
            "generation_method": "supervised_evaluation"
        },
        "extra": extra or {}
    }
    
    with open(corrections_path, 'w', encoding='utf-8') as f:
        json.dump(corrections_data, f, indent=2, ensure_ascii=False)
    
    # Save evaluation results
    evaluation_path = Path(output_path) / "evaluation_results.json"
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"- Corrections: {corrections_path}")
    logger.info(f"- Evaluation: {evaluation_path}")


def main():
    """Main entry point for supervised evaluator"""
    parser = argparse.ArgumentParser(
        description="Supervised Evaluator Pipeline",
        epilog="""
Example usage:
  python run_supervised_evaluator.py \\
    --assignment ejemplos/3p/consigna.txt \\
    --requirements ejemplos/3p/requirements/*.json \\
    --submissions ejemplos/3p/alu*.py \\
    --reference-corrections ejemplos/3p/alu*.txt \\
    --output-dir outputs/supervised_evaluation

This will:
1. Generate prompts from requirements using the PromptGeneratorAgent
2. Generate corrections for each submission using the CodeCorrectorAgent  
3. Compare generated corrections with reference corrections using SupervisedEvaluator
4. Save both generated corrections and evaluation metrics to the output directory
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--requirements", "-r", nargs="+", required=True, help="Paths to requirement files (.json)")
    parser.add_argument("--submissions", "-s", nargs="+", required=True, help="Paths to student submission files (.py)")
    parser.add_argument("--reference-corrections", "-c", nargs="+", required=True, help="Paths to reference correction files (.txt)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for results")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate input counts
    if len(args.submissions) != len(args.reference_corrections):
        logger.error("Number of submissions must equal number of reference corrections")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()
    
    # LangSmith configuration: warn but don't fail to preserve existing callers
    if LANGSMITH_AVAILABLE:
        if not os.environ.get("LANGSMITH_API_KEY") or not os.environ.get("LANGSMITH_PROJECT"):
            logger.warning("LangSmith available but not fully configured; traces may be incomplete")
        else:
            logger.info(f"🔗 LangSmith tracing enabled for project: {os.environ.get('LANGSMITH_PROJECT')}")
    else:
        logger.info("LangSmith library not installed; running without LangSmith tracing")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    prompt_config = get_agent_config(config, 'prompt_generator')
    corrector_config = get_agent_config(config, 'code_corrector')
    logger.info(f"🤖 Prompt Generator Model: {prompt_config.get('model_name', 'Unknown')}")
    logger.info(f"🤖 Code Corrector Model: {corrector_config.get('model_name', 'Unknown')}")
    logger.info(f"📊 Requirements: {len(args.requirements)}")
    logger.info(f"📝 Submissions: {len(args.submissions)}")
    logger.info(f"📋 Reference corrections: {len(args.reference_corrections)}")
    logger.info("-" * 60)
    
    try:
        # Load all inputs
        logger.info("Loading inputs...")
        assignment_text = load_assignment(args.assignment)
        requirements = load_requirements(args.requirements)
        submissions = load_submissions(args.submissions)
        reference_corrections = load_reference_corrections(args.reference_corrections)
        
        logger.info(f"✓ Loaded {len(requirements)} requirements")
        logger.info(f"✓ Loaded {len(submissions)} submissions")
        logger.info(f"✓ Loaded {len(reference_corrections)} reference corrections")
        
        # Initialize agents and optionally override their LLMs with a CompositeLLM
        logger.info("Initializing agents...")
        prompt_generator = PromptGeneratorAgent(config)
        code_corrector = CodeCorrectorAgent(config)

        # Load evaluator config separately
        evaluator_config = load_config("src/config/evaluator_config.yaml")

        # Build composite LLM from agent configs
        composite = CompositeLLM.from_agent_configs({
            'prompt_generation': get_agent_config(config, 'prompt_generator'),
            'code_correction': get_agent_config(config, 'code_corrector'),
            'evaluation': evaluator_config,  # Use evaluator config directly
        })

        # Create shared bound object
        shared = composite.get_shared_bound()
        prompt_generator.llm = shared
        code_corrector.llm = shared
        # Use the 2-step supervised evaluator implementation
        supervised_evaluator = SupervisedEvaluator2Step(evaluator_config, llm=shared)

        
        # Step 1: Generate prompts for all requirements
        logger.info("Step 1: Generating prompts (one dedicated MLflow run)...")
        with mlflow_logger.run(run_name="prompt_generation", tags={
            "agent": "prompt_generator",
            "phase": "prompt_generation",
            "assignment_file": Path(args.assignment).name
        }):
            shared.stage = 'prompt_generation'
            generated_prompts = generate_prompts_traced(prompt_generator, requirements, assignment_text)
            logger.info(f"✓ Generated {len(generated_prompts)} prompts")

        logger.info("Step 2: Processing submissions with unified LangSmith traces...")
        all_evals = []
        for i, submission in enumerate(submissions):
            with mlflow_logger.run(run_name=f"supervised_evaluation_{i+1}", tags={
                "agent": "supervised_evaluator",
                "submission_index": str(i),
                "assignment_file": Path(args.assignment).name
            }):
                logger.info(f"Processing submission {i+1}/{len(submissions)}")
                shared.stage = 'code_correction'
                submission_corrections, evaluation_results = process_submission_traced(
                    code_corrector=code_corrector,
                    supervised_evaluator=supervised_evaluator,
                    generated_prompts=generated_prompts,
                    submission=submission,
                    reference_correction=reference_corrections[i],
                    requirements=requirements,
                    assignment_text=assignment_text,
                    submission_index=i
                )

                submission_output_dir = Path(args.output_dir) / f"submission_{i+1}"
                submission_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract extra data for persistence
                extra_data = evaluation_results.get('extra', {})
                save_results(str(submission_output_dir), submission_corrections, evaluation_results, extra=extra_data)

                mlflow_logger.log_artifacts(str(submission_output_dir), artifact_path=submission_output_dir.name)
                if isinstance(evaluation_results, dict):
                    overall = evaluation_results.get('overall_score')
                    if overall is not None:
                        mlflow_logger.log_metric('overall_score', float(overall))

                    criterion_avgs = evaluation_results.get('criterion_averages', {})
                    if isinstance(criterion_avgs, dict):
                        for crit, val in criterion_avgs.items():
                            mlflow_logger.log_metric(f"criterion_avg_{crit}", float(val))

                    mlflow_logger.log_metric('num_generated_corrections', len(submission_corrections))

                all_evals.append({"submission": submission_output_dir.name, "evaluation": evaluation_results})

        # Print aggregate summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total submissions processed: {len(all_evals)}")
        total_corrections = sum(len(e.get('evaluation', {}).get('corrections', [])) for e in all_evals)
        logger.info(f"Total corrections generated (approx): {total_corrections}")
        overall_scores = [e['evaluation'].get('overall_score') for e in all_evals if isinstance(e.get('evaluation'), dict) and 'overall_score' in e['evaluation']]
        
        # Compute aggregate metrics
        aggregate_metrics = {
            "total_submissions": len(all_evals),
            "total_corrections": total_corrections,
            "average_overall_score": None,
            "criterion_averages": {},
            "timestamp": time.time()
        }
        
        if overall_scores:
            avg_overall = sum(overall_scores) / len(overall_scores)
            aggregate_metrics["average_overall_score"] = avg_overall
            logger.info(f"Average overall evaluation score: {avg_overall:.3f}")
        
        # Calculate average for each criterion across all submissions
        all_criterion_scores = {}
        for eval_data in all_evals:
            evaluation = eval_data.get('evaluation', {})
            if isinstance(evaluation, dict):
                criterion_avgs = evaluation.get('criterion_averages', {})
                if isinstance(criterion_avgs, dict):
                    for criterion, score in criterion_avgs.items():
                        if criterion not in all_criterion_scores:
                            all_criterion_scores[criterion] = []
                        all_criterion_scores[criterion].append(score)
        
        for criterion, scores in all_criterion_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                aggregate_metrics["criterion_averages"][criterion] = avg_score
                logger.info(f"Average {criterion}: {avg_score:.3f}")
        
        # Save aggregate metrics to JSON
        aggregate_path = Path(args.output_dir) / "aggregate_metrics.json"
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump(aggregate_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Aggregate metrics saved to: {aggregate_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
