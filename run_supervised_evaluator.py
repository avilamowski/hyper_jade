#!/usr/bin/env python3
"""
Supervised Evaluator - Top-level orchestrator

This script coordinates the supervised evaluation pipeline:
1. Generates prompts from requirements using PromptGeneratorAgent
2. Builds evaluation dataset from submissions and reference corrections
3. Generates corrections using CodeCorrectorAgent
4. Evaluates generated corrections against reference corrections using SupervisedEvaluator

Receives:
- assignment txt
- requirement jsons
- submissions list of .py files
- reference correction list of .txt files
- output path for results

Outputs:
- corrections json
- evaluation of corrections json
"""

import sys
import os
import json
import argparse
import time
import logging
import mlflow
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.evaluators.supervised_evaluator import SupervisedEvaluator
from src.config import load_config, get_agent_config, load_langsmith_config
from src.core.mlflow_utils import mlflow_logger
from src.models import Requirement, GeneratedPrompt, Submission, Correction, PromptType
from src.agents.utils.composite_llm import CompositeLLM

# Import LangSmith tracing utilities
try:
    from langsmith import trace, traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Define no-op decorators and context managers
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
    """Generate prompts with LangSmith tracing"""
    return prompt_generator.generate_prompts_batch(requirements, assignment_text)


@traceable(name="submission_correction_and_evaluation", run_type="chain")
def process_submission_traced(
    code_corrector: CodeCorrectorAgent,
    supervised_evaluator: SupervisedEvaluator,
    generated_prompts: List[GeneratedPrompt],
    submission: Submission,
    reference_correction: str,
    requirements: List[Requirement],
    assignment_text: str,
    submission_index: int
) -> tuple[List[Correction], Dict[str, Any]]:
    """Process a single submission with both correction and evaluation in one LangSmith trace"""
    
    # Create a context that groups both operations
    with trace(name=f"submission_{submission_index+1}_processing", run_type="chain") as run_context:
        run_context.add_tags([f"submission_index:{submission_index}", "correction_and_evaluation"])
        
        # Generate corrections for this submission
        with trace(name="generate_corrections", run_type="llm") as correction_context:
            correction_context.add_tags(["code_correction", f"prompts_count:{len(generated_prompts)}"])
            submission_corrections = code_corrector.correct_code_batch(generated_prompts, submission, assignment_text)
        
        # Evaluate the corrections for this submission
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
    try:
        with open(assignment_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to load assignment from {assignment_path}: {e}")


def load_requirements(requirement_paths: List[str]) -> List[Requirement]:
    """Load requirements from JSON files"""
    requirements = []
    for req_path in requirement_paths:
        try:
            with open(req_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert type string to PromptType enum
            prompt_type = PromptType(data["type"]) if isinstance(data["type"], str) else data["type"]
            
            requirement: Requirement = {
                "requirement": data["requirement"],
                "function": data["function"],
                "type": prompt_type
            }
            requirements.append(requirement)
        except Exception as e:
            logger.error(f"Failed to load requirement from {req_path}: {e}")
            continue
    
    return requirements


def load_submissions(submission_paths: List[str]) -> List[Submission]:
    """Load student submissions from Python files"""
    submissions = []
    for sub_path in submission_paths:
        try:
            with open(sub_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            submission: Submission = {
                "code": code
            }
            submissions.append(submission)
        except Exception as e:
            logger.error(f"Failed to load submission from {sub_path}: {e}")
            continue
    
    return submissions


def load_reference_corrections(correction_paths: List[str]) -> List[str]:
    """Load reference corrections from text files"""
    corrections = []
    for corr_path in correction_paths:
        try:
            with open(corr_path, 'r', encoding='utf-8') as f:
                correction = f.read().strip()
            corrections.append(correction)
        except Exception as e:
            logger.error(f"Failed to load reference correction from {corr_path}: {e}")
            continue
    
    return corrections


def save_results(output_path: str, corrections: List[Correction], evaluation_results: Dict[str, Any]):
    """Save both corrections and evaluation results to JSON files"""
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
        }
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
    
    # Log LangSmith tracing status
    if LANGSMITH_AVAILABLE:
        langsmith_project = os.environ.get("LANGSMITH_PROJECT")
        if langsmith_project:
            logger.info(f"🔗 LangSmith tracing enabled for project: {langsmith_project}")
        else:
            logger.info("🔗 LangSmith available but no project configured")
    else:
        logger.info("⚠️  LangSmith not available - traces will not be created")
    
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

        # Build a composite LLM routing map from agent configs. This allows the
        # runner to use different providers per stage while presenting a single
        # top-level object if desired. We keep changes confined to the runner
        # by replacing the `.llm` attribute on the agents after construction.
        try:
            prompt_cfg = get_agent_config(config, 'prompt_generator')
            corrector_cfg = get_agent_config(config, 'code_corrector')
            evaluator_cfg = get_agent_config(config, 'agent_evaluator')

            mapping = {
                'prompt_generation': prompt_cfg,
                'code_correction': corrector_cfg,
                'evaluation': evaluator_cfg,
            }

            composite = CompositeLLM.from_agent_configs(mapping)

            # Create one shared bound object so all stages use the same Python
            # object (useful to keep tracing under one wrapper). The runner
            # will set shared.stage before calling each agent.
            shared = composite.get_shared_bound()

            # Assign the same shared object to agents. We'll set `shared.stage`
            # prior to each invocation to route to the correct backend.
            prompt_generator.llm = shared
            code_corrector.llm = shared

            # Evaluator we create with the same shared object as well so
            # evaluation calls remain in the same wrapper/tracing context.
            supervised_evaluator = SupervisedEvaluator(config, llm=shared)
        except Exception as e:
            logger.warning(f"CompositeLLM not fully configured or failed to init: {e}. Falling back to default agent LLMs.")
            # Fall back to original behavior: share the code_corrector's LLM instance
            supervised_evaluator = SupervisedEvaluator(config, llm=code_corrector.llm)
        
        # Step 1: Generate prompts for all requirements
        logger.info("Step 1: Generating prompts (one dedicated MLflow run)...")
        try:
            mlflow_logger.start_run(run_name="prompt_generation", tags={
                "agent": "prompt_generator",
                "phase": "prompt_generation",
                "assignment_file": Path(args.assignment).name
            })

            # Ensure shared LLM routes to prompt generation backend
            try:
                shared.stage = 'prompt_generation'
            except Exception:
                pass

            # Use the traced function for prompt generation
            generated_prompts = generate_prompts_traced(prompt_generator, requirements, assignment_text)
            logger.info(f"✓ Generated {len(generated_prompts)} prompts")

        except Exception as e:
            logger.exception(f"Prompt generation failed: {e}")
            raise
        finally:
            mlflow_logger.end_run()
        
        # Step 2: For each submission: generate corrections and evaluate within the same LangSmith trace
        logger.info("Step 2: Processing submissions with unified LangSmith traces...")
        all_evals = []
        for i, submission in enumerate(submissions):
            submission_name = f"submission_{i+1}"
            try:
                # start a run per submission
                current_run_id = mlflow_logger.start_run(run_name=f"supervised_evaluation_{i+1}", tags={
                    "agent": "supervised_evaluator",
                    "submission_index": str(i),
                    "assignment_file": Path(args.assignment).name
                })

                logger.info(f"Processing submission {i+1}/{len(submissions)}")

                # Set LLM stages for the operations (though they'll be in the same LangSmith trace)
                try:
                    shared.stage = 'code_correction'  # This will be used for both correction and evaluation
                except Exception:
                    pass

                # Use the traced function that combines both correction and evaluation
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

                # Save per-submission results
                submission_output_dir = Path(args.output_dir) / f"submission_{i+1}"
                submission_output_dir.mkdir(parents=True, exist_ok=True)
                save_results(str(submission_output_dir), submission_corrections, evaluation_results)

                # Log saved artifacts and key metrics to the current MLflow run
                try:
                    # Attach the saved files as artifacts to the current run
                    mlflow_logger.log_artifacts(str(submission_output_dir), artifact_path=submission_output_dir.name)

                    # Log main evaluation metrics if available
                    if isinstance(evaluation_results, dict):
                        overall = evaluation_results.get('overall_score')
                        if overall is not None:
                            mlflow_logger.log_metric('overall_score', float(overall))

                        # Log per-criterion averages if present
                        criterion_avgs = evaluation_results.get('criterion_averages', {})
                        if isinstance(criterion_avgs, dict):
                            for crit, val in criterion_avgs.items():
                                try:
                                    mlflow_logger.log_metric(f"criterion_avg_{crit}", float(val))
                                except Exception:
                                    continue

                        # Also log the number of generated corrections for this submission
                        try:
                            mlflow_logger.log_metric('num_generated_corrections', len(submission_corrections))
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Failed to log artifacts/metrics to MLflow for submission {i+1}: {e}")

                all_evals.append({"submission": submission_output_dir.name, "evaluation": evaluation_results})

            except Exception as e:
                logger.exception(f"Error processing submission {i+1}: {e}")
            finally:
                mlflow_logger.end_run()

        # Print aggregate summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total submissions processed: {len(all_evals)}")
        total_corrections = sum(len(e.get('evaluation', {}).get('corrections', [])) for e in all_evals)
        logger.info(f"Total corrections generated (approx): {total_corrections}")
        overall_scores = [e['evaluation'].get('overall_score') for e in all_evals if isinstance(e.get('evaluation'), dict) and 'overall_score' in e['evaluation']]
        if overall_scores:
            logger.info(f"Average overall evaluation score: {sum(overall_scores)/len(overall_scores):.3f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
