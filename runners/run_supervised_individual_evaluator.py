#!/usr/bin/env python3
"""Supervised individual evaluator runner using auxiliary and individual metrics."""

import sys
import os
import json
import argparse
import time
import logging
import mlflow
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Union

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.agents.rag_prompt_generator.rag_prompt_generator import RAGPromptGeneratorAgent
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.evaluators.supervised_evaluator_aux import AuxiliaryMetricsEvaluator
from src.evaluators.supervised_evaluator_individual import IndividualMetricsEvaluator
from src.config import load_config, get_agent_config, load_langsmith_config
from src.core.mlflow_utils import mlflow_logger
from src.models import Requirement, GeneratedPrompt, Submission, Correction, PromptType, ReferenceCorrection
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
async def generate_prompts_traced(prompt_generator, requirements: List[Requirement], assignment_text: str) -> List[GeneratedPrompt]:
    """Generate prompts - handles both standard and RAG prompt generators."""
    # Check if it's RAG (async) or standard (sync)
    if isinstance(prompt_generator, RAGPromptGeneratorAgent):
        return await prompt_generator.generate_prompts_batch(requirements, assignment_text)
    else:
        return prompt_generator.generate_prompts_batch(requirements, assignment_text)


@traceable(name="submission_correction_and_evaluation", run_type="chain")
def process_submission_traced(
    code_corrector: CodeCorrectorAgent,
    aux_evaluator: AuxiliaryMetricsEvaluator,
    individual_evaluator: IndividualMetricsEvaluator,
    generated_prompts: List[GeneratedPrompt],
    submission: Submission,
    reference_correction: Union[str, ReferenceCorrection],
    requirements: List[Requirement],
    assignment_text: str,
    submission_index: int,
    shared_llm: Any = None
) -> tuple[List[Correction], Dict[str, Any]]:
    """Process one submission: correct code and evaluate using auxiliary + individual metrics"""
    with trace(name=f"submission_{submission_index+1}_processing", run_type="chain") as run_context:
        run_context.add_tags([f"submission_index:{submission_index}", "correction_and_evaluation"])
        
        # Step 1: Generate corrections
        with trace(name="generate_corrections", run_type="llm") as correction_context:
            correction_context.add_tags(["code_correction", f"prompts_count:{len(generated_prompts)}"])
            submission_corrections = code_corrector.correct_code_batch(generated_prompts, submission, assignment_text)
        
        # Combine all corrections into a single text
        generated_correction_text = "\n\n".join([
            f"## {corr['requirement']['function']}\n{corr['result']}"
            for corr in submission_corrections
        ])
        
        # Switch to evaluation stage for the evaluators
        if shared_llm is not None:
            shared_llm.stage = 'evaluation'
        
        # Step 2: Compute auxiliary metrics
        with trace(name="compute_auxiliary_metrics", run_type="llm") as aux_context:
            aux_context.add_tags(["auxiliary_metrics", "match_missing_extra"])
            aux_metrics = aux_evaluator.compute_all_auxiliary_metrics(
                generated_text=generated_correction_text,
                reference_text=reference_correction,
                submission=submission,
                assignment=assignment_text,
                requirements=requirements  # Pass as list of Requirement objects
            )
        
        # Step 3: Evaluate individual metrics
        with trace(name="evaluate_individual_metrics", run_type="llm") as eval_context:
            eval_context.add_tags(["individual_metrics", "completeness_restraint_etc"])
            evaluation_results = individual_evaluator.evaluate_all_metrics(
                generated_text=generated_correction_text,
                reference_text=reference_correction,
                submission=submission,
                aux_metrics=aux_metrics,
                assignment=assignment_text,
                requirements=requirements  # Pass as list of Requirement objects
            )
        
        # Combine auxiliary and individual results
        combined_results = {
            "auxiliary_metrics": aux_metrics,
            "scores": evaluation_results.get("scores", {}),
            "explanations": evaluation_results.get("explanations", {}),
            "overall_score": evaluation_results.get("overall_score", 0.0),
            "timings": {
                **aux_metrics.get("timings", {}),
                **evaluation_results.get("timings", {})
            },
            "corrections": submission_corrections
        }
    
    return submission_corrections, combined_results


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


def load_reference_corrections(correction_paths: List[str]) -> List[Union[str, ReferenceCorrection]]:
    """Load reference corrections from files (.txt or .json)"""
    corrections = []
    for corr_path in correction_paths:
        path = Path(corr_path)
        if path.suffix == '.json':
            # Load as ReferenceCorrection
            with open(corr_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Direct list of corrections
                    corrections.append({"corrections": data})
                elif isinstance(data, dict) and 'corrections' in data:
                    # Already in ReferenceCorrection format
                    corrections.append(data)
                else:
                    logger.warning(f"Unknown JSON format in {corr_path}, treating as empty correction list")
                    corrections.append({"corrections": []})
        else:
            # Load as plain text
            with open(corr_path, 'r', encoding='utf-8') as f:
                corrections.append(f.read().strip())
    return corrections


def save_results(output_path: str, corrections: List[Correction], evaluation_results: Dict[str, Any], extra: Dict[str, Any] = None):
    """Save corrections, auxiliary metrics, and evaluation results to JSON files"""
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
            "generation_method": "supervised_individual_evaluation"
        },
        "extra": extra or {}
    }
    
    with open(corrections_path, 'w', encoding='utf-8') as f:
        json.dump(corrections_data, f, indent=2, ensure_ascii=False)
    
    # Save auxiliary metrics separately
    aux_metrics_path = Path(output_path) / "auxiliary_metrics.json"
    aux_metrics_data = {
        "auxiliary_metrics": evaluation_results.get("auxiliary_metrics", {}),
        "timestamp": time.time()
    }
    with open(aux_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(aux_metrics_data, f, indent=2, ensure_ascii=False)
    
    # Save evaluation results (scores, explanations, overall score)
    evaluation_path = Path(output_path) / "evaluation_results.json"
    eval_data = {
        "scores": evaluation_results.get("scores", {}),
        "explanations": evaluation_results.get("explanations", {}),
        "overall_score": evaluation_results.get("overall_score", 0.0),
        "timings": evaluation_results.get("timings", {}),
        "timestamp": time.time()
    }
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"- Corrections: {corrections_path}")
    logger.info(f"- Auxiliary metrics: {aux_metrics_path}")
    logger.info(f"- Evaluation: {evaluation_path}")


def main():
    """Main entry point for supervised individual evaluator"""
    parser = argparse.ArgumentParser(
        description="Supervised Individual Evaluator Pipeline (Auxiliary + Individual Metrics)",
        epilog="""
Example usage:
  python run_supervised_individual_evaluator.py \\
    --assignment ejemplos/3p/consigna.txt \\
    --requirements ejemplos/3p/requirements/*.json \\
    --submissions ejemplos/3p/alu*.py \\
    --reference-corrections ejemplos/3p/alu*.json \\
    --output-dir outputs/supervised_individual_evaluation \\
    --experiment-name "baseline_v1"

This will:
1. Generate prompts from requirements using the PromptGeneratorAgent
2. Generate corrections for each submission using the CodeCorrectorAgent
3. Compute auxiliary metrics (MATCH, MISSING, EXTRA) using AuxiliaryMetricsEvaluator
4. Evaluate individual metrics (completeness, restraint, etc.) using IndividualMetricsEvaluator
5. Save corrections, auxiliary metrics, and evaluation results to the output directory
6. Tag LangSmith and MLflow runs with the experiment name for easier tracking

Note: Reference corrections can be provided as either:
  - .txt files (plain text corrections)
  - .json files (structured corrections as a list of strings)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--requirements", "-r", nargs="+", required=True, help="Paths to requirement files (.json)")
    parser.add_argument("--submissions", "-s", nargs="+", required=True, help="Paths to student submission files (.py)")
    parser.add_argument("--reference-corrections", "-c", nargs="+", required=True, help="Paths to reference correction files (.txt or .json)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for results")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--evaluator-config", default="src/config/evaluator_config.yaml", help="Evaluator configuration file path")
    parser.add_argument("--experiment-name", "-e", help="Experiment name for LangSmith tracking (optional)")
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
    
    evaluator_config = load_config(args.evaluator_config)
    if not evaluator_config:
        logger.error("Error: Could not load evaluator configuration")
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
    logger.info(f"🤖 Evaluator Config: {args.evaluator_config}")
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
        
        # Initialize agents and evaluators
        logger.info("Initializing agents and evaluators...")
        
        # Check if RAG is enabled
        enable_rag = config.get('enable_rag', False)
        
        if enable_rag:
            logger.info("🧠 RAG Mode: ENABLED - Using RAGPromptGeneratorAgent")
            prompt_generator = RAGPromptGeneratorAgent(config)
            # RAG requires async initialization
            asyncio.run(prompt_generator.initialize())
        else:
            logger.info("📝 RAG Mode: DISABLED - Using standard PromptGeneratorAgent")
            prompt_generator = PromptGeneratorAgent(config)
        
        code_corrector = CodeCorrectorAgent(config)

        # Build composite LLM from agent configs
        composite = CompositeLLM.from_agent_configs({
            'prompt_generation': get_agent_config(config, 'prompt_generator'),
            'code_correction': get_agent_config(config, 'code_corrector'),
            'evaluation': evaluator_config,
        })

        # Create shared bound object
        shared = composite.get_shared_bound()
        prompt_generator.llm = shared
        code_corrector.llm = shared
        
        # Initialize both evaluators with shared LLM
        aux_evaluator = AuxiliaryMetricsEvaluator(evaluator_config, llm=shared)
        individual_evaluator = IndividualMetricsEvaluator(evaluator_config, llm=shared)
        
        logger.info("✓ Initialized auxiliary metrics evaluator")
        logger.info("✓ Initialized individual metrics evaluator")
        pipeline_tags = []
        if args.experiment_name:
            pipeline_tags.append(f"experiment:{args.experiment_name}")
            logger.info(f"🏷️  Experiment name: {args.experiment_name}")
            
        # Build tags list for LangSmith
        pipeline_tags += [
            f"num_submissions:{len(submissions)}", 
            f"num_requirements:{len(requirements)}",
            "full_pipeline",
            f"assignment:{Path(args.assignment).name}"
        ]
        
        # Wrap entire pipeline in a single LangSmith trace
        with trace(name="supervised_individual_evaluation_pipeline", run_type="chain") as pipeline_context:
            pipeline_context.add_tags(pipeline_tags)
            
            # Step 1: Generate prompts for all requirements
            logger.info("Step 1: Generating prompts (one dedicated MLflow run)...")
            mlflow_tags = {
                "agent": "prompt_generator",
                "phase": "prompt_generation",
                "assignment_file": Path(args.assignment).name
            }
            if args.experiment_name:
                mlflow_tags["experiment_name"] = args.experiment_name
            
            with mlflow_logger.run(run_name="prompt_generation", tags=mlflow_tags):
                shared.stage = 'prompt_generation'
                # Handle async prompt generation for RAG
                if enable_rag:
                    generated_prompts = asyncio.run(generate_prompts_traced(prompt_generator, requirements, assignment_text))
                else:
                    generated_prompts = asyncio.run(generate_prompts_traced(prompt_generator, requirements, assignment_text))
                logger.info(f"✓ Generated {len(generated_prompts)} prompts")
                
                # Save generated prompts to prompts folder
                prompts_dir = Path(args.output_dir) / "prompts"
                prompts_dir.mkdir(parents=True, exist_ok=True)
                
                for j, prompt in enumerate(generated_prompts):
                    prompt_data = {
                        "requirement": {
                            "requirement": prompt["requirement"]["requirement"],
                            "function": prompt["requirement"]["function"],
                            "type": prompt["requirement"]["type"].value if hasattr(prompt["requirement"]["type"], "value") else str(prompt["requirement"]["type"])
                        },
                        "jinja_template": prompt["jinja_template"],
                        "examples": prompt.get("examples", ""),
                        "index": prompt.get("index", j),
                        "timestamp": time.time()
                    }
                    prompt_file = prompts_dir / f"prompt_{j+1:02d}_{prompt['requirement']['function']}.json"
                    with open(prompt_file, 'w', encoding='utf-8') as f:
                        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Saved {len(generated_prompts)} prompts to {prompts_dir}")

            logger.info("Step 2: Processing submissions with unified LangSmith traces...")
            all_evals = []
            for i, submission in enumerate(submissions):
                submission_mlflow_tags = {
                    "agent": "supervised_individual_evaluator",
                    "submission_index": str(i),
                    "assignment_file": Path(args.assignment).name
                }
                if args.experiment_name:
                    submission_mlflow_tags["experiment_name"] = args.experiment_name
                
                with mlflow_logger.run(run_name=f"supervised_individual_evaluation_{i+1}", tags=submission_mlflow_tags):
                    logger.info(f"Processing submission {i+1}/{len(submissions)}")
                    shared.stage = 'code_correction'
                    submission_corrections, evaluation_results = process_submission_traced(
                        code_corrector=code_corrector,
                        aux_evaluator=aux_evaluator,
                        individual_evaluator=individual_evaluator,
                        generated_prompts=generated_prompts,
                        submission=submission,
                        reference_correction=reference_corrections[i],
                        requirements=requirements,
                        assignment_text=assignment_text,
                        submission_index=i,
                        shared_llm=shared
                    )

                    # Use the submission filename (without extension) for unambiguous identification
                    submission_name = Path(args.submissions[i]).stem
                    submission_output_dir = Path(args.output_dir) / submission_name
                    submission_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    save_results(str(submission_output_dir), submission_corrections, evaluation_results)

                    # Log MLflow artifacts and metrics
                    mlflow_logger.log_artifacts(str(submission_output_dir), artifact_path=submission_output_dir.name)
                    
                    # Log overall score
                    overall_score = evaluation_results.get('overall_score')
                    if overall_score is not None:
                        mlflow_logger.log_metric('overall_score', float(overall_score))
                    
                    # Log individual metric scores
                    scores = evaluation_results.get('scores', {})
                    for metric_name, score in scores.items():
                        mlflow_logger.log_metric(f"metric_{metric_name}", float(score))
                    
                    # Log auxiliary metrics as parameters (they are text, not numbers)
                    aux_metrics = evaluation_results.get('auxiliary_metrics', {})
                    for aux_name, aux_value in aux_metrics.items():
                        # Log length as a metric instead of the full text
                        mlflow_logger.log_metric(f"aux_{aux_name}_length", len(aux_value))
                    
                    mlflow_logger.log_metric('num_generated_corrections', len(submission_corrections))

                    all_evals.append({
                        "submission": submission_output_dir.name,
                        "evaluation": evaluation_results
                    })


            # Print aggregate summary
            logger.info("=" * 60)
            logger.info("SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total submissions processed: {len(all_evals)}")
            total_corrections = sum(len(e.get('evaluation', {}).get('corrections', [])) for e in all_evals)
            logger.info(f"Total corrections generated: {total_corrections}")
            
            overall_scores = [
                e['evaluation'].get('overall_score') 
                for e in all_evals 
                if isinstance(e.get('evaluation'), dict) and 'overall_score' in e['evaluation']
            ]
            
            # Compute aggregate metrics
            aggregate_metrics = {
                "total_submissions": len(all_evals),
                "total_corrections": total_corrections,
                "average_overall_score": None,
                "metric_averages": {},
                "timestamp": time.time()
            }
            
            if overall_scores:
                avg_overall = sum(overall_scores) / len(overall_scores)
                aggregate_metrics["average_overall_score"] = avg_overall
                logger.info(f"Average overall evaluation score: {avg_overall:.3f}")
            
            # Print metric averages
            all_scores = {}
            for e in all_evals:
                scores = e.get('evaluation', {}).get('scores', {})
                for metric_name, score in scores.items():
                    if metric_name not in all_scores:
                        all_scores[metric_name] = []
                    all_scores[metric_name].append(score)
            
            if all_scores:
                logger.info("\nMetric Averages:")
                for metric_name, scores_list in sorted(all_scores.items()):
                    avg = sum(scores_list) / len(scores_list)
                    aggregate_metrics["metric_averages"][metric_name] = avg
                    logger.info(f"  {metric_name}: {avg:.3f}")
            
            # Save aggregate metrics to JSON
            aggregate_path = Path(args.output_dir) / "aggregate_metrics.json"
            with open(aggregate_path, 'w', encoding='utf-8') as f:
                json.dump(aggregate_metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Aggregate metrics saved to: {aggregate_path}")
            
            logger.info("=" * 60)
            logger.info("✓ Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
