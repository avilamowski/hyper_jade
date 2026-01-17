#!/usr/bin/env python3
"""
MLflow utilities for logging metrics and artifacts across agents
"""

import mlflow
import logging
import time
import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from langsmith import Client as LangSmithClient
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)

class MLflowLogger:
    """Centralized MLflow logging utility for all agents"""
    
    def __init__(self, experiment_name: str = "assignment_evaluation", config_path: Optional[str] = None):
        self.experiment_name = experiment_name
        self.config = self._load_config(config_path)
        self._setup_experiment()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load MLflow configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "mlflow_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded MLflow config from: {config_path}")
            return config.get('mlflow', {})
        except Exception as e:
            logger.warning(f"Could not load MLflow config from {config_path}: {e}")
            return {}
    
    def _setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            # Set tracking URI if configured
            tracking_uri = self.config.get('tracking_uri')
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"Set MLflow tracking URI: {tracking_uri}")
            
            # Set experiment
            experiment_name = self.config.get('experiment_name', self.experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run and return the run ID"""
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
        return run_id

    from contextlib import contextmanager

    @contextmanager
    def run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager that starts and ends an MLflow run.

        Usage:
            with mlflow_logger.run('name', tags=...):
                ...
        """
        run_id = None
        try:
            run_id = self.start_run(run_name=run_name, tags=tags)
            yield run_id
        except Exception as e:
            logger.exception(f"MLflow run '{run_name}' failed: {e}")
            raise
        finally:
            try:
                self.end_run()
            except Exception:
                pass
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric to MLflow"""
        try:
            mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged metric: {key} = {value}")
        except Exception as e:
            logger.warning(f"Could not log metric {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics to MLflow"""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Could not log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to MLflow"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Could not log artifact {local_path}: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log all files in a directory as artifacts"""
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f"Logged artifacts from directory: {local_dir}")
        except Exception as e:
            logger.warning(f"Could not log artifacts from {local_dir}: {e}")
    
    def log_param(self, key: str, value: Any):
        """Log a parameter to MLflow"""
        try:
            mlflow.log_param(key, value)
            logger.debug(f"Logged parameter: {key} = {value}")
        except Exception as e:
            logger.warning(f"Could not log parameter {key}: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters to MLflow"""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            logger.warning(f"Could not log parameters: {e}")
    
    def log_text(self, text: str, artifact_file: str):
        """Log text as an artifact"""
        try:
            mlflow.log_text(text, artifact_file)
            logger.info(f"Logged text artifact: {artifact_file}")
        except Exception as e:
            logger.warning(f"Could not log text artifact {artifact_file}: {e}")
    
    def log_prompt(self, prompt: str, prompt_name: str, step: str):
        """Log a prompt with metadata for tracing"""
        try:
            # Create prompt metadata
            prompt_data = {
                "prompt": prompt,
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "length_chars": len(prompt),
                "length_lines": len(prompt.split('\n'))
            }
            
            # Log as JSON artifact
            prompt_json = json.dumps(prompt_data, indent=2, ensure_ascii=False)
            mlflow.log_text(prompt_json, f"prompts/{prompt_name}_{step}.json")
            
            # Also log the raw prompt
            mlflow.log_text(prompt, f"prompts/{prompt_name}_{step}.txt")
            
            logger.info(f"Logged prompt for {step}: {prompt_name}")
            
        except Exception as e:
            logger.warning(f"Could not log prompt {prompt_name}: {e}")
    
    def log_trace_step(self, step_name: str, step_data: Dict[str, Any], step_number: int = None):
        """Log a trace step for tracking agent progress"""
        try:
            trace_data = {
                "step_name": step_name,
                "step_number": step_number,
                "timestamp": datetime.now().isoformat(),
                "data": step_data
            }
            
            # Log trace step
            trace_json = json.dumps(trace_data, indent=2, ensure_ascii=False)
            step_id = f"{step_number:02d}" if step_number else "00"
            mlflow.log_text(trace_json, f"traces/step_{step_id}_{step_name}.json")
            
            # Log as metric for tracking
            self.log_metric(f"trace_step_{step_name}_completed", 1.0, step=step_number)
            
            logger.info(f"Logged trace step: {step_name}")
            
        except Exception as e:
            logger.warning(f"Could not log trace step {step_name}: {e}")
    
    def log_agent_input_output(self, agent_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Log agent inputs and outputs for tracing"""
        try:
            io_data = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "inputs": inputs,
                "outputs": outputs
            }
            
            # Log as JSON artifact
            io_json = json.dumps(io_data, indent=2, ensure_ascii=False)
            mlflow.log_text(io_json, f"agent_io/{agent_name}_input_output.json")
            
            logger.info(f"Logged agent I/O for {agent_name}")
            
        except Exception as e:
            logger.warning(f"Could not log agent I/O for {agent_name}: {e}")
    
    def log_code_analysis_metrics(self, analysis_result: str, requirement_name: str):
        """Log specific metrics for code analysis results"""
        try:
            # Extract YES/NO result from analysis
            result = "UNKNOWN"
            if "<RESULT>YES</RESULT>" in analysis_result:
                result = "YES"
            elif "<RESULT>NO ERROR</RESULT>" in analysis_result:
                result = "NO"
            
            # Log the result as a metric
            self.log_metric(f"{requirement_name}_result", 1.0 if result == "YES" else 0.0)
            
            # Log analysis length metrics
            self.log_metrics({
                f"{requirement_name}_analysis_length_chars": len(analysis_result),
                f"{requirement_name}_analysis_length_lines": len(analysis_result.split('\n'))
            })
            
            logger.info(f"Logged code analysis metrics for {requirement_name}: {result}")
            
        except Exception as e:
            logger.warning(f"Could not log code analysis metrics: {e}")
    
    def log_requirement_metrics(self, requirement_text: str, requirement_number: int):
        """Log metrics for individual requirements"""
        try:
            # Log requirement complexity metrics
            self.log_metrics({
                f"requirement_{requirement_number}_length_chars": len(requirement_text),
                f"requirement_{requirement_number}_length_words": len(requirement_text.split()),
                f"requirement_{requirement_number}_length_lines": len(requirement_text.split('\n'))
            })
            
        except Exception as e:
            logger.warning(f"Could not log requirement metrics: {e}")
    
    def log_prompt_metrics(self, prompt_template: str, prompt_name: str):
        """Log metrics for generated prompts"""
        try:
            # Log prompt complexity metrics
            self.log_metrics({
                f"{prompt_name}_template_length_chars": len(prompt_template),
                f"{prompt_name}_template_length_lines": len(prompt_template.split('\n')),
                f"{prompt_name}_has_code_variable": "{{ code }}" in prompt_template
            })
            
        except Exception as e:
            logger.warning(f"Could not log prompt metrics: {e}")

    def log_agent_evaluation_metrics(self, agent_name: str, evaluation_result: Dict[str, Any]):
        """Log evaluation metrics for agent outputs"""
        try:
            # Extract scores from evaluation result
            metrics = {}
            
            # Log overall score
            if "overall_score" in evaluation_result:
                metrics[f"{agent_name}_overall_score"] = evaluation_result["overall_score"]
            
            # Log average criteria score
            if "average_criteria_score" in evaluation_result:
                metrics[f"{agent_name}_average_criteria_score"] = evaluation_result["average_criteria_score"]
            
            # Log individual criteria scores
            for criterion, data in evaluation_result.items():
                if isinstance(data, dict) and "score" in data:
                    metrics[f"{agent_name}_{criterion}_score"] = data["score"]
            
            # Log metrics
            if metrics:
                self.log_metrics(metrics)
                logger.info(f"Logged evaluation metrics for {agent_name}: {list(metrics.keys())}")
            
            # Log detailed evaluation as artifact
            evaluation_json = json.dumps(evaluation_result, indent=2, ensure_ascii=False)
            self.log_text(evaluation_json, f"evaluations/{agent_name}_evaluation.json")
            
        except Exception as e:
            logger.warning(f"Could not log agent evaluation metrics: {e}")

    def log_agent_evaluation_summary(self, evaluations: Dict[str, Dict[str, Any]]):
        """Log summary metrics for all agent evaluations"""
        try:
            summary_metrics = {}
            
            for agent_name, evaluation in evaluations.items():
                if "overall_score" in evaluation:
                    summary_metrics[f"{agent_name}_overall_score"] = evaluation["overall_score"]
                if "average_criteria_score" in evaluation:
                    summary_metrics[f"{agent_name}_average_criteria_score"] = evaluation["average_criteria_score"]
            
            # Calculate overall system score
            if summary_metrics:
                overall_scores = [score for key, score in summary_metrics.items() if key.endswith("_overall_score")]
                if overall_scores:
                    summary_metrics["system_overall_score"] = sum(overall_scores) / len(overall_scores)
                
                average_scores = [score for key, score in summary_metrics.items() if key.endswith("_average_criteria_score")]
                if average_scores:
                    summary_metrics["system_average_criteria_score"] = sum(average_scores) / len(average_scores)
            
            # Log summary metrics
            if summary_metrics:
                self.log_metrics(summary_metrics)
                logger.info(f"Logged evaluation summary: {list(summary_metrics.keys())}")
            
            # Log complete evaluation summary as artifact
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "evaluations": evaluations,
                "summary_metrics": summary_metrics
            }
            summary_json = json.dumps(summary_data, indent=2, ensure_ascii=False)
            self.log_text(summary_json, "evaluations/system_evaluation_summary.json")
            
        except Exception as e:
            logger.warning(f"Could not log evaluation summary: {e}")

def get_run_token_usage_and_cost(trace_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get token usage and cost information from LangSmith trace.
    
    This function retrieves token usage and cost from a specific LangSmith trace.
    Requires trace_id to avoid race conditions when running parallel evaluations.
    
    Args:
        trace_id: LangSmith trace ID (required). If None, tries to get from current context.
    
    Returns:
        Dictionary with token usage and cost information:
        {
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_tokens": int,
            "estimated_cost_usd": float,
            "model_info": list of model/provider info,
            "trace_count": int
        }
    """
    result = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "model_info": [],
        "trace_count": 0
    }
    
    if not os.environ.get("LANGSMITH_API_KEY"):
        logger.warning("LangSmith API key not configured - cannot get token usage")
        return result
    
    # Try to get trace_id from current context if not provided
    if trace_id is None:
        try:
            current_run = get_current_run_tree()
            if current_run and hasattr(current_run, 'trace_id'):
                trace_id = str(current_run.trace_id)
                logger.debug(f"Got trace_id from current context: {trace_id}")
            else:
                logger.warning("No trace_id provided and could not get from context")
                return result
        except Exception as e:
            logger.warning(f"Could not get trace_id from context: {e}")
            return result
    
    if not trace_id:
        logger.warning("trace_id is required but not provided")
        return result
    
    try:
        langsmith_client = LangSmithClient()
        project_name = os.environ.get("LANGSMITH_PROJECT", "default")
        
        logger.info(f"🔍 Searching LangSmith for token usage (trace_id: {trace_id})")
        
        # Flush any pending traces to ensure they're available
        try:
            langsmith_client.flush()
            logger.debug("Flushed LangSmith client to ensure traces are available")
        except Exception as e:
            logger.debug(f"Could not flush LangSmith client: {e}")
        
        # Search for run by trace_id
        run = None
        try:
            # First try: search for runs with this trace_id
            runs_with_trace = list(langsmith_client.list_runs(
                project_name=project_name,
                trace_id=trace_id,
                is_root=True,
                limit=1
            ))
            if runs_with_trace:
                run = runs_with_trace[0]
                logger.debug(f"Found run by trace_id search: {trace_id}")
        except Exception as e:
            logger.debug(f"Could not search by trace_id: {e}")
        
        # Second try: read run directly (trace_id might be the run_id itself)
        if run is None:
            try:
                run = langsmith_client.read_run(trace_id, load_child_runs=False)
                logger.debug(f"Found run by direct trace_id lookup: {trace_id}")
            except Exception as e:
                logger.warning(f"Could not read run by trace_id {trace_id}: {e}")
                return result
        
        # Process run - data is in the root run itself (based on test script)
        if run is None:
            logger.warning(f"Could not find run for trace_id: {trace_id}")
            return result
        
        try:
            # Ensure we have full run data
            if not hasattr(run, 'total_tokens'):
                run = langsmith_client.read_run(run.id, load_child_runs=False)
            
            result["trace_count"] = 1
            
            # Extract tokens (directly available in run)
            input_tokens = getattr(run, 'prompt_tokens', 0) or getattr(run, 'input_tokens', 0) or 0
            output_tokens = getattr(run, 'completion_tokens', 0) or getattr(run, 'output_tokens', 0) or 0
            total_tokens = getattr(run, 'total_tokens', 0) or (input_tokens + output_tokens) or 0
            
            # Extract cost (Decimal type, convert to float)
            total_cost = getattr(run, 'total_cost', None)
            if total_cost is not None:
                # Convert Decimal to float
                if hasattr(total_cost, '__float__'):
                    total_cost = float(total_cost)
                else:
                    total_cost = float(total_cost)
            else:
                total_cost = 0.0
            
            # Set results
            result["total_input_tokens"] = input_tokens
            result["total_output_tokens"] = output_tokens
            result["total_tokens"] = total_tokens
            result["estimated_cost_usd"] = total_cost
            
            # Store info
            if total_tokens > 0:
                result["model_info"].append({
                    "run_id": str(run.id),
                    "run_name": run.name or "unknown",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": total_cost
                })
            
            logger.info(f"✅ LangSmith run '{run.name}': {total_tokens} tokens, ${total_cost:.6f}")
            
        except Exception as e:
            logger.warning(f"Error processing LangSmith run: {e}")
            return result
        
    except Exception as e:
        logger.warning(f"Error getting token usage from LangSmith: {e}")
    
    return result

# Global MLflow logger instance
mlflow_logger = MLflowLogger()
