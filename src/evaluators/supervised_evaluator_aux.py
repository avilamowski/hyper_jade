"""
Auxiliary Metrics Evaluator (LangGraph Parallel Version)

Computes auxiliary metrics (MATCH, MISSING, EXTRA) in parallel using LangGraph.
Each auxiliary metric runs as a separate node, and results are aggregated.
"""

from __future__ import annotations
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from ..config import load_config
from ..models import Correction, Submission
from ..agents.utils.reducers import keep_last

logger = logging.getLogger(__name__)


class AuxiliaryMetricsState(TypedDict):
    """State for auxiliary metrics computation"""
    # Inputs
    assignment: Annotated[str, keep_last]
    requirements: Annotated[str, keep_last]
    student_code: Annotated[str, keep_last]
    reference_correction: Annotated[str, keep_last]
    generated_correction: Annotated[str, keep_last]
    
    # Outputs - all metrics stored in a single dictionary
    auxiliary_metrics: Annotated[Dict[str, str], keep_last]
    
    # Metadata
    timings: Annotated[Optional[Dict[str, float]], keep_last]


class AuxiliaryMetricsEvaluator:
    """
    Evaluator for computing auxiliary metrics using LangGraph.
    
    Computes MATCH, MISSING, and EXTRA metrics in parallel, then aggregates results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None):
        if config is not None:
            self.config = config
        else:
            self.config = load_config("src/config/evaluator_config.yaml")
        
        self.evaluator_config = {
            'model_name': self.config['model_name'],
            'provider': self.config['provider'],
            'temperature': self.config['temperature'],
        }
        
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._setup_llm()
        
        # Setup Jinja environment for auxiliary metric templates
        repo_root = Path(__file__).resolve().parents[2]
        self.templates_dir = repo_root / "templates"
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["jinja", "html", "txt"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Load auxiliary metric configurations from config
        supervised_cfg = self.config.get("supervised_evaluator", {})
        self.aux_metric_configs = supervised_cfg.get("auxiliary_metrics", {})
        
        if not self.aux_metric_configs:
            raise ValueError(
                "No auxiliary_metrics found in config. "
                "Expected format: supervised_evaluator.auxiliary_metrics with metric definitions"
            )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _setup_llm(self):
        provider = self.evaluator_config["provider"]
        model_name = self.evaluator_config["model_name"]
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=self.evaluator_config["temperature"],
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        elif provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(
                model=model_name,
                temperature=self.evaluator_config["temperature"]
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _extract_result_text(self, raw_response) -> str:
        """Extract text content from LLM response."""
        if hasattr(raw_response, 'content'):
            text = raw_response.content
        elif isinstance(raw_response, dict):
            choices = raw_response.get('choices')
            if choices and isinstance(choices, list) and len(choices) > 0:
                text = choices[0].get('message', {}).get('content', str(raw_response))
            else:
                text = str(raw_response)
        else:
            text = str(raw_response)
        
        return text.strip()
    
    # -------------------------- LangGraph Nodes ------------------------------ #
    
    def _create_metric_node(self, metric_name: str):
        """Create a node function for computing a specific auxiliary metric"""
        
        def compute_metric_node(state: AuxiliaryMetricsState) -> AuxiliaryMetricsState:
            """Compute a single auxiliary metric"""
            logger.info(f"Computing {metric_name.upper()} auxiliary metric")
            
            t0 = time.time()
            result_text = self._compute_single_metric(metric_name, state)
            elapsed = time.time() - t0
            
            # Update state - add to auxiliary_metrics dictionary
            auxiliary_metrics = state.get("auxiliary_metrics", {}).copy()
            auxiliary_metrics[metric_name] = result_text
            
            timings = state.get("timings", {}).copy()
            timings[metric_name] = elapsed
            
            return {
                "auxiliary_metrics": auxiliary_metrics,
                "timings": timings
            }
        
        return compute_metric_node
    
    def _node_aggregate_results(self, state: AuxiliaryMetricsState) -> AuxiliaryMetricsState:
        """LangGraph node that validates all metrics are present (no-op in parallel execution)"""
        logger.info("Validating auxiliary metrics")
        
        auxiliary_metrics = state.get("auxiliary_metrics", {})
        logger.info(f"Collected {len(auxiliary_metrics)} auxiliary metrics")
        
        # Just pass through - metrics are already in the dictionary
        return {}
    
    def _compute_single_metric(self, metric_name: str, state: AuxiliaryMetricsState) -> str:
        """Helper method to compute a single auxiliary metric using its template"""
        if metric_name not in self.aux_metric_configs:
            raise ValueError(f"Unknown auxiliary metric: {metric_name}")
        
        template_name = self.aux_metric_configs[metric_name]["template"]
        template = self.jinja_env.get_template(template_name)
        
        # Render the template
        rendered = template.render(
            assignment=state.get("assignment", ""),
            requirements=state.get("requirements", ""),
            student_code=state.get("student_code", ""),
            reference_correction=state.get("reference_correction", ""),
            generated_correction=state.get("generated_correction", "")
        )
        
        # Split system/human if separator exists
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = f"Compute auxiliary metric: {metric_name}"
            human_prompt = rendered
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        raw_response = self.llm.invoke(messages)
        result_text = self._extract_result_text(raw_response)
        
        logger.info(f"Completed {metric_name} metric")
        return result_text
    
    # -------------------------- Graph Builder ------------------------------- #
    
    def _build_graph(self):
        """Build a parallel LangGraph for auxiliary metrics computation"""
        graph = StateGraph(AuxiliaryMetricsState)
        
        # Dynamically add nodes for each configured auxiliary metric
        for metric_name in self.aux_metric_configs.keys():
            node_name = f"compute_{metric_name}"
            node_func = self._create_metric_node(metric_name)
            graph.add_node(node_name, node_func)
        
        # Add aggregation node
        graph.add_node("aggregate_results", self._node_aggregate_results)
        
        # Parallel execution: all metrics run concurrently from START
        for metric_name in self.aux_metric_configs.keys():
            node_name = f"compute_{metric_name}"
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, "aggregate_results")
        
        # End after aggregation
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()
    
    # -------------------------- Public API ---------------------------------- #
    
    def compute_all_auxiliary_metrics(
        self,
        generated_text: str,
        reference_text: str,
        submission: Submission,
        assignment: Optional[str] = None,
        requirements: Optional[str] = None,
        metrics_to_compute: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Compute all configured auxiliary metrics using LangGraph.
        
        Args:
            generated_text: AI-generated correction text
            reference_text: Human reference correction text
            submission: Student submission
            assignment: Optional assignment description
            requirements: Optional requirements/rubric
            metrics_to_compute: Optional list of specific metrics to compute.
                              If None, computes all configured metrics.
        
        Returns:
            Dictionary mapping metric names to their plain text outputs
        """
        logger.info("Computing auxiliary metrics via LangGraph")
        
        # Prepare initial state
        initial_state: AuxiliaryMetricsState = {
            "assignment": assignment or "",
            "requirements": requirements or "",
            "student_code": submission.get('code', ''),
            "reference_correction": reference_text,
            "generated_correction": generated_text,
            "auxiliary_metrics": {},  # Initialize as empty dict
            "timings": {}
        }
        
        # If specific metrics requested, we need to build a custom graph
        if metrics_to_compute is not None and set(metrics_to_compute) != set(self.aux_metric_configs.keys()):
            # Build custom graph with only requested metrics
            custom_graph = self._build_custom_graph(metrics_to_compute)
            result_state = custom_graph.invoke(initial_state)
        else:
            # Use default graph with all metrics
            result_state = self.graph.invoke(initial_state)
        
        # Extract auxiliary metrics from result
        aux_metrics = result_state.get("auxiliary_metrics", {})
        
        # Log timings
        timings = result_state.get("timings", {})
        for metric_name, elapsed in timings.items():
            logger.info(f"Metric {metric_name} completed in {elapsed:.2f}s")
        
        return aux_metrics
    
    def _build_custom_graph(self, metrics_to_compute: List[str]):
        """Build a custom graph with only specified metrics"""
        graph = StateGraph(AuxiliaryMetricsState)
        
        # Add only requested metric nodes
        for metric_name in metrics_to_compute:
            if metric_name in self.aux_metric_configs:
                node_name = f"compute_{metric_name}"
                node_func = self._create_metric_node(metric_name)
                graph.add_node(node_name, node_func)
        
        # Add aggregation node
        graph.add_node("aggregate_results", self._node_aggregate_results)
        
        # Connect requested metrics in parallel
        for metric_name in metrics_to_compute:
            if metric_name in self.aux_metric_configs:
                node_name = f"compute_{metric_name}"
                graph.add_edge(START, node_name)
                graph.add_edge(node_name, "aggregate_results")
        
        # End after aggregation
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()
