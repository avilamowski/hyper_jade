"""
Individual Evaluation Metrics Evaluator (LangGraph Version)

Computes individual evaluation metrics (completeness, restraint, precision, 
content_similarity, correctness) using auxiliary metrics as inputs.
Each metric runs as a separate LangGraph node and can execute in parallel.
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


class IndividualMetricsState(TypedDict):
    """State for individual evaluation metrics computation"""
    # Inputs
    assignment: Annotated[str, keep_last]
    requirements: Annotated[str, keep_last]
    student_code: Annotated[str, keep_last]
    reference_correction: Annotated[str, keep_last]
    generated_correction: Annotated[str, keep_last]
    auxiliary_metrics: Annotated[Dict[str, str], keep_last]
    
    # Outputs - all scores and explanations stored in dictionaries
    scores: Annotated[Dict[str, int], keep_last]
    explanations: Annotated[Dict[str, str], keep_last]
    overall_score: Annotated[Optional[float], keep_last]
    
    # Metadata
    timings: Annotated[Optional[Dict[str, float]], keep_last]


class IndividualMetricsEvaluator:
    """
    Evaluator for computing individual evaluation metrics using LangGraph.
    
    Each metric is computed via a separate LLM call using its own template.
    Metrics can run in parallel since they have independent dependencies on auxiliary metrics.
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
        
        # Load evaluation metrics configurations from config
        supervised_cfg = self.config.get("supervised_evaluator", {})
        self.eval_metric_configs = supervised_cfg.get("evaluation_metrics", {})
        
        if not self.eval_metric_configs:
            raise ValueError(
                "No evaluation_metrics found in config. "
                "Expected format: supervised_evaluator.evaluation_metrics with metric definitions"
            )
        
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._setup_llm()
        
        # Setup Jinja environment
        repo_root = Path(__file__).resolve().parents[2]
        self.templates_dir = repo_root / "templates"
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["jinja", "html", "txt"]),
            trim_blocks=True,
            lstrip_blocks=True,
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
    
    def _parse_metric_response(self, response_text: str, metric_name: str) -> Dict[str, Any]:
        """
        Parse the response from an individual metric evaluation.
        
        Expected format:
        <metric_name>: <score> - <explanation>
        
        <RESULT>
        <SCORE><score></SCORE>
        <EXPLANATION><explanation></EXPLANATION>
        </RESULT>
        """
        score = None
        explanation = ""
        
        # Try to extract XML RESULT block first
        from xml.etree import ElementTree
        
        m = re.search(r"<RESULT>(.*?)</RESULT>", response_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            xml_text = m.group(1)
            try:
                root = ElementTree.fromstring(f"<root>{xml_text}</root>")
                score_elem = root.find(".//SCORE")
                expl_elem = root.find(".//EXPLANATION")
                
                if score_elem is not None and score_elem.text:
                    score = int(score_elem.text.strip())
                if expl_elem is not None and expl_elem.text:
                    explanation = expl_elem.text.strip()
            except (ElementTree.ParseError, ValueError) as e:
                logger.warning(f"Failed to parse XML for {metric_name}: {e}")
        
        # Fallback to line parsing if XML parsing failed or no XML found
        if score is None:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if ':' not in line:
                    continue
                
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                
                key = parts[0].strip().lower()
                value_part = parts[1].strip()
                
                # Check if this line is for our metric
                if metric_name.lower() in key:
                    # Extract score (first number in the value)
                    score_match = re.search(r'(\d+)', value_part)
                    if score_match:
                        score = int(score_match.group(1))
                    
                    # Extract explanation (after the dash)
                    if ' - ' in value_part:
                        explanation = value_part.split(' - ', 1)[1].strip()
                    break
        
        if score is None:
            logger.warning(f"Could not parse score for {metric_name}, defaulting to 0")
            score = 0
        
        return {
            "score": score,
            "explanation": explanation
        }
    
    # -------------------------- LangGraph Nodes ------------------------------ #
    
    def _create_metric_node(self, metric_name: str):
        """Create a node function for evaluating a specific metric"""
        
        def evaluate_metric_node(state: IndividualMetricsState) -> IndividualMetricsState:
            """Evaluate a single metric"""
            logger.info(f"Evaluating {metric_name} metric")
            
            t0 = time.time()
            score, explanation = self._evaluate_single_metric(metric_name, state)
            elapsed = time.time() - t0
            
            # Update state - add to scores and explanations dictionaries
            scores = state.get("scores", {}).copy()
            scores[metric_name] = score
            
            explanations = state.get("explanations", {}).copy()
            explanations[metric_name] = explanation
            
            timings = state.get("timings", {}).copy()
            timings[metric_name] = elapsed
            
            return {
                "scores": scores,
                "explanations": explanations,
                "timings": timings
            }
        
        return evaluate_metric_node
    
    def _node_aggregate_results(self, state: IndividualMetricsState) -> IndividualMetricsState:
        """LangGraph node that computes the overall score"""
        logger.info("Computing overall score")
        
        scores = state.get("scores", {})
        
        # Calculate overall score as average
        if scores:
            overall_score = sum(scores.values()) / len(scores)
        else:
            overall_score = 0.0
        
        logger.info(f"Overall score from {len(scores)} metrics: {overall_score:.2f}")
        
        return {
            "overall_score": overall_score
        }
    
    def _evaluate_single_metric(self, metric_name: str, state: IndividualMetricsState) -> tuple[int, str]:
        """Helper method to evaluate a single metric using its template"""
        if metric_name not in self.eval_metric_configs:
            raise ValueError(f"Unknown evaluation metric: {metric_name}")
        
        config = self.eval_metric_configs[metric_name]
        template_name = config["template"]
        required_aux = config.get("required_aux_metrics", [])
        
        # Check that required auxiliary metrics are available
        aux_metrics = state.get("auxiliary_metrics", {})
        missing_aux = [aux for aux in required_aux if aux not in aux_metrics]
        if missing_aux:
            logger.warning(
                f"Metric {metric_name} requires auxiliary metrics {missing_aux} "
                f"which are not provided. Available: {list(aux_metrics.keys())}"
            )
        
        template = self.jinja_env.get_template(template_name)
        
        # Render the template with all inputs
        rendered = template.render(
            assignment=state.get("assignment", ""),
            requirements=state.get("requirements", ""),
            student_code=state.get("student_code", ""),
            reference_correction=state.get("reference_correction", ""),
            generated_correction=state.get("generated_correction", ""),
            aux_metrics=aux_metrics
        )
        
        # Split system/human if separator exists
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = f"Evaluate metric: {metric_name}"
            human_prompt = rendered
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        raw_response = self.llm.invoke(messages)
        result_text = self._extract_result_text(raw_response)
        parsed = self._parse_metric_response(result_text, metric_name)
        
        logger.info(f"Completed {metric_name}: score={parsed['score']}")
        
        return parsed['score'], parsed['explanation']
    
    # -------------------------- Graph Builder ------------------------------- #
    
    def _build_graph(self):
        """Build a parallel LangGraph for evaluation metrics computation"""
        graph = StateGraph(IndividualMetricsState)
        
        # Dynamically add nodes for each configured evaluation metric
        for metric_name in self.eval_metric_configs.keys():
            node_name = f"evaluate_{metric_name}"
            node_func = self._create_metric_node(metric_name)
            graph.add_node(node_name, node_func)
        
        # Add aggregation node
        graph.add_node("aggregate_results", self._node_aggregate_results)
        
        # Parallel execution: all metrics run concurrently from START
        for metric_name in self.eval_metric_configs.keys():
            node_name = f"evaluate_{metric_name}"
            graph.add_edge(START, node_name)
            graph.add_edge(node_name, "aggregate_results")
        
        # End after aggregation
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()
    
    # -------------------------- Public API ---------------------------------- #
    
    # -------------------------- Public API ---------------------------------- #
    
    def evaluate_all_metrics(
        self,
        aux_metrics: Dict[str, str],
        generated_text: str,
        reference_text: str,
        submission: Submission,
        assignment: Optional[str] = None,
        requirements: Optional[str] = None,
        metrics_to_evaluate: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all configured metrics using auxiliary metrics via LangGraph.
        
        Args:
            aux_metrics: Dictionary of auxiliary metric outputs
            generated_text: AI-generated correction text
            reference_text: Human reference correction text
            submission: Student submission
            assignment: Optional assignment description
            requirements: Optional requirements/rubric
            metrics_to_evaluate: Optional list of specific metrics to evaluate.
                                If None, evaluates all configured metrics.
        
        Returns:
            Dictionary with 'scores', 'explanations', 'overall_score', and 'timings'
        """
        logger.info("Evaluating metrics via LangGraph")
        
        # Prepare initial state
        initial_state: IndividualMetricsState = {
            "assignment": assignment or "",
            "requirements": requirements or "",
            "student_code": submission.get('code', ''),
            "reference_correction": reference_text,
            "generated_correction": generated_text,
            "auxiliary_metrics": aux_metrics,
            "scores": {},  # Initialize as empty dict
            "explanations": {},  # Initialize as empty dict
            "overall_score": None,
            "timings": {}
        }
        
        # If specific metrics requested, build custom graph
        if metrics_to_evaluate is not None and set(metrics_to_evaluate) != set(self.eval_metric_configs.keys()):
            custom_graph = self._build_custom_graph(metrics_to_evaluate)
            result_state = custom_graph.invoke(initial_state)
        else:
            # Use default graph with all metrics
            result_state = self.graph.invoke(initial_state)
        
        # Extract results
        scores = result_state.get("scores", {})
        explanations = result_state.get("explanations", {})
        overall_score = result_state.get("overall_score", 0.0)
        timings = result_state.get("timings", {})
        
        # Log timings
        for metric_name, elapsed in timings.items():
            logger.info(f"Metric {metric_name} completed in {elapsed:.2f}s")
        
        return {
            'scores': scores,
            'explanations': explanations,
            'overall_score': overall_score,
            'timings': timings
        }
    
    def _build_custom_graph(self, metrics_to_evaluate: List[str]):
        """Build a custom graph with only specified metrics"""
        graph = StateGraph(IndividualMetricsState)
        
        # Add only requested metric nodes
        for metric_name in metrics_to_evaluate:
            if metric_name in self.eval_metric_configs:
                node_name = f"evaluate_{metric_name}"
                node_func = self._create_metric_node(metric_name)
                graph.add_node(node_name, node_func)
        
        # Add aggregation node
        graph.add_node("aggregate_results", self._node_aggregate_results)
        
        # Connect requested metrics in parallel
        for metric_name in metrics_to_evaluate:
            if metric_name in self.eval_metric_configs:
                node_name = f"evaluate_{metric_name}"
                graph.add_edge(START, node_name)
                graph.add_edge(node_name, "aggregate_results")
        
        # End after aggregation
        graph.add_edge("aggregate_results", END)
        
        return graph.compile()
