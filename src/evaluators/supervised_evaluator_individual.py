"""
Individual Evaluation Metrics Evaluator (LangGraph Version)

Computes individual evaluation metrics (completeness, restraint, precision, 
content_similarity) using auxiliary metrics as inputs.
Each metric runs as a separate LangGraph node and can execute in parallel.
"""

from __future__ import annotations
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from ..config import load_config
from ..models.models import Correction, Submission, Requirement, ReferenceCorrection
from ..agents.utils.reducers import keep_last, merge_dicts
from .metric_computers import compute_completeness, compute_restraint, compute_precision

logger = logging.getLogger(__name__)


class IndividualMetricsState(TypedDict):
    """State for individual evaluation metrics computation"""
    # Inputs
    assignment: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]
    student_code: Annotated[str, keep_last]
    reference_correction: Annotated[Union[str, ReferenceCorrection], keep_last]
    generated_correction: Annotated[str, keep_last]
    auxiliary_metrics: Annotated[Dict[str, str], keep_last]
    
    # Outputs - all scores and explanations stored in dictionaries
    scores: Annotated[Dict[str, float], merge_dicts]
    explanations: Annotated[Dict[str, str], merge_dicts]
    overall_score: Annotated[Optional[float], keep_last]
    
    # Metadata
    timings: Annotated[Optional[Dict[str, float]], merge_dicts]


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
    
    @staticmethod
    def _format_requirements_as_xml(requirements: List[Requirement]) -> str:
        """
        Format a list of requirements as XML tags.
        
        Args:
            requirements: List of Requirement dictionaries with 'requirement', 'function', and 'type' keys
            
        Returns:
            XML-formatted string with requirements using <requirement> tags
            
        Example:
            <requirement function="vender_productos" type="error_presence">
            No recibe por parámetro la ruta.
            </requirement>
            <requirement function="vender_productos" type="error_presence">
            No valida que el parámetro 'cantidad' sea un entero positivo.
            </requirement>
        """
        if not requirements:
            return ""
        
        xml_parts = []
        for req in requirements:
            req_type = req['type'].value if hasattr(req['type'], 'value') else str(req['type'])
            xml_parts.append(
                f'<requirement function="{req["function"]}" type="{req_type}">\n'
                f'{req["requirement"]}\n'
                f'</requirement>'
            )
        
        return "\n".join(xml_parts)
    
    @staticmethod
    def _format_reference_correction(reference_correction: Union[str, ReferenceCorrection]) -> str:
        """
        Convert ReferenceCorrection to XML format.
        
        Args:
            reference_correction: Either a string or ReferenceCorrection dict
            
        Returns:
            XML-formatted string of the reference correction
        
        Example:
            <reference_corrections>
                <correction>No agrega a ventas pasadas el stock restado (-0.75)</correction>
                <correction>No guarda/actualiza/crea correctamente el archivo.</correction>
            </reference_corrections>
        """
        if isinstance(reference_correction, str):
            return reference_correction
        elif isinstance(reference_correction, dict) and 'corrections' in reference_correction:
            # Format as XML
            corrections = reference_correction['corrections']
            if not corrections:
                return "<reference_corrections>\n</reference_corrections>"
            
            xml_parts = ["<reference_corrections>"]
            for correction in corrections:
                xml_parts.append(f"    <correction>{correction}</correction>")
            xml_parts.append("</reference_corrections>")
            
            return "\n".join(xml_parts)
        else:
            return str(reference_correction)
    
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
    
    def _get_llm_for_metric(self, metric_name: str, metric_config: Dict[str, Any]):
        """
        Get the LLM instance for a specific metric.
        
        If the metric config has 'model_name' and 'provider', use those.
        Otherwise, fall back to the default evaluator LLM.
        """
        # Check if metric has custom LLM config
        if "model_name" in metric_config and "provider" in metric_config:
            provider = metric_config["provider"]
            model_name = metric_config["model_name"]
            temperature = metric_config.get("temperature", self.evaluator_config["temperature"])
            
            logger.info(f"Using custom LLM for {metric_name}: {provider}/{model_name} (temp={temperature})")
            
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                )
            elif provider == "ollama":
                from langchain_ollama.llms import OllamaLLM
                return OllamaLLM(
                    model=model_name,
                    temperature=temperature
                )
            elif provider in ("gemini", "google", "google-genai"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
            else:
                raise ValueError(f"Unsupported provider for {metric_name}: {provider}")
        
        # Use default LLM
        logger.info(f"Using default LLM for {metric_name}: {self.evaluator_config['provider']}/{self.evaluator_config['model_name']}")
        return self.llm
    
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
    
    def _evaluate_single_metric(self, metric_name: str, state: IndividualMetricsState) -> tuple[float, str]:
        """
        Helper method to evaluate a single metric.
        
        Supports three modes:
        1. Python function only: if config has "function" key but no "template", calls the function directly
        2. LLM template only: if config has "template" key but no "function", uses LLM with template
        3. Hybrid (template + function): if config has both "template" AND "function", 
           uses LLM to generate response, then calls function to post-process it
        
        Each metric can optionally specify its own LLM configuration (model_name, provider, temperature)
        which overrides the default evaluator config.
        """
        if metric_name not in self.eval_metric_configs:
            raise ValueError(f"Unknown evaluation metric: {metric_name}")
        
        config = self.eval_metric_configs[metric_name]
        
        # Get auxiliary metrics dictionary
        aux_metrics = state.get("auxiliary_metrics", {})
        
        # Special handling for content_similarity: it computes directly from auxiliary_metrics
        # even if template is specified (template is ignored)
        if metric_name == "content_similarity" and "function" in config:
            function_name = config["function"]
            logger.info(f"Evaluating {metric_name} directly using function: {function_name}")
            
            from .metric_computers import compute_content_similarity
            
            if function_name != "compute_content_similarity":
                raise ValueError(
                    f"Function '{function_name}' is not supported for content_similarity. "
                    f"Must be 'compute_content_similarity'"
                )
            
            # Get LLM for this metric (may have custom config)
            metric_llm = self._get_llm_for_metric(metric_name, config)
            
            # Get content_similarity LLM config from metric config
            content_similarity_llm_config = {
                'model_name': config.get('model_name', self.evaluator_config['model_name']),
                'provider': config.get('provider', self.evaluator_config['provider']),
                'temperature': config.get('temperature', self.evaluator_config['temperature'])
            }
            
            requirements_list = state.get("requirements", [])
            
            score, explanation = compute_content_similarity(
                aux_metrics,
                student_code=state.get("student_code", ""),
                assignment=state.get("assignment", ""),
                requirements=requirements_list,
                llm=metric_llm,
                content_similarity_llm_config=content_similarity_llm_config
            )
            logger.info(f"Completed {metric_name} (Direct computation): score={score}")
            return score, explanation
        
        # Mode 1: Python function only (no template)
        if "function" in config and "template" not in config:
            function_name = config["function"]
            logger.info(f"Evaluating {metric_name} using Python function: {function_name}")
            
            function_map = {
                "compute_completeness": compute_completeness,
                "compute_restraint": compute_restraint,
                "compute_precision": compute_precision,
            }
            
            if function_name not in function_map:
                raise ValueError(
                    f"Unknown function '{function_name}' for metric '{metric_name}'. "
                    f"Available functions: {list(function_map.keys())}"
                )
            
            compute_func = function_map[function_name]
            
            # Special handling for precision which needs additional parameters
            if function_name == "compute_precision":
                precision_llm_config = config.get("precision_llm_config")
                score, explanation = compute_func(
                    aux_metrics,
                    student_code=state.get("student_code", ""),
                    assignment=state.get("assignment", ""),
                    requirements=state.get("requirements", ""),
                    precision_llm_config=precision_llm_config
                )
            else:
                score, explanation = compute_func(aux_metrics)
            
            logger.info(f"Completed {metric_name} (Python function): score={score}")
            return score, explanation
        
        # Mode 2 & 3: Template-based (with or without post-processing function)
        if "template" not in config:
            raise ValueError(
                f"Metric '{metric_name}' must have either 'template' or 'function' in config"
            )
        
        # Check if this metric has custom LLM config
        metric_llm = self._get_llm_for_metric(metric_name, config)
        
        template_name = config["template"]
        # Format requirements as XML
        requirements_list = state.get("requirements", [])
        requirements_xml = self._format_requirements_as_xml(requirements_list)
        
        # Format reference correction (convert from ReferenceCorrection if needed) with <human> tags
        reference_correction_raw = state.get("reference_correction", "")
        human_correction_xml = self._format_reference_correction(reference_correction_raw)
        
        # Format generated correction with <generated> tags
        generated_text = state.get("generated_correction", "")
        if generated_text.strip():
            generated_correction_xml = f"<generated>\n{generated_text}\n</generated>"
        else:
            generated_correction_xml = ""
        
        template = self.jinja_env.get_template(template_name)
        
        # Render the template with all inputs
        rendered = template.render(
            assignment=state.get("assignment", ""),
            requirements=requirements_xml,
            student_code=state.get("student_code", ""),
            human=human_correction_xml,
            generated=generated_correction_xml,
            # Keep old names for backward compatibility
            reference_correction=human_correction_xml,
            generated_correction=generated_correction_xml,
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
        
        raw_response = metric_llm.invoke(messages)
        result_text = self._extract_result_text(raw_response)
        
        # Mode 3: Hybrid - if config has both template AND function, 
        # use function to post-process the LLM response
        # Note: content_similarity is handled earlier and won't reach here
        if "function" in config:
            function_name = config["function"]
            logger.info(f"Post-processing {metric_name} LLM response with function: {function_name}")
            
            # This is for other hybrid functions that post-process LLM responses
            # content_similarity is handled earlier and returns before reaching here
            raise ValueError(
                f"Hybrid mode (template + function) is not currently supported for '{metric_name}'. "
                f"Use either template-only or function-only mode."
            )
        
        # Mode 2: Template only - parse the LLM response normally
        parsed = self._parse_metric_response(result_text, metric_name)
        
        # Convert score from 0-5 scale to 0-1 scale if needed
        score = parsed['score']
        if isinstance(score, int) and score >= 0 and score <= 5:
            score = score / 5.0
        
        logger.info(f"Completed {metric_name} (LLM only): score={score}")
        
        return score, parsed['explanation']
    
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
        reference_text: Union[str, ReferenceCorrection],
        submission: Submission,
        assignment: Optional[str] = None,
        requirements: Optional[Union[List[Requirement], str]] = None,
        metrics_to_evaluate: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all configured metrics using auxiliary metrics via LangGraph.
        
        Args:
            aux_metrics: Dictionary of auxiliary metric outputs
            generated_text: AI-generated correction text
            reference_text: Human reference correction (string or ReferenceCorrection)
            submission: Student submission
            assignment: Optional assignment description
            requirements: Optional requirements/rubric as List[Requirement] or string (for backward compatibility)
            metrics_to_evaluate: Optional list of specific metrics to evaluate.
                                If None, evaluates all configured metrics.
        
        Returns:
            Dictionary with 'scores', 'explanations', 'overall_score', and 'timings'
        """
        logger.info("Evaluating metrics via LangGraph")
        
        # Convert requirements to list if it's a string (backward compatibility)
        if isinstance(requirements, str):
            # Parse legacy string format into simple requirements
            requirements_list = []
            for line in requirements.strip().split('\n'):
                if line.strip():
                    requirements_list.append({
                        "requirement": line.strip(),
                        "function": "unknown",
                        "type": "error_presence"
                    })
        elif requirements is None:
            requirements_list = []
        else:
            requirements_list = requirements
        
        # Prepare initial state
        initial_state: IndividualMetricsState = {
            "assignment": assignment or "",
            "requirements": requirements_list,
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
