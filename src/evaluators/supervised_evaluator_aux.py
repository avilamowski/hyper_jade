"""
Auxiliary Metrics Evaluator (LangGraph Parallel Version)

Computes auxiliary metrics (MATCH, MISSING, EXTRA) using configurable strategies.
Default strategy runs metrics in parallel using LangGraph.
Alternative strategies (per_correction, dependent) provide more robust classification.
"""

from __future__ import annotations
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from ..config import load_config
from ..models.models import Correction, Submission, Requirement, ReferenceCorrection
from ..agents.utils.reducers import keep_last, merge_dicts

logger = logging.getLogger(__name__)


class AuxiliaryMetricsState(TypedDict):
    """State for auxiliary metrics computation"""
    # Inputs
    assignment: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]
    student_code: Annotated[str, keep_last]
    reference_correction: Annotated[Union[str, ReferenceCorrection], keep_last]
    generated_correction: Annotated[str, keep_last]
    
    # Outputs - all metrics stored in a single dictionary
    auxiliary_metrics: Annotated[Dict[str, str], merge_dicts]
    
    # Metadata
    timings: Annotated[Optional[Dict[str, float]], merge_dicts]


class AuxiliaryMetricsEvaluator:
    """
    Evaluator for computing auxiliary metrics using configurable strategies.
    
    Strategies:
    - 'independent' (default): Runs MATCH, MISSING, EXTRA in parallel using LangGraph
    - 'per_correction': Classifies each correction individually, then aggregates
    - 'dependent': Runs MATCH first, injects results into MISSING/EXTRA prompts
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
        self.all_aux_metric_configs = supervised_cfg.get("auxiliary_metrics", {})
        
        if not self.all_aux_metric_configs:
            raise ValueError(
                "No auxiliary_metrics found in config. "
                "Expected format: supervised_evaluator.auxiliary_metrics with strategy definitions"
            )
        
        # Load strategy from config (default: independent)
        self.strategy_name = supervised_cfg.get("aux_metrics_strategy", "independent")
        
        # Get templates for the selected strategy
        self.aux_metric_configs = self.all_aux_metric_configs.get(self.strategy_name, {})
        if not self.aux_metric_configs:
            raise ValueError(
                f"No templates found for strategy '{self.strategy_name}'. "
                f"Available strategies: {list(self.all_aux_metric_configs.keys())}"
            )
        
        self.strategy = self._create_strategy(self.strategy_name)
        logger.info(f"AuxiliaryMetricsEvaluator using strategy: {self.strategy_name}")
        
        # Build the graph (used by independent strategy)
        if self.strategy_name == "independent":
            self.graph = self._build_graph()
        else:
            self.graph = None
    
    def _create_strategy(self, name: str):
        """Create a strategy instance by name."""
        from .aux_metrics_strategies import get_strategy
        return get_strategy(name, self)
    
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
        Convert ReferenceCorrection to XML format using <human> tags.
        Each correction is wrapped in its own <human> tag.
        
        Args:
            reference_correction: Either a string or ReferenceCorrection dict
            
        Returns:
            XML-formatted string with <human> tags
        """
        if isinstance(reference_correction, str):
            # Wrap plain text in <human> tag
            if not reference_correction.strip():
                return ""
            return f"<human>\n{reference_correction}\n</human>"
        elif isinstance(reference_correction, dict) and 'corrections' in reference_correction:
            # Format as XML with each correction in its own <human> tag
            corrections = reference_correction['corrections']
            if not corrections:
                return ""
            
            wrapped_corrections = []
            for correction in corrections:
                if correction.strip():
                    wrapped_corrections.append(f"<human>\n{correction.strip()}\n</human>")
            
            return "\n\n".join(wrapped_corrections)
        else:
            return f"<human>\n{str(reference_correction)}\n</human>"
    
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
        elif provider in ("gemini", "google", "google-genai"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.evaluator_config["temperature"],
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _extract_result_text(self, raw_response) -> str:
        """
        Extract text content from LLM response.
        
        If the response contains <OUTPUT>...</OUTPUT> tags, only the content
        within those tags is extracted and stored. This allows the LLM to output
        additional analysis/context that is visible in logs but not stored in state.
        
        If no <OUTPUT> tags are found, the full response is returned.
        """
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
        
        text = text.strip()
        
        # Check for <OUTPUT> tags and extract only that content
        output_match = re.search(
            r'<OUTPUT>(.*?)</OUTPUT>',
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if output_match:
            # Extract only the content within <OUTPUT> tags
            extracted = output_match.group(1).strip()
            logger.debug(f"Extracted OUTPUT section ({len(extracted)} chars) from full response ({len(text)} chars)")
            return extracted
        else:
            # No <OUTPUT> tags found, return full text
            return text
    
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
        
        # Format requirements as XML
        requirements_list = state.get("requirements", [])
        requirements_xml = self._format_requirements_as_xml(requirements_list)
        
        # Format reference correction (convert from ReferenceCorrection if needed) with <human> tags
        reference_correction_raw = state.get("reference_correction", "")
        human_correction_xml = self._format_reference_correction(reference_correction_raw)
        
        # Format generated correction with <generated> tags
        # Each correction should be wrapped in its own <generated> tag
        generated_text = state.get("generated_correction", "")
        if generated_text.strip():
            # Split by double newlines to separate individual corrections
            # Format: "## function_name\n<RESULT>...</RESULT>"
            corrections = generated_text.strip().split("\n\n")
            wrapped_corrections = []
            for correction in corrections:
                if correction.strip():
                    wrapped_corrections.append(f"<generated>\n{correction.strip()}\n</generated>")
            generated_correction_xml = "\n\n".join(wrapped_corrections)
        else:
            generated_correction_xml = ""
        
        # Render the template
        rendered = template.render(
            assignment=state.get("assignment", ""),
            requirements=Markup(requirements_xml),
            student_code=state.get("student_code", ""),
            human=Markup(human_correction_xml),
            generated=Markup(generated_correction_xml),
            # Keep old names for backward compatibility
            reference_correction=Markup(human_correction_xml),
            generated_correction=Markup(generated_correction_xml)
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
        reference_text: Union[str, ReferenceCorrection],
        submission: Submission,
        assignment: Optional[str] = None,
        requirements: Optional[Union[List[Requirement], str]] = None,
        metrics_to_compute: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Compute all configured auxiliary metrics.
        
        Uses the configured strategy (independent, per_correction, or dependent).
        
        Args:
            generated_text: AI-generated correction text
            reference_text: Human reference correction (string or ReferenceCorrection)
            submission: Student submission
            assignment: Optional assignment description
            requirements: Optional requirements/rubric as List[Requirement] or string (for backward compatibility)
            metrics_to_compute: Optional list of specific metrics to compute.
                              If None, computes all configured metrics.
        
        Returns:
            Dictionary mapping metric names to their plain text outputs
        """
        logger.info(f"Computing auxiliary metrics via strategy: {self.strategy_name}")
        
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
        
        # Use non-independent strategies directly
        if self.strategy_name != "independent":
            t0 = time.time()
            result = self.strategy.compute(
                generated_text=generated_text,
                reference_text=reference_text,
                student_code=submission.get('code', ''),
                assignment=assignment or "",
                requirements=requirements_list
            )
            elapsed = time.time() - t0
            logger.info(f"Strategy {self.strategy_name} completed in {elapsed:.2f}s")
            return result
        
        # Independent strategy: use LangGraph
        # Prepare initial state
        initial_state: AuxiliaryMetricsState = {
            "assignment": assignment or "",
            "requirements": requirements_list,
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
