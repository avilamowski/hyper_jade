#!/usr/bin/env python3
"""
Code Correction Agent (LangGraph Map-Reduce Version)

This module implements a parallelized code corrector using LangGraph with Map-Reduce pattern:
1. Maps over each generated_prompt to create correction tasks
2. Processes each correction in parallel (template rendering + LLM analysis)
3. Reduces/collects all corrections into the final state

Supports both individual usage (single prompt) and batch processing (multiple prompts).
All I/O operations (file reading/writing) are handled by the caller (run_code_corrector.py).
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional, Annotated
import re

from dotenv import load_dotenv, dotenv_values
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage

# LangGraph
from langgraph.graph import StateGraph, END, START

from src.config import get_agent_config
from src.agents.utils.reducers import keep_last, concat
from src.models import Requirement, GeneratedPrompt, Submission, Correction

# --------------------------------------------------------------------------- #
# Load .env ASAP and keep values in-memory
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
DOTENV = dotenv_values()
    # --------------------------------------------------------------------------- #
# LangGraph State
# --------------------------------------------------------------------------- #
class CodeCorrectorState(TypedDict):
    # Input data
    assignment_description: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]  
    generated_prompts: Annotated[List[GeneratedPrompt], keep_last] 
    submission: Annotated[Submission, keep_last]
    
    # Output - use Annotated to allow multiple concurrent writes
    corrections: Annotated[List[Correction], concat]


# --------------------------------------------------------------------------- #
# CodeCorrectorAgent
# --------------------------------------------------------------------------- #
class CodeCorrectorAgent:
    """
    Agent that orchestrates parallel code correction using LangGraph with Map-Reduce pattern.
    Supports both individual usage (single prompt) and batch processing (multiple prompts).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.agent_config = get_agent_config(self.config, "code_corrector")
        self.llm = self._setup_llm()
        self.graph = self._build_graph()

    def _setup_llm(self):
        provider = str(self.agent_config.get("provider", "openai")).lower().strip()
        model_name = self.agent_config.get("model_name", "gpt-4o-mini")
        temperature = float(self.agent_config.get("temperature", 0.1))

        if provider == "openai":
            # Prefer explicit api_key from config, else pull from .env dict
            api_key = self.agent_config.get("api_key") or DOTENV.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY missing from .env and no api_key provided in config."
                )
            return ChatOpenAI(
                model=model_name, temperature=temperature, api_key=api_key
            )

        return OllamaLLM(model=model_name, temperature=temperature)

    # -------------------------- Utilities ----------------------------------- #
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        return re.sub(r"```[a-zA-Z]*\n|```", "", text).strip()

    def _call_llm(self, rendered_prompt: str) -> str:
        try:
            if isinstance(self.llm, ChatOpenAI):
                resp = self.llm.invoke([HumanMessage(content=rendered_prompt)])
                out = getattr(resp, "content", str(resp))
            else:
                out = self.llm.invoke(rendered_prompt)
            return self._strip_code_fences(str(out))
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")

    # -------------------------- LangGraph Nodes ----------------------------- #

    def _create_correction_processor(self, generated_prompt: GeneratedPrompt, index: int):
        """Create a node function for processing a specific generated prompt"""

        def process_correction_node(state: CodeCorrectorState) -> CodeCorrectorState:
            """Process a single correction - render template and invoke LLM"""
            submission = state["submission"]
            student_code = submission["code"]

            # Render the Jinja template with the student code
            rendered_prompt = Template(generated_prompt["jinja_template"]).render(
                code=student_code
            )

            # Call LLM for analysis
            analysis = self._call_llm(rendered_prompt)

            # Store correction result
            correction: Correction = {
                "requirement": generated_prompt["requirement"],
                "result": analysis,
            }

            # Use the aggregation function for concurrent writes
            state["corrections"] = [correction]

            return state

        return process_correction_node

    def _create_dynamic_graph(self, generated_prompts: List[GeneratedPrompt]):
        """Create dynamic graph with nodes for each generated prompt"""
        # Create a fresh graph for this batch
        dynamic_graph = self._build_base_graph(with_direct_edge=False)

        # Add a node for each generated prompt
        for i, generated_prompt in enumerate(generated_prompts):
            node_name = f"process_correction_{i}"
            node_function = self._create_correction_processor(generated_prompt, i)

            dynamic_graph.add_node(node_name, node_function)

            # Connect START to this correction node
            dynamic_graph.add_edge(START, node_name)

            # Connect this correction node directly to END
            dynamic_graph.add_edge(node_name, END)

        # Compile the graph with all dynamic connections
        self.graph = dynamic_graph.compile()

    # -------------------------- Graph Builder ------------------------------- #

    def _build_base_graph(self, with_direct_edge: bool = True):
        """Build the base LangGraph structure without dynamic nodes"""
        graph = StateGraph(CodeCorrectorState)

        # Only add direct edge if we're not going to have dynamic nodes
        if with_direct_edge:
            graph.add_edge(START, END)

        return graph

    def _build_graph(self):
        """Build a LangGraph with parallel processing capabilities"""
        # For single correction, use simple graph
        return self._build_base_graph().compile()

    # -------------------------- Public API ---------------------------------- #

    def correct_code(
        self, 
        generated_prompt: GeneratedPrompt, 
        submission: Submission,
        assignment_description: str = ""
    ) -> Correction:
        """
        Generate a single correction from a generated prompt and submission.
        Returns a Correction object.

        Args:
            generated_prompt: The GeneratedPrompt object with requirement and jinja_template
            submission: The Submission object with student code
            assignment_description: Optional assignment description
        """
        # For single correction, use batch processing with one item
        results = self.correct_code_batch(
            [generated_prompt], submission, assignment_description
        )

        return results[0]

    def correct_code_batch(
        self, 
        generated_prompts: List[GeneratedPrompt], 
        submission: Submission,
        assignment_description: str = ""
    ) -> List[Correction]:
        """
        Generate multiple corrections in parallel from a list of generated prompts.
        Returns a list of Correction objects.

        Args:
            generated_prompts: List of GeneratedPrompt objects
            submission: The Submission object with student code
            assignment_description: Optional assignment description
        """
        # Create dynamic graph for this batch
        self._create_dynamic_graph(generated_prompts)

        # Create state for batch processing
        state: CodeCorrectorState = {
            "assignment_description": assignment_description,
            "requirements": [gp["requirement"] for gp in generated_prompts],
            "generated_prompts": generated_prompts,
            "submission": submission,
            "corrections": [],
        }

        # Run the graph with dynamic nodes
        result_state = self.graph.invoke(state)

        # Extract results
        results: List[Correction] = result_state["corrections"]

        return results

    @property
    def compiled_graph(self):
        """Access to the compiled LangGraph for external use."""
        return self.graph
