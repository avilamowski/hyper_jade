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
import logging
import os

from dotenv import load_dotenv, dotenv_values
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph
from langgraph.graph import StateGraph, END, START

from src.config import get_agent_config
from src.agents.utils.reducers import keep_last, concat
from src.agents.prompt_generator.prompt_generator import split_examples
from src.models import Requirement, GeneratedPrompt, Submission, Correction

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Load .env ASAP and keep values in-memory
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
DOTENV = dotenv_values()

# Import GroupFunctionsAgent lazily to avoid circular imports
def _get_group_functions_agent():
    from src.agents.group_functions import GroupFunctionsAgent
    return GroupFunctionsAgent


def _get_linter_correction_agent():
    from src.agents.code_corrector.linter_correction_agent import LinterCorrectionAgent
    return LinterCorrectionAgent


# --------------------------------------------------------------------------- #
# LangGraph State
# --------------------------------------------------------------------------- #
class CodeCorrectorState(TypedDict):
    # Input data
    assignment_description: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]  
    generated_prompts: Annotated[List[GeneratedPrompt], keep_last] 
    submission: Annotated[Submission, keep_last]
    grouped_code: Annotated[Optional[List[Dict[str, Any]]], keep_last]  # Optional grouped functions
    
    # Output - use Annotated to allow multiple concurrent writes
    corrections: Annotated[List[Correction], concat]
    
    # Extra field for additional metadata/data that can be loaded any time
    extra: Annotated[Optional[Dict[str, Any]], keep_last]


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
        
        # Initialize group_functions if enabled
        group_functions_config = get_agent_config(self.config, 'group_functions')
        self.group_functions_enabled = group_functions_config.get('enabled', False)
        self.group_functions_agent = None
        
        if self.group_functions_enabled:
            GroupFunctionsAgent = _get_group_functions_agent()
            self.group_functions_agent = GroupFunctionsAgent(self.config)
        
        # Initialize linter if enabled
        linter_config = self.agent_config.get('linter', {})
        self.linter_enabled = linter_config.get('enabled', False)
        self.linter_agent = None
        if self.linter_enabled:
            LinterCorrectionAgent = _get_linter_correction_agent()
            self.linter_agent = LinterCorrectionAgent(self.config)
            logger.info("Linter correction agent initialized")
        
        self.graph = self._build_graph()

    def _setup_llm(self):
        provider = str(self.agent_config.get("provider", "openai")).lower().strip()
        model_name = self.agent_config.get("model_name")
        
        if not model_name:
            raise ValueError(f"model_name is required in code_corrector configuration")
        
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

        if provider in ("gemini", "google", "google-genai"):
            api_key = self.agent_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key
            )

        return OllamaLLM(model=model_name, temperature=temperature)

    # -------------------------- Utilities ----------------------------------- #
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        return re.sub(r"```[a-zA-Z]*\n|```", "", text).strip()

    def _call_llm(self, rendered_prompt: str) -> str:
        def _extract_text(raw):
            # Check if raw has .content attribute
            if hasattr(raw, "content"):
                content = raw.content
                
                # Handle Gemini's response format: content can be a list of parts
                # Each part has {'type': 'text', 'text': '...', 'extras': {...}}
                if isinstance(content, list):
                    # Extract text from all parts and concatenate
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                        elif isinstance(part, dict) and part.get('type') == 'text':
                            # Sometimes the text might be in a different structure
                            text_parts.append(str(part.get('text', '')))
                        elif hasattr(part, 'text'):
                            text_parts.append(part.text)
                        else:
                            # Fallback: convert to string
                            text_parts.append(str(part))
                    return ''.join(text_parts)
                
                # If content is already a string, return it
                if isinstance(content, str):
                    return content
                
                # Otherwise try to convert to string
                return str(content)
            
            # dict-like OpenAI shape
            if isinstance(raw, dict):
                choices = raw.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    return (first.get("message", {}) or {}).get("content") or first.get("text") or str(raw)
                return raw.get("text") or str(raw)
            
            # fallback to str
            return str(raw)

        try:
            # Both ChatOpenAI and ChatGoogleGenerativeAI use the same message format
            if isinstance(self.llm, (ChatOpenAI, ChatGoogleGenerativeAI)):
                resp = self.llm.invoke([HumanMessage(content=rendered_prompt)])
                out_text = _extract_text(resp)
            else:
                resp = self.llm.invoke(rendered_prompt)
                out_text = _extract_text(resp)

            # Ensure out_text is a string before passing to _strip_code_fences
            if out_text is None:
                out_text = ""
            elif not isinstance(out_text, str):
                out_text = str(out_text)

            return self._strip_code_fences(out_text)
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {e}")

    # -------------------------- LangGraph Nodes ----------------------------- #

    def _create_correction_processor(self, generated_prompt: GeneratedPrompt, index: int):
        """Create a node function for processing a specific generated prompt"""

        def process_correction_node(state: CodeCorrectorState) -> CodeCorrectorState:
            """Process a single correction - render template and invoke LLM"""
            submission = state["submission"]
            grouped_code = state.get("grouped_code", [])
            
            # Determine which code to use
            if grouped_code:
                # Try to find grouped code for this requirement's function
                requirement_function = generated_prompt["requirement"].get("function", "")
                # Clean function name: "validar(texto)" -> "validar"
                requirement_function_clean = requirement_function.split('(')[0].strip() if requirement_function else ""
                matched_group = None
                
                for group in grouped_code:
                    if group.get("function_name") == requirement_function_clean:
                        matched_group = group
                        break
                
                if matched_group:
                    student_code = matched_group["code"]
                else:
                    # Fallback to full code if no match found
                    student_code = submission["code"]
            else:
                student_code = submission["code"]

            # Check if template already has examples injected (no placeholders)
            # If it does, we only need to inject the student code
            template_str = generated_prompt["jinja_template"]
            has_placeholders = "{{ correct_examples }}" in template_str or "{{ erroneous_examples }}" in template_str
            
            if has_placeholders:
                # Template has placeholders, need to render with examples
                examples_str = generated_prompt.get("examples", "")
                if examples_str:
                    correct_examples, erroneous_examples = split_examples(examples_str)
                    rendered_prompt = Template(template_str).render(
                        code=student_code,
                        correct_examples=correct_examples,
                        erroneous_examples=erroneous_examples,
                    )
                else:
                    # No examples available, render with empty examples
                    rendered_prompt = Template(template_str).render(
                        code=student_code,
                        correct_examples="",
                        erroneous_examples="",
                    )
            else:
                # Template already has examples injected, only need to inject student code
                rendered_prompt = Template(template_str).render(
                    code=student_code,
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
        assignment_description: str = "",
        submission_name: Optional[str] = None
    ) -> List[Correction]:
        """
        Generate multiple corrections in parallel from a list of generated prompts.
        Returns a list of Correction objects.

        Args:
            generated_prompts: List of GeneratedPrompt objects
            submission: The Submission object with student code
            assignment_description: Optional assignment description
            submission_name: Optional submission name for grouping (e.g., "alu1")
        """
        # Apply function grouping if enabled
        grouped_code = None
        if self.group_functions_enabled and self.group_functions_agent:
            # Extract unique function names from requirements
            function_names = set()
            for gp in generated_prompts:
                func_name = gp["requirement"].get("function", "")
                # Parse function name from "function_name(params)" format
                if func_name:
                    # Extract just the function name, removing parameters
                    func_name_clean = func_name.split('(')[0].strip()
                    if func_name_clean:
                        function_names.add(func_name_clean)
            
            # Only group if we have function names
            if function_names:
                submissions_for_grouping = [{
                    "name": submission_name or "submission",
                    "code": submission["code"]
                }]
                
                # Use group_code_by_functions (already @traceable)
                grouped_code = self.group_functions_agent.group_code_by_functions(
                    function_names=list(function_names),
                    submissions=submissions_for_grouping
                )
                
                logger.info(f"Grouped {len(grouped_code)} function segments for {len(function_names)} functions")
        
        # Create dynamic graph for this batch
        self._create_dynamic_graph(generated_prompts)

        # Create state for batch processing
        state: CodeCorrectorState = {
            "assignment_description": assignment_description,
            "requirements": [gp["requirement"] for gp in generated_prompts],
            "generated_prompts": generated_prompts,
            "submission": submission,
            "grouped_code": grouped_code,
            "corrections": [],
            "extra": {},
        }

        # Run the graph with dynamic nodes
        result_state = self.graph.invoke(state)

        # Extract results
        results: List[Correction] = result_state["corrections"]
        
        # Add linter corrections if enabled
        if self.linter_enabled and self.linter_agent:
            try:
                linter_corrections = self.linter_agent.generate_linter_corrections(
                    code=submission["code"]
                )
                results.extend(linter_corrections)
                logger.info(f"Added {len(linter_corrections)} linter corrections")
            except Exception as e:
                logger.error(f"Linter correction failed: {e}")

        return results

    @property
    def compiled_graph(self):
        """Access to the compiled LangGraph for external use."""
        return self.graph
