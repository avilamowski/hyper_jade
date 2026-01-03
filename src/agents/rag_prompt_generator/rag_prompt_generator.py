"""
RAG-Enhanced Prompt Generator Agent

This module implements a RAG-enhanced prompt generator that uses course theory
to generate better examples and prompts. It follows the same interface as the
standard PromptGeneratorAgent but uses RAG functionality.
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional, Annotated
import logging
import time
import re
import asyncio
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models import PromptType, Requirement, GeneratedPrompt
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from src.config import get_agent_config
from src.agents.utils.reducers import keep_last, concat

# Import LangSmith for tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create a no-op decorator if LangSmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .rag_system import RAGSystem
from .code_generator import CodeExampleGenerator
from .config import USE_RAG
from src.agents.prompt_generator.prompt_generator import PromptType, split_examples

logger = logging.getLogger(__name__)


def format_rag_examples_simple(examples: str) -> str:
    """
    Convert RAG examples (with metadata) to simple format (just code).
    Removes metadata like class names, improvements, theory alignment.
    Preserves the distinction between good and bad examples.
    
    Args:
        examples: RAG-formatted examples string with metadata
    
    Returns:
        Simple formatted examples string matching the standard format (Good example X: / Bad Example X:)
    """
    # Split into individual example blocks by looking for "Good example" or "Bad example" headers
    import re
    
    # Find all example blocks with their headers
    example_pattern = r'((?:Good|Bad) example \d+:.*?)(?=(?:Good|Bad) example \d+:|$)'
    example_blocks = re.findall(example_pattern, examples, re.DOTALL | re.IGNORECASE)
    
    if not example_blocks:
        # Fallback: try to extract code blocks from ```python blocks
        logger.warning("Could not find example blocks, trying to extract code blocks")
        code_blocks = re.findall(r'```python\n(.*?)\n```', examples, re.DOTALL)
        
        if not code_blocks:
            logger.warning("Could not extract code blocks from RAG examples, returning as-is")
            return examples
        
        # Format all as good examples (old behavior as fallback)
        formatted_examples = []
        for i, code in enumerate(code_blocks, 1):
            code = code.strip()
            # Clean escape sequences
            code = code.replace('\\n', '\n')
            code = code.replace('\\t', '\t')
            code = code.replace('\\r', '\r')
            
            lines = code.split('\n')
            indented_lines = []
            for line in lines:
                if line.strip():
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append('')
            indented_code = '\n'.join(indented_lines)
            formatted_examples.append(f"Good example {i}:\n{indented_code}")
        
        return "\n\n".join(formatted_examples)
    
    # Process each example block, preserving Good/Bad distinction
    formatted_examples = []
    good_count = 0
    bad_count = 0
    
    for block in example_blocks:
        # Extract the header to determine if it's good or bad
        is_bad = block.strip().lower().startswith('bad')
        
        # Extract code from this block
        code_match = re.search(r'```python\n(.*?)\n```', block, re.DOTALL)
        if not code_match:
            # Try without language specifier
            code_match = re.search(r'```\n(.*?)\n```', block, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            
            # Clean code: handle escape sequences
            code = code.replace('\\n', '\n')
            code = code.replace('\\t', '\t')
            code = code.replace('\\r', '\r')
            
            # Clean and indent code
            lines = code.split('\n')
            indented_lines = []
            for line in lines:
                if line.strip():
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append('')
            indented_code = '\n'.join(indented_lines)
            
            # Format with appropriate header
            if is_bad:
                bad_count += 1
                formatted_examples.append(f"Bad example {bad_count}:\n{indented_code}")
            else:
                good_count += 1
                formatted_examples.append(f"Good example {good_count}:\n{indented_code}")
    
    logger.info(f"Formatted {good_count} good examples and {bad_count} bad examples from RAG output")
    return "\n\n".join(formatted_examples)


# --- LangGraph Node: RAG-Enhanced Example Generation ---
@traceable(name="rag_generate_examples")
async def rag_example_generation_node(requirement: Requirement, agent_config: dict, llm, rag_system, code_generator, assignment_description: str = "") -> str:
    """
    Node for generating examples using RAG system and course theory.
    Generates both good examples (RAG-enhanced) and bad examples (original).
    Returns the formatted examples as a string.
    """
    try:
        # Generate good examples (RAG-enhanced) and bad examples (original)
        good_examples, bad_examples = await code_generator.generate_enhanced_examples(
            requirement=requirement["requirement"],
            num_examples=3,
            max_theory_results=5,
            dataset="python",  # Default to python for now
            assignment_description=assignment_description
        )
        
        if not good_examples and not bad_examples:
            logger.error("RAG example generation failed: No examples generated")
            return "Error generating examples using RAG system."
        
        # Format GOOD examples (RAG-enhanced) for the prompt
        formatted_good_examples = []
        logger.info(f"Formatting {len(good_examples)} GOOD examples for prompt generation")
        for i, example in enumerate(good_examples, 1):
            code = example.get("code", "")
            description = example.get("description", f"Example {i}")
            improvements = example.get("improvements", [])
            theory_alignment = example.get("theory_alignment", "")
            class_name = example.get("class_name", "Unknown")
            
            formatted_example = f"Good example {i}: {description}\n"
            formatted_example += f"**{class_name}**\n"  # Add class name
            formatted_example += f"```python\n{code}\n```\n"
            
            if improvements:
                # Handle both list and string formats
                if isinstance(improvements, list):
                    improvements_str = '; '.join(str(imp) for imp in improvements if imp)
                else:
                    improvements_str = str(improvements)
                if improvements_str:
                    formatted_example += f"Improvements: {improvements_str}\n"
            
            if theory_alignment:
                # Handle both list and string formats (sometimes it gets parsed as list)
                if isinstance(theory_alignment, list):
                    theory_alignment_str = ' '.join(str(ta) for ta in theory_alignment if ta)
                else:
                    theory_alignment_str = str(theory_alignment).strip()
                if theory_alignment_str:
                    formatted_example += f"Theory alignment: {theory_alignment_str}\n"
            
            formatted_good_examples.append(formatted_example)
        
        # Format BAD examples (original, not RAG-enhanced)
        formatted_bad_examples = []
        logger.info(f"Formatting {len(bad_examples)} BAD examples for prompt generation")
        for i, example in enumerate(bad_examples, 1):
            code = example.get("code", "")
            approach = example.get("approach", f"Does not satisfy the requirement")
            
            formatted_example = f"Bad example {i}:\n"
            formatted_example += f"```python\n{code}\n```\n"
            formatted_example += f"Why it's incorrect: {approach}\n"
            
            formatted_bad_examples.append(formatted_example)
        
        # Combine good and bad examples
        combined_examples = "\n\n".join(formatted_good_examples)
        if formatted_bad_examples:
            combined_examples += "\n\n" + "\n\n".join(formatted_bad_examples)
        
        logger.info(f"Combined {len(good_examples)} good + {len(bad_examples)} bad examples")
        
        return combined_examples
        
    except Exception as e:
        logger.error(f"Error in RAG example generation: {e}")
        return "Error generating examples using RAG system."


# --- LangGraph Node: RAG-Enhanced Prompt Generation ---
@traceable(name="rag_generate_prompts")
def rag_prompt_generation_node(
    requirement: Requirement,
    assignment_description: str,
    examples: str,
    agent_config: dict,
    llm,
) -> str:
    """
    Node for generating a Jinja2 template prompt using requirement, assignment, and RAG-enhanced examples.
    Returns the generated Jinja2 template as a string.
    """
    
    # Extract prompt type from the Requirement object
    prompt_type = requirement["type"]
    requirement_body = requirement["requirement"]

    template_map = agent_config.get("templates", {})
    env = Environment(
        loader=FileSystemLoader("templates"), autoescape=select_autoescape(["jinja"])
    )
    template_file = template_map.get(prompt_type.value, template_map.get("default"))
    template = env.get_template(template_file)
    
    # Split RAG examples into good and bad, same as standard version
    good_examples, bad_examples = split_examples(examples)
    
    # Pass examples as context so LLM understands what examples will be provided,
    # but instruct it to use {{ good_examples }} and {{ bad_examples }} placeholders
    prompt = template.render(
        requirement=requirement_body,
        assignment_description=assignment_description,
        code="{{ code }}",
        good_examples=good_examples,  # Pass as context
        bad_examples=bad_examples,   # Pass as context
    )

    logger.info("[RAG Node] Invoking LLM for template...")
    response = llm.invoke([HumanMessage(content=prompt)])

    # Fix: Extract content properly based on LLM type
    if hasattr(response, "content"):
        jinja_template = response.content.strip()
    else:
        jinja_template = str(response).strip()

    # Debug: Log the raw response
    logger.info(f"[RAG Node] Raw LLM response length: {len(jinja_template)}")
    logger.info(f"[RAG Node] Raw LLM response preview: {jinja_template[:200]}...")

    # Clean up markdown formatting
    if "```jinja" in jinja_template:
        jinja_template = (
            jinja_template.split("```jinja", 1)[-1].split("```", 1)[0].strip()
        )
    elif "```" in jinja_template:
        jinja_template = jinja_template.split("```", 1)[-1].strip()

    # Debug: Log after markdown cleanup
    logger.info(f"[RAG Node] After markdown cleanup length: {len(jinja_template)}")
    logger.info(f"[RAG Node] After markdown cleanup preview: {jinja_template[:200]}...")

    # Clean escaped braces (LLM sometimes generates \{\{ instead of {{)
    jinja_template = jinja_template.replace(r"\{\{", "{{").replace(r"\}\}", "}}")
    
    # Ensure the template has the basic structure
    if "{{ code }}" not in jinja_template and "{{code}}" not in jinja_template:
        jinja_template = jinja_template.replace("{{ student_code }}", "{{ code }}")
        jinja_template = jinja_template.replace("{{code}}", "{{ code }}")
        if "{{ code }}" not in jinja_template:
            jinja_template += "\n\nCode to analyze:\n{{ code }}"
    
    # Ensure the template has {{ good_examples }} and {{ bad_examples }} placeholders
    # If not present, try to add them in appropriate places
    has_good = "{{ good_examples }}" in jinja_template or "{{good_examples}}" in jinja_template
    has_bad = "{{ bad_examples }}" in jinja_template or "{{bad_examples}}" in jinja_template
    
    # Also check for old {{ examples }} format and suggest replacement
    if "{{ examples }}" in jinja_template or "{{examples}}" in jinja_template:
        logger.warning("Template uses old {{ examples }} format. Consider updating to use {{ good_examples }} and {{ bad_examples }} separately.")
    
    # If neither placeholder is present, add them before {{ code }}
    if not has_good and not has_bad:
        examples_placeholder = "{{ good_examples }}\n\n{{ bad_examples }}"
        
        # Check if "Code to analyze:" already exists
        if "Code to analyze:" in jinja_template:
            # Insert examples before the existing "Code to analyze:" line
            jinja_template = jinja_template.replace(
                "Code to analyze:",
                f"{examples_placeholder}\n\nCode to analyze:",
                1  # Only replace first occurrence to avoid duplication
            )
        elif "{{ code }}" in jinja_template:
            # No "Code to analyze:" label exists, add placeholders and label before {{ code }}
            jinja_template = jinja_template.replace(
                "{{ code }}",
                f"{examples_placeholder}\n\nCode to analyze:\n{{ code }}",
                1  # Only replace first occurrence
            )

    # Debug: Log final template
    logger.info(f"[RAG Node] Final template length: {len(jinja_template)}")
    logger.info(f"[RAG Node] Final template preview: {jinja_template[:200]}...")

    return jinja_template


class RAGPromptGeneratorState(TypedDict):
    # All fields are Annotated to allow concurrent writes
    assignment_description: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]
    current_requirement: Annotated[Optional[Requirement], keep_last]
    current_requirement_index: Annotated[Optional[int], keep_last]
    examples: Annotated[Optional[str], keep_last]
    jinja_template: Annotated[Optional[str], keep_last]
    generated_prompts: Annotated[List[GeneratedPrompt], concat]
    # List of {requirement, template, examples, etc.}
    
    # Extra field for additional metadata/data that can be loaded any time
    extra: Annotated[Optional[Dict[str, Any]], keep_last]


class RAGPromptGeneratorAgent:
    """
    RAG-Enhanced Agent that orchestrates parallel prompt generation using LangGraph with Map-Reduce pattern.
    Uses RAG system to generate better examples based on course theory.
    Supports both individual usage (single requirement) and batch processing (multiple requirements).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = get_agent_config(config, "prompt_generator")
        self.llm = self._setup_llm()
        self.rag_system = None
        self.code_generator = None
        self.graph = None

    def _setup_llm(self):
        if self.agent_config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.agent_config.get("model_name", "gpt-4"),
                temperature=self.agent_config.get("temperature", 0.1),
            )
        else:
            return OllamaLLM(
                model=self.agent_config.get("model_name", "qwen2.5:7b"),
                temperature=self.agent_config.get("temperature", 0.1),
            )

    async def initialize(self):
        """Initialize RAG system and code generator."""
        try:
            logger.info("🧠 Initializing RAG-Enhanced Prompt Generator")
            logger.info("=" * 50)
            
            self.rag_system = RAGSystem()
            await self.rag_system.initialize()
            
            self.code_generator = CodeExampleGenerator(self.rag_system)
            await self.code_generator.initialize()
            
            # Build the graph after initialization
            self.graph = self._build_graph()
            
            logger.info("✅ RAG Prompt Generator initialized successfully")
            logger.info("🔧 RAG Mode: ENABLED")
            logger.info("📚 Course theory integration: ACTIVE")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Failed to initialize RAG Prompt Generator: {e}")
            raise

    # -------------------------- LangGraph Nodes ------------------------------ #

    def _create_requirement_processor(self, requirement: Requirement, index: int):
        """Create a node function for processing a specific requirement with RAG enhancement"""
        
        async def process_requirement_node(state: RAGPromptGeneratorState) -> RAGPromptGeneratorState:
            """Process a single requirement with RAG enhancement - generate examples and prompt template"""
            assignment_description = state.get("assignment_description", "")
            
            logger.info(f"[RAG Process {index + 1}] Processing requirement: {requirement['requirement'][:50]}...")

            try:
                # Generate RAG-enhanced examples
                examples = await rag_example_generation_node(
                    requirement, self.agent_config, self.llm, self.rag_system, self.code_generator, assignment_description
                )
                
                # Convert type string to PromptType if needed
                req_type = requirement.get("type")
                if isinstance(req_type, str):
                    for pt in PromptType:
                        if pt.value == req_type:
                            req_type = pt
                            break
                    requirement["type"] = req_type
                
                # Generate Jinja2 template using RAG-enhanced examples
                jinja_template = rag_prompt_generation_node(
                    requirement,
                    assignment_description,
                    examples,
                    self.agent_config,
                    self.llm,
                )
                
                # Create result
                generated_prompt = {
                    "requirement": requirement,
                    "examples": examples,
                    "jinja_template": jinja_template,
                    "index": index,
                    "extra": {
                        "rag_enhanced": True,
                        "theory_sources_used": True,
                        "generation_method": "rag"
                    }
                }
                
                logger.info(f"[RAG Process {index + 1}] Completed successfully")
                
                return {
                    "generated_prompts": [generated_prompt]
                }
                
            except Exception as e:
                logger.error(f"[RAG Process {index + 1}] Error processing requirement with RAG: {e}")
                # Fallback to basic template
                generated_prompt = {
                    "requirement": requirement,
                    "examples": "Error generating RAG-enhanced examples",
                    "jinja_template": "Error: Could not generate template",
                    "index": index,
                    "extra": {
                        "rag_enhanced": False,
                        "theory_sources_used": False,
                        "generation_method": "fallback",
                        "error": str(e)
                    }
                }
                
                return {
                    "generated_prompts": [generated_prompt]
                }
        
        return process_requirement_node

    def _create_dynamic_graph(self, requirements: List[Requirement]):
        """Create dynamic graph with nodes for each requirement, enabling parallel processing"""
        # Create a fresh graph for this batch
        dynamic_graph = StateGraph(RAGPromptGeneratorState)

        # Add a node for each requirement
        for i, requirement in enumerate(requirements):
            node_name = f"rag_process_requirement_{i}"
            node_function = self._create_requirement_processor(requirement, i)

            dynamic_graph.add_node(node_name, node_function)

            # Connect START to this requirement node
            dynamic_graph.add_edge(START, node_name)

            # Connect this requirement node directly to END
            dynamic_graph.add_edge(node_name, END)

        # Compile the graph with all dynamic connections
        self.graph = dynamic_graph.compile()
        logger.info(f"Created dynamic RAG graph with {len(requirements)} parallel nodes")

    def _build_graph(self):
        """Build a simple base LangGraph (used before requirements are known)"""
        workflow = StateGraph(RAGPromptGeneratorState)
        # Simple pass-through for initialization
        workflow.add_edge(START, END)
        return workflow.compile()

    @traceable(name="RAGPromptGeneratorAgent.generate_prompts_batch")
    async def generate_prompts_batch(
        self, requirements: List[Requirement], assignment_description: str
    ) -> List[GeneratedPrompt]:
        """
        Generate prompts for multiple requirements using RAG enhancement with parallel processing.
        This is the main method that should be called from the runner.
        """
        if not self.rag_system or not self.code_generator:
            raise Exception("RAG system not initialized. Call initialize() first.")

        logger.info(f"Generating {len(requirements)} RAG-enhanced prompts in parallel...")

        # Create dynamic graph for parallel processing
        self._create_dynamic_graph(requirements)

        # Create state for batch processing
        state: RAGPromptGeneratorState = {
            "assignment_description": assignment_description,
            "requirements": requirements,
            "generated_prompts": [],
            "extra": {},
            "current_requirement": None,
            "current_requirement_index": None,
            "examples": None,
            "jinja_template": None,
        }

        # Run the graph with dynamic nodes (all requirements process in parallel)
        result_state = await self.graph.ainvoke(state)

        # Extract results and sort by index to maintain order
        results: List[GeneratedPrompt] = sorted(
            result_state["generated_prompts"], 
            key=lambda x: x["index"]
        )

        logger.info(f"Generated {len(results)} RAG-enhanced prompts in parallel")
        return results

    def as_graph_node(self):
        """
        Expose the agent as a LangGraph node for use in external graphs.
        Returns a function that takes a state dict and returns a state dict with the generated templates.
        Note: This creates a synchronous wrapper around the async batch processing.
        """
        def node(state):
            # If state has requirements list, process in batch
            if "requirements" in state and state["requirements"]:
                assignment_description = state.get("assignment_description", "")
                requirements = state["requirements"]
                
                # Use async batch processing with parallel execution
                results = asyncio.run(self.generate_prompts_batch(requirements, assignment_description))
                
                state["generated_prompts"] = results
                return state
            else:
                # Single requirement processing (shouldn't happen, but fallback)
                logger.warning("as_graph_node called without requirements list")
                return state

        return node
