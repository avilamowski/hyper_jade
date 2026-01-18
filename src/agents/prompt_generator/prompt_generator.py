#!/usr/bin/env python3

"""
Prompt Generator Agent (LangGraph Parallel Version)

This module implements a parallelized prompt generator using LangGraph with Map-Reduce pattern:
1. prepare_batch: Prepares state structures for parallel processing (no I/O)
2. process_requirement: Processes individual requirements in parallel (examples + prompt generation)
3. collect_results: Collects and organizes all generated prompts

Supports both individual usage (single requirement) and batch processing (multiple requirements).
All I/O operations (file reading/writing) are handled by the caller (run_prompt_generator.py).
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional, Annotated
import logging
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.agents.requirement_generator.requirement_generator import (
    RequirementGeneratorState,
)
from src.models import PromptType, Requirement, GeneratedPrompt
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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

logger = logging.getLogger(__name__)


def split_examples(examples: str) -> tuple[str, str]:
    """
    Split examples string into correct_examples and erroneous_examples using XML parsing.
    Expects XML format with <correct> and <erroneous> tags.
    
    Returns:
        tuple: (correct_examples, erroneous_examples) as strings
    """
    import re
    
    # Normalize line endings
    examples = examples.replace('\r\n', '\n').replace('\r', '\n')
    
    # Try to extract <correct> and <erroneous> sections using XML tags
    correct_match = re.search(r'<correct>(.*?)</correct>', examples, re.DOTALL | re.IGNORECASE)
    erroneous_match = re.search(r'<erroneous>(.*?)</erroneous>', examples, re.DOTALL | re.IGNORECASE)
    
    if correct_match and erroneous_match:
        # Successfully found XML tags
        correct_section = correct_match.group(1).strip()
        erroneous_section = erroneous_match.group(1).strip()
        
        # Parse individual examples from each section
        correct_examples = _parse_xml_examples(correct_section, "correct")
        erroneous_examples = _parse_xml_examples(erroneous_section, "erroneous")
        
        return correct_examples, erroneous_examples
    else:
        # Fallback: try old text-based format for backward compatibility
        logger.warning("XML tags not found, falling back to text-based parsing")
        return _split_examples_legacy(examples)


def _parse_xml_examples(section: str, example_type: str) -> str:
    """
    Parse individual <example> blocks from a section and format them.
    
    Args:
        section: XML content inside <correct> or <erroneous> tags
        example_type: "correct" or "erroneous"
    
    Returns:
        Formatted string with examples
    """
    import re
    
    # Find all <example> blocks
    example_pattern = r'<example>(.*?)</example>'
    example_blocks = re.findall(example_pattern, section, re.DOTALL | re.IGNORECASE)
    
    formatted_examples = []
    for i, block in enumerate(example_blocks, 1):
        # Extract code from the example block
        code_match = re.search(r'<code>(.*?)</code>', block, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1).strip()
            
            # Format the example with proper indentation
            lines = code.split('\n')
            indented_lines = []
            for line in lines:
                if line.strip():
                    indented_lines.append('    ' + line)
                else:
                    indented_lines.append('')
            indented_code = '\n'.join(indented_lines)
            
            # Format with appropriate header
            if example_type == "correct":
                formatted_examples.append(f"Correct example {i}:\n{indented_code}")
            else:
                formatted_examples.append(f"Erroneous Example {i}:\n{indented_code}")
    
    return "\n\n".join(formatted_examples)


def _split_examples_legacy(examples: str) -> tuple[str, str]:
    """
    Legacy text-based splitting for backward compatibility.
    Looks for "Correct example" and "Erroneous Example" patterns.
    
    Returns:
        tuple: (correct_examples, erroneous_examples) as strings
    """
    import re
    
    # Try to find the split point between correct and erroneous examples
    # Look for "Erroneous Example" pattern (case insensitive)
    erroneous_pattern = re.compile(r'(?i)(?:^|\n)\s*Erroneous\s+Example', re.MULTILINE)
    
    match = erroneous_pattern.search(examples)
    if match:
        split_pos = match.start()
        correct_examples = examples[:split_pos].strip()
        erroneous_examples = examples[split_pos:].strip()
    else:
        # If no clear split found, try line-by-line search
        lines = examples.split('\n')
        split_line = None
        for i, line in enumerate(lines):
            if re.match(r'^\s*Erroneous\s+Example', line, re.IGNORECASE):
                split_line = i
                break
        
        if split_line:
            correct_examples = '\n'.join(lines[:split_line]).strip()
            erroneous_examples = '\n'.join(lines[split_line:]).strip()
        else:
            # Fallback: return all as correct_examples and empty erroneous_examples
            logger.warning("Could not find clear split between correct and erroneous examples. Using all as correct_examples.")
            correct_examples = examples
            erroneous_examples = ""
    
    return correct_examples, erroneous_examples



# --- LangGraph Node: Example Generation ---
@traceable(name="example_generation")
def example_generation_node(requirement: Requirement, assignment_description: str, theory_summary: str, agent_config: dict, llm) -> str:
    """
    Node for generating examples from a requirement.
    Returns the generated examples as a string.
    """

    template_map = agent_config.get("templates", {})
    env = Environment(
        loader=FileSystemLoader("templates"), autoescape=select_autoescape(["jinja"])
    )
    
    # Get requirement type to select appropriate examples template
    requirement_type = requirement.get("type")
    if hasattr(requirement_type, 'value'):
        requirement_type_str = requirement_type.value
    else:
        requirement_type_str = str(requirement_type) if requirement_type else None
    
    # Select examples template based on type (with fallback to default)
    examples_template_key = f"examples_{requirement_type_str}" if requirement_type_str else "examples"
    examples_template_file = template_map.get(examples_template_key, template_map.get("examples", "examples.jinja"))
    examples_template = env.get_template(examples_template_file)
    
    # Support separate quantities for correct and erroneous examples
    # Default to example_quantity for backward compatibility
    example_quantity = agent_config.get("example_quantity", 3)
    correct_example_quantity = agent_config.get("correct_example_quantity", example_quantity)
    erroneous_example_quantity = agent_config.get("erroneous_example_quantity", example_quantity)
    
    # Use the requirement text from the Requirement object
    requirement_text = requirement["requirement"]
    
    examples_prompt = examples_template.render(
        requirement=requirement_text,
        example_quantity=example_quantity,  # Keep for backward compatibility
        correct_example_quantity=correct_example_quantity,
        erroneous_example_quantity=erroneous_example_quantity,
        assignment_description=assignment_description,
        theory_summary=theory_summary
    )

    logger.info("[Node] Invoking LLM for examples...")
    examples_response = llm.invoke([HumanMessage(content=examples_prompt)])

    # Fix: Extract content properly based on LLM type
    if hasattr(examples_response, "content"):
        content = examples_response.content
        # Handle case where content is a list (e.g., Gemini)
        if isinstance(content, list):
            # Extract text from each part in the list, ignoring extras
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    # Fallback to string conversion
                    text_parts.append(str(item))
            examples = "\n".join(text_parts).strip()
        else:
            examples = content.strip()
    else:
        examples = str(examples_response).strip()

    # Clean up markdown formatting
    if "```" in examples:
        examples = examples.split("```", 1)[-1].strip()

    return examples


# --- LangGraph Node: Prompt Generation ---
@traceable(name="prompt_generation")
def prompt_generation_node(
    requirement: Requirement,
    assignment_description: str,
    examples: str,
    agent_config: dict,
    llm,
) -> str:
    """
    Node for generating a Jinja2 template prompt using requirement, assignment, and examples.
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
    
    # Split examples into correct and erroneous for better template structure
    correct_examples, erroneous_examples = split_examples(examples)
    
    # Pass examples as context so LLM understands what examples will be provided,
    # but instruct it to use {{ correct_examples }} and {{ erroneous_examples }} placeholders
    prompt = template.render(
        requirement=requirement_body,
        assignment_description=assignment_description,
        code="{{ code }}",
        correct_examples=correct_examples,  # Pass as context
        erroneous_examples=erroneous_examples,   # Pass as context
    )

    logger.info("[Node] Invoking LLM for template...")
    response = llm.invoke([HumanMessage(content=prompt)])

    # Fix: Extract content properly based on LLM type
    if hasattr(response, "content"):
        content = response.content
        # Handle case where content is a list (e.g., Gemini)
        if isinstance(content, list):
            # Extract text from each part in the list, ignoring extras
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    # Fallback to string conversion
                    text_parts.append(str(item))
            jinja_template = "\n".join(text_parts).strip()
        else:
            jinja_template = content.strip()
    else:
        jinja_template = str(response).strip()

    # Clean up markdown formatting
    if "```jinja" in jinja_template:
        jinja_template = (
            jinja_template.split("```jinja", 1)[-1].split("```", 1)[0].strip()
        )
    elif "```" in jinja_template:
        jinja_template = jinja_template.split("```", 1)[-1].strip()
    
    # Clean escaped braces (LLM sometimes generates \{\{ instead of {{)
    jinja_template = jinja_template.replace(r"\{\{", "{{").replace(r"\}\}", "}}")

    # Ensure the template has the basic structure
    if "{{ code }}" not in jinja_template and "{{code}}" not in jinja_template:
        jinja_template = jinja_template.replace("{{ student_code }}", "{{ code }}")
        jinja_template = jinja_template.replace("{{code}}", "{{ code }}")
        if "{{ code }}" not in jinja_template:
            jinja_template += "\n\nCode to analyze:\n{{ code }}"
    
    # Ensure the template has {{ correct_examples }} and {{ erroneous_examples }} placeholders
    # If not present, try to add them in appropriate places
    has_good = "{{ correct_examples }}" in jinja_template or "{{correct_examples}}" in jinja_template
    has_bad = "{{ erroneous_examples }}" in jinja_template or "{{erroneous_examples}}" in jinja_template
    
    # Also check for old {{ examples }} format and suggest replacement
    if "{{ examples }}" in jinja_template or "{{examples}}" in jinja_template:
        logger.warning("Template uses old {{ examples }} format. Consider updating to use {{ correct_examples }} and {{ erroneous_examples }} separately.")
    
    # If neither placeholder is present, add them before {{ code }}
    if not has_good and not has_bad:
        examples_placeholder = "{{ correct_examples }}\n\n{{ erroneous_examples }}"
        
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

    return jinja_template


class PromptGeneratorState(TypedDict):
    # All fields are Annotated to allow concurrent writes
    assignment_description: Annotated[str, keep_last]
    theory_summary: Annotated[str, keep_last]
    requirements: Annotated[List[Requirement], keep_last]  # List of requirement strings
    
    # Output - use Annotated to allow multiple concurrent writes
    generated_prompts: Annotated[
        List[GeneratedPrompt], concat
    ]  # List of {requirement, template, examples, etc.}
    
    # Extra field for additional metadata/data that can be loaded any time
    extra: Annotated[Optional[Dict[str, Any]], keep_last]


class PromptGeneratorAgent:
    """
    Agent that orchestrates parallel prompt generation using LangGraph with Map-Reduce pattern.
    Supports both individual usage (single requirement) and batch processing (multiple requirements).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = get_agent_config(config, "prompt_generator")
        self.llm = self._setup_llm()
        self.graph = self._build_graph()
        
        # Check if code_corrector has theory_summary enabled (for generated templates)
        code_corrector_config = get_agent_config(config, "code_corrector")
        theory_config_corrector = code_corrector_config.get('theory_summary', {})
        self.theory_summary_for_templates_enabled = theory_config_corrector.get('enabled', False)
        
        # Check if prompt_generator has theory_summary enabled (for example generation)
        theory_config_examples = self.agent_config.get('theory_summary', {})
        self.theory_summary_for_examples_enabled = theory_config_examples.get('enabled', False)
        self.theory_summary_path = theory_config_examples.get('path', 'data/clases_summary.txt')
        
        # Load theory summary content if enabled for example generation
        self.theory_summary_content = ""
        if self.theory_summary_for_examples_enabled:
            self.theory_summary_content = self._load_theory_summary(self.theory_summary_path)
        
        # Log that we're using standard (non-RAG) mode
        logger.info("📝 Initializing Standard Prompt Generator")
        logger.info("=" * 50)
        logger.info("🔧 RAG Mode: DISABLED")
        logger.info(f"📚 Theory Summary for Example Generation: {'ENABLED' if self.theory_summary_for_examples_enabled else 'DISABLED'}")
        logger.info(f"📚 Theory Summary for Code Correction: {'ENABLED' if self.theory_summary_for_templates_enabled else 'DISABLED'}")
        logger.info("=" * 50)

    def _setup_llm(self):
        provider = str(self.agent_config.get("provider", "openai")).lower().strip()
        model_name = self.agent_config.get("model_name", "gpt-4")
        temperature = float(self.agent_config.get("temperature", 0.1))
        
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
            )
        elif provider in ("gemini", "google", "google-genai"):
            import os
            api_key = self.agent_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro",
                temperature=temperature,
                google_api_key=api_key
            )
        else:
            return OllamaLLM(
                model=model_name or "qwen2.5:7b",
                temperature=temperature,
            )

    def _load_theory_summary(self, path: str) -> str:
        """Load theory summary from file if it exists."""
        try:
            summary_path = Path(path)
            if summary_path.exists():
                content = summary_path.read_text(encoding='utf-8')
                logger.info(f"Loaded theory summary from {path} ({len(content)} chars)")
                return content
            else:
                logger.warning(f"Theory summary file not found at {path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading theory summary: {e}")
            return ""

    def _append_theory_summary_section(self, jinja_template: str) -> str:
        """
        Append conditional theory_summary section to the generated template.
        This section will be rendered by the code_corrector when theory_summary is enabled.
        """
        if not self.theory_summary_for_templates_enabled:
            return jinja_template
        
        # Find where {{ code }} is and insert theory_summary section before it
        theory_section = (
            "\n{% if theory_summary %}\n"
            "---\n"
            "Course Theory Reference:\n"
            "{{ theory_summary }}\n"
            "---\n"
            "{% endif %}\n\n"
        )
        
        # Insert before {{ code }} (with or without "Code to analyze:" label)
        if "Code to analyze:\n{{ code }}" in jinja_template:
            jinja_template = jinja_template.replace(
                "Code to analyze:\n{{ code }}",
                f"{theory_section}Code to analyze:\n{{{{ code }}}}"
            )
        elif "{{ code }}" in jinja_template:
            jinja_template = jinja_template.replace(
                "{{ code }}",
                f"{theory_section}{{{{ code }}}}"
            )
        else:
            # Fallback: append at the end
            jinja_template += "\n" + theory_section
        
        logger.info("✓ Added theory_summary section to generated template")
        return jinja_template

    # -------------------------- LangGraph Nodes ------------------------------ #

    def _create_requirement_processor(self, requirement: Requirement, index: int):
        """Create a node function for processing a specific requirement"""

        def process_requirement_node(
            state: PromptGeneratorState,
        ) -> PromptGeneratorState:
            """Process a single requirement - generate examples and prompt template"""
            assignment_description = state["assignment_description"]
            theory_summary = state.get("theory_summary", "")

            logger.info(f"Processing requirement {index + 1}: {requirement['requirement'][:50]}...")

            # Generate examples
            examples = example_generation_node(requirement, assignment_description, theory_summary, self.agent_config, self.llm)

            # Generate Jinja2 template
            jinja_template = prompt_generation_node(
                requirement,
                assignment_description,
                examples,
                self.agent_config,
                self.llm,
            )
            
            # Post-process: Add theory_summary section if enabled
            jinja_template = self._append_theory_summary_section(jinja_template)

            # Store result with full requirement data
            result: GeneratedPrompt = {
                "requirement": requirement,  # Store the full Requirement object
                "examples": examples,
                "jinja_template": jinja_template,
                "index": index,
            }

            # Use the aggregation function for concurrent writes
            state["generated_prompts"] = [result]
            logger.info(f"Completed processing requirement {index + 1}")

            return state

        return process_requirement_node

    def _create_dynamic_graph(self, requirements: List[Requirement]):
        """Create dynamic graph with nodes for each requirement, connecting each directly to END"""
        # Create a fresh graph for this batch without the direct edge
        dynamic_graph = self._build_base_graph(with_direct_edge=False)

        # Add a node for each requirement
        for i, requirement in enumerate(requirements):
            node_name = f"process_requirement_{i}"
            node_function = self._create_requirement_processor(requirement, i)

            dynamic_graph.add_node(node_name, node_function)

            # Connect START to this requirement node
            dynamic_graph.add_edge(START, node_name)

            # Connect this requirement node directly to END
            dynamic_graph.add_edge(node_name, END)

        # Compile the graph with all dynamic connections
        self.graph = dynamic_graph.compile()



    # -------------------------- Graph Builder ------------------------------- #

    def _build_base_graph(self, with_direct_edge: bool = True):
        """Build the base LangGraph structure without dynamic nodes"""
        graph = StateGraph(PromptGeneratorState)

        # Only add direct edge if we're not going to have dynamic nodes
        if with_direct_edge:
            graph.add_edge(START, END)

        return graph

    def _build_graph(self):
        """Build a LangGraph with parallel processing capabilities"""
        # For single requirement, use simple graph
        return self._build_base_graph().compile()

    # -------------------------- Public API ---------------------------------- #

    def generate_prompt(
        self, requirement: Requirement, assignment_description: str
    ) -> GeneratedPrompt:
        """
        Generate a single prompt from requirement and assignment content.
        Returns a dictionary with the generated template and examples.

        Args:
            requirement: The Requirement object with requirement, function, and type fields
            assignment_description: The assignment description text (already read from file)
        """
        logger.info("Generating prompt from Requirement object")

        # For single requirement, use batch processing with one item
        results = self.generate_prompts_batch([requirement], assignment_description)

        return results[0]

    def generate_prompts_batch(
        self, requirements: List[Requirement], assignment_description: str
    ) -> List[GeneratedPrompt]:
        """
        Generate multiple prompts in parallel from a list of requirements.
        Returns a list of dictionaries with generated templates and examples.

        Args:
            requirements: List of Requirement objects
            assignment_description: The assignment description text (already read from file)
        """
        logger.info(
            f"Generating {len(requirements)} prompts in parallel using dynamic graph"
        )
        
        # Use pre-loaded theory summary content (empty string if disabled)
        theory_summary = self.theory_summary_content
        if theory_summary:
            logger.info(f"Using theory summary for example generation ({len(theory_summary)} chars)")
        else:
            logger.info("Theory summary for example generation is disabled or not available")

        # Create dynamic graph for this batch
        self._create_dynamic_graph(requirements)

        # Create state for batch processing
        state: PromptGeneratorState = {
            "assignment_description": assignment_description,
            "theory_summary": theory_summary,
            "requirements": requirements,
            "generated_prompts": [],
            "extra": {},
        }

        # Run the graph with dynamic nodes
        result_state = self.graph.invoke(state)

        # Extract results and sort by index to maintain order
        results: List[GeneratedPrompt] = sorted(
            result_state["generated_prompts"], 
            key=lambda x: x["index"]
        )

        logger.info(f"Generated {len(results)} prompts in parallel")
        return results

    def generate_prompts(self, assignment_description: str, rubric: Any) -> Any:
        """
        Generate prompts from a rubric (for compatibility with assignment_evaluator).
        This method converts rubric items to requirements and uses the batch processing logic.

        Args:
            assignment_description: The assignment description text (already read from file)
            rubric: A rubric object with items to convert to requirements

        Returns:
            A prompt set object compatible with the assignment evaluator
        """
        logger.info("Generating prompts from rubric")

        # Convert rubric items to Requirement objects
        requirements: List[Requirement] = []
        for item in rubric.items:
            # Create a Requirement object from the rubric item
            requirement: Requirement = {
                "requirement": f"{item.title}\n{item.description}",
                "function": "",  # No specific function mentioned in rubric
                "type": PromptType.REQUIREMENT_PRESENCE  # Default type for rubric items
            }
            requirements.append(requirement)

        # Use batch processing to generate prompts
        results = self.generate_prompts_batch(requirements, assignment_description)

        # Convert results to the expected format
        # For now, return a simple object that mimics the expected structure
        class PromptSet:
            def __init__(self, prompts):
                self.prompts = prompts

        class CorrectionPrompt:
            def __init__(self, rubric_item_id, rubric_item_title, prompt, criteria):
                self.rubric_item_id = rubric_item_id
                self.rubric_item_title = rubric_item_title
                self.prompt = prompt
                self.criteria = criteria

        prompts = []
        for i, result in enumerate(results):
            rubric_item = rubric.items[i]
            prompt = CorrectionPrompt(
                rubric_item_id=rubric_item.id,
                rubric_item_title=rubric_item.title,
                prompt=result["jinja_template"],
                criteria=[rubric_item.description],  # Use description as criteria
            )
            prompts.append(prompt)

        return PromptSet(prompts)

    def as_graph_node(self):
        """
        Expose the agent as a LangGraph node for use in external graphs.
        Returns a function that takes a state dict and returns a state dict with the generated templates.
        """

        def node(state):
            # If state has requirements list, process in batch
            if "requirements" in state and state["requirements"]:
                # Process each requirement
                for i, requirement in enumerate(state["requirements"]):
                    individual_state = state.copy()
                    individual_state["current_requirement"] = requirement
                    individual_state["current_requirement_index"] = i
                    result_state = self.graph.invoke(individual_state)

                    # Accumulate results
                    if "generated_prompts" not in state:
                        state["generated_prompts"] = []
                    state["generated_prompts"].append(
                        result_state["generated_prompts"][0]
                    )

                return state
            else:
                # Single requirement processing
                result = self.graph.invoke(state)
                return result

        return node
