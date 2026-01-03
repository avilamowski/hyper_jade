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

logger = logging.getLogger(__name__)


# --- LangGraph Node: RAG-Enhanced Example Generation ---
@traceable(name="rag_generate_examples")
async def rag_example_generation_node(requirement: Requirement, agent_config: dict, llm, rag_system, code_generator) -> str:
    """
    Node for generating examples using RAG system and course theory.
    Returns the generated examples as a string.
    """
    try:
        # Generate examples using RAG system
        import asyncio
        
        # Generate examples using RAG system
        examples = await code_generator.generate_enhanced_examples(
            requirement=requirement["requirement"],
            num_examples=3,
            max_theory_results=5,
            dataset="python"  # Default to python for now
        )
        
        if not examples:
            logger.error("RAG example generation failed: No examples generated")
            return "Error generating examples using RAG system."
        
        # Format examples for the prompt
        formatted_examples = []
        logger.info(f"Formatting {len(examples)} examples for prompt generation")
        for i, example in enumerate(examples, 1):
            code = example.get("code", "")
            description = example.get("description", f"Example {i}")
            improvements = example.get("improvements", [])
            theory_alignment = example.get("theory_alignment", "")
            class_name = example.get("class_name", "Unknown")
            
            logger.info(f"Example {i} details: class_name={class_name}, "
                       f"improvements_type={type(improvements).__name__}, "
                       f"theory_alignment_type={type(theory_alignment).__name__}, "
                       f"code_length={len(code)}, "
                       f"has_improvements={bool(improvements)}, "
                       f"has_theory_alignment={bool(theory_alignment)}")
            
            # Log the actual content for debugging
            if improvements:
                logger.info(f"  Example {i} improvements: {improvements}")
            if theory_alignment:
                logger.info(f"  Example {i} theory_alignment (first 200 chars): {str(theory_alignment)[:200]}")
            
            formatted_example = f"Example {i}: {description}\n"
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
            
            formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples)
        
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
    
    # Create enhanced prompt that includes RAG information
    # Read the template file directly to get the raw template content
    template_path = f"templates/{template_file}"
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        logger.error(f"Error reading template file {template_path}: {e}")
        # Fallback to rendering the template
        template_content = template.render(
            requirement=requirement_body,
            assignment_description=assignment_description,
            code="{{ code }}",
            examples=examples,
        )
    
    # Create a prompt that includes the template structure and the actual values
    enhanced_prompt = f"""
You are a Prompt Engineer creating a Jinja2 template for a code analyzer.

REQUIREMENT TO ANALYZE:
{requirement_body}

ASSIGNMENT DESCRIPTION:
{assignment_description}

RAG-ENHANCED EXAMPLES:
{examples}

TEMPLATE STRUCTURE TO FOLLOW:
{template_content}

Create a complete Jinja2 template that follows the structure above but fills in the specific requirement and uses the provided examples.
"""
    
    # Add RAG-specific instructions
    rag_instructions = """

IMPORTANT RAG ENHANCEMENT:
The examples above were generated using course theory and materials.
Use these examples as a strong foundation for your analysis template.
The examples are based on actual course content and should provide better
context for student code analysis.

"""
    
    enhanced_prompt += rag_instructions

    logger.info("[RAG Node] Invoking LLM for template...")
    response = llm.invoke([HumanMessage(content=enhanced_prompt)])

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

    # Ensure the template has the basic structure
    if "{{ code }}" not in jinja_template and "{{code}}" not in jinja_template:
        jinja_template = jinja_template.replace("{{ student_code }}", "{{ code }}")
        jinja_template = jinja_template.replace("{{code}}", "{{ code }}")
        if "{{ code }}" not in jinja_template:
            jinja_template += "\n\nCode to analyze:\n{{ code }}"

    # Debug: Log final template
    logger.info(f"[RAG Node] Final template length: {len(jinja_template)}")
    logger.info(f"[RAG Node] Final template preview: {jinja_template[:200]}...")

    return jinja_template


class RAGPromptGeneratorState(TypedDict):
    # All fields are Annotated to allow concurrent writes
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
                    requirement, self.agent_config, self.llm, self.rag_system, self.code_generator
                )
                
                # Convert type string to PromptType if needed
                req_type = requirement.get("type")
                if isinstance(req_type, str):
                    from src.agents.prompt_generator.prompt_generator import PromptType
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
            import asyncio
            
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
