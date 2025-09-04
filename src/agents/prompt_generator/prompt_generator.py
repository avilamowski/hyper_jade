#!/usr/bin/env python3

"""
Prompt Generator Agent (LangGraph Node Version)

This module defines two main nodes for LangChain/LangGraph:
1. ExampleGenerationNode: Generates examples from a requirement.
2. PromptGenerationNode: Generates a Jinja2 template prompt using examples and assignment description.
Each node logs input/output and timing for traceability.
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict
import logging
import time
from pathlib import Path

from src.agents.requirement_generator.requirement_generator import RequirementGeneratorState
from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from src.config import get_agent_config
from src.agents.utils.agent_evaluator import AgentEvaluator

# Import MLflow logger lazily to avoid circular imports

def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed"""
    try:
        from src.core.mlflow_utils import mlflow_logger
        return mlflow_logger
    except ImportError:
        logger.warning("MLflow not available - logging will be disabled")
        return None


def safe_log_call(logger_instance, method_name, *args, **kwargs):
    """Safely call a logging method, doing nothing if logger is None"""
    if logger_instance is not None and hasattr(logger_instance, method_name):
        try:
            method = getattr(logger_instance, method_name)
            method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to call {method_name}: {e}")


logger = logging.getLogger(__name__)


# --- LangGraph Node: Example Generation ---
def example_generation_node(requirement: str, agent_config: dict, llm, mlflow_logger=None) -> dict:
    """
    Node for generating examples from a requirement.
    Returns a dict with examples and trace info.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import time
    start_time = time.time()
    template_map = agent_config.get("templates", {})
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["jinja"])
    )
    examples_template_file = template_map["examples"]
    examples_template = env.get_template(examples_template_file)
    example_quantity = agent_config.get("example_quantity")
    examples_prompt = examples_template.render(
        requirement=requirement,
        example_quantity=example_quantity
    )
    logger.info("[Node] Invoking LLM for examples...")
    safe_log_call(mlflow_logger, "log_text", examples_prompt, "examples_prompt.txt")
    examples_response = llm.invoke([HumanMessage(content=examples_prompt)])
    examples = str(examples_response).strip()
    if "```" in examples:
        examples = examples.split("```", 1)[-1].strip()
    safe_log_call(mlflow_logger, "log_text", examples, "generated_examples.txt")
    duration = time.time() - start_time
    trace = {
        "node": "example_generation",
        "input": requirement,
        "output": examples,
        "duration": duration
    }
    safe_log_call(mlflow_logger, "log_trace_step", "example_generation", trace, step_number=1)
    return {"examples": examples, "trace": trace}

# --- LangGraph Node: Prompt Generation ---
def prompt_generation_node(requirement: str, assignment_description: str, examples: str, agent_config: dict, llm, mlflow_logger=None) -> dict:
    """
    Node for generating a Jinja2 template prompt using requirement, assignment, and examples.
    Returns a dict with jinja_template and trace info.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from src.agents.utils.prompt_types import PromptType
    import time
    start_time = time.time()
    # Extract prompt type from requirement (first line)
    lines = requirement.splitlines()
    prompt_type = PromptType.PRESENCE
    requirement_body = requirement
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("[") and "]" in first_line:
            type_tag = first_line[1:first_line.index("]")].strip().lower()
            prompt_type = PromptType(type_tag)
            requirement_body = first_line[first_line.index("]")+1:] + ("\n".join(lines[1:]).strip() if len(lines) > 1 else "")
    template_map = agent_config.get("templates", {})
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["jinja"])
    )
    template_file = template_map.get(prompt_type.value, template_map.get("default"))
    template = env.get_template(template_file)
    prompt = template.render(requirement=requirement_body, assignment_description=assignment_description, code="{{ code }}", examples=examples)
    safe_log_call(mlflow_logger, "log_text", prompt, "final_prompt.txt")
    logger.info("[Node] Invoking LLM for template...")
    response = llm.invoke([HumanMessage(content=prompt)])
    jinja_template = str(response).strip()
    # Clean up markdown formatting
    if "```jinja" in jinja_template:
        jinja_template = jinja_template.split("```jinja", 1)[-1].split("```", 1)[0].strip()
    elif "```" in jinja_template:
        jinja_template = jinja_template.split("```", 1)[-1].strip()
    # Ensure the template has the basic structure
    if "{{ code }}" not in jinja_template and "{{code}}" not in jinja_template:
        jinja_template = jinja_template.replace("{{ student_code }}", "{{ code }}")
        jinja_template = jinja_template.replace("{{code}}", "{{ code }}")
        if "{{ code }}" not in jinja_template:
            jinja_template += "\n\nCode to analyze:\n{{ code }}"
    duration = time.time() - start_time
    trace = {
        "node": "prompt_generation",
        "input": {
            "requirement": requirement,
            "assignment_description": assignment_description,
            "examples": examples
        },
        "output": jinja_template,
        "duration": duration
    }
    safe_log_call(mlflow_logger, "log_trace_step", "prompt_generation", trace, step_number=2)
    return {"jinja_template": jinja_template, "trace": trace}


# --- LangGraph Graph Definition ---
try:
    from langgraph.graph import StateGraph
except ImportError:
    StateGraph = None  # LangGraph not installed

class PromptGeneratorState(RequirementGeneratorState):
    prompt_template: str


class PromptGeneratorAgent:
    """
    Agent that orchestrates the example and prompt generation nodes, and exposes itself as a LangGraph node/graph.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = get_agent_config(config, 'prompt_generator')
        self.llm = self._setup_llm()
        self.evaluator = None
        if config.get('agents', {}).get('agent_evaluator', {}).get('enabled', False):
            self.evaluator = AgentEvaluator(config)
        self.graph = self._build_graph() if StateGraph else None

    def _setup_llm(self):
        if self.agent_config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.agent_config.get("model_name", "gpt-4"),
                temperature=self.agent_config.get("temperature", 0.1)
            )
        else:
            return OllamaLLM(
                model=self.agent_config.get("model_name", "qwen2.5:7b"),
                temperature=self.agent_config.get("temperature", 0.1)
            )

    def _build_graph(self):
        """
        Build a LangGraph graph that wires up the example and prompt generation nodes.
        The graph expects input dict with keys: requirement, assignment_description.
        """
        graph = StateGraph(PromptGeneratorState)
        # Register nodes
        def example_node_wrapper(state):
            result = example_generation_node(
                state["requirement"], self.agent_config, self.llm, get_mlflow_logger()
            )
            state = dict(state)
            state["examples"] = result["examples"]
            state["example_trace"] = result["trace"]
            return state
        def prompt_node_wrapper(state):
            result = prompt_generation_node(
                state["requirement"], state["assignment_description"], state["examples"], self.agent_config, self.llm, get_mlflow_logger()
            )
            state = dict(state)
            state["jinja_template"] = result["jinja_template"]
            state["prompt_trace"] = result["trace"]
            return state
        graph.add_node("example_generation", example_node_wrapper)
        graph.add_node("prompt_generation", prompt_node_wrapper)
        # Wire nodes: example_generation -> prompt_generation
        graph.set_entry_point("example_generation")
        graph.add_edge("example_generation", "prompt_generation")
        graph.add_edge("prompt_generation", "__end__")
        return graph

    def as_graph_node(self):
        """
        Expose the agent as a LangGraph node for use in external graphs.
        Returns a function that takes a state dict and returns a state dict with the generated template.
        """
        def node(state):
            # Run the graph if available
            if self.graph:
                result = self.graph.run(state)
                return result
            # Fallback: run sequentially
            requirement = state["requirement"]
            assignment_description = state["assignment_description"]
            example_result = example_generation_node(requirement, self.agent_config, self.llm, get_mlflow_logger())
            examples = example_result["examples"]
            prompt_result = prompt_generation_node(requirement, assignment_description, examples, self.agent_config, self.llm, get_mlflow_logger())
            state = dict(state)
            state["examples"] = examples
            state["jinja_template"] = prompt_result["jinja_template"]
            state["example_trace"] = example_result["trace"]
            state["prompt_trace"] = prompt_result["trace"]
            return state
        return node

    def generate_prompt(self, requirement_file_path: str, assignment_file_path: str, output_file_path: str) -> str:
        """
        Orchestrates the nodes for prompt generation.
        """
        mlflow_logger = get_mlflow_logger()
        safe_log_call(mlflow_logger, "start_run",
            run_name="prompt_generation",
            tags={
                "agent": "prompt_generator",
                "requirement_file": Path(requirement_file_path).name,
                "assignment_file": Path(assignment_file_path).name,
                "output_file": Path(output_file_path).name
            }
        )
        start_time = time.time()
        try:
            with open(requirement_file_path, 'r', encoding='utf-8') as f:
                requirement = f.read().strip()
            with open(assignment_file_path, 'r', encoding='utf-8') as f:
                assignment_description = f.read().strip()
            safe_log_call(mlflow_logger, "log_text", requirement, "input_requirement.txt")
            safe_log_call(mlflow_logger, "log_text", assignment_description, "input_assignment.txt")
            safe_log_call(mlflow_logger, "log_trace_step", "read_inputs", {
                "requirement_file": requirement_file_path,
                "assignment_file": assignment_file_path,
                "requirement_length": len(requirement),
                "assignment_length": len(assignment_description)
            }, step_number=0)
            # Node 1: Example Generation
            example_result = example_generation_node(requirement, self.agent_config, self.llm, mlflow_logger)
            examples = example_result["examples"]
            # Node 2: Prompt Generation
            prompt_result = prompt_generation_node(requirement, assignment_description, examples, self.agent_config, self.llm, mlflow_logger)
            jinja_template = prompt_result["jinja_template"]
            # Save the template
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(jinja_template)
            prompt_name = Path(output_file_path).stem
            safe_log_call(mlflow_logger, "log_text", jinja_template, f"generated_templates/{Path(output_file_path).name}")
            safe_log_call(mlflow_logger, "log_prompt_metrics", jinja_template, prompt_name)
            # Evaluate agent output if evaluator is enabled
            if self.evaluator:
                try:
                    logger.info("Evaluating prompt generator output...")
                    evaluation_result = self.evaluator.evaluate_prompt_generator(
                        requirement,
                        assignment_description,
                        jinja_template,
                        output_file_path
                    )
                    safe_log_call(mlflow_logger, "log_agent_evaluation_metrics", "prompt_generator", evaluation_result)
                    logger.info(f"Prompt generator evaluation completed. Overall score: {evaluation_result.get('overall_score', 'N/A')}")
                except Exception as eval_error:
                    logger.warning(f"Error during prompt generator evaluation: {eval_error}")
            total_time = time.time() - start_time
            safe_log_call(mlflow_logger, "log_metrics", {
                "total_generation_time_seconds": total_time,
                "template_generation_rate": 1.0 / total_time if total_time > 0 else 0
            })
            logger.info(f"Generated Jinja2 template: {output_path}")
            return str(output_path)
        except Exception as e:
            safe_log_call(mlflow_logger, "log_metric", "error_occurred", 1.0)
            safe_log_call(mlflow_logger, "log_text", str(e), "error_log.txt")
            logger.error(f"Error generating prompt: {e}")
            raise
        finally:
            safe_log_call(mlflow_logger, "end_run")
