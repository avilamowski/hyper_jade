#!/usr/bin/env python3

"""
Group Functions Agent (LangGraph Version)

This module implements a group functions generator using LangGraph:
1. extract_function_names: Extracts function names mentioned in assignment description
2. group_code_by_functions: Groups student code by identified functions and their dependencies

Supports both individual usage (single submission) and batch processing (multiple submissions).
All I/O operations (file reading/writing) are handled by the caller.
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional, Annotated
import logging
import re
import ast
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from src.config import get_agent_config
from src.agents.utils.reducers import keep_last, concat
from src.models import GroupedCode

logger = logging.getLogger(__name__)


class GroupFunctionsState(TypedDict):
    # Input
    assignment_description: Annotated[str, keep_last]
    submissions: Annotated[
        List[Dict[str, str]], keep_last
    ]  # List of {name: str, code: str}

    # Intermediate
    function_names: Annotated[
        List[str], keep_last
    ]  # Simple function names from assignment

    # Output - use Annotated to allow multiple concurrent writes
    grouped_code: Annotated[List[GroupedCode], concat]


def extract_functions_from_response(response: str) -> Dict[str, str]:
    """
    Extract functions and their code from the LLM response.
    The response should be in the format:
    <FUNCTION name="function_name">
    def function_name():
        # function code
    </FUNCTION>
    """
    functions = {}
    lines = response.splitlines()
    current_function = None
    current_code = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<FUNCTION name="):
            # Save previous function if it was left unclosed
            if current_function is not None:
                functions[current_function] = "\n".join(current_code)
            match = re.search(r'name="([^"]+)"', stripped)
            if match:
                current_function = match.group(1)
                current_code = []
        elif stripped.startswith("</FUNCTION>"):
            if current_function is not None:
                functions[current_function] = "\n".join(current_code)
                current_function = None
                current_code = []
        elif current_function is not None:
            current_code.append(line)

    # Handle case where last function doesn't have closing tag
    if current_function is not None and current_code:
        functions[current_function] = "\n".join(current_code)

    return functions


def parse_code_and_extract_functions(code: str) -> Dict[str, str]:
    """
    Parse Python code and extract all function definitions WITH line numbers.
    Returns a dictionary with function_name -> function_code_with_line_numbers.
    """
    functions = {}

    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function code based on line numbers
                start_line = node.lineno - 1  # ast uses 1-based indexing

                # Find the end line by looking for the next function or end of file
                end_line = len(lines)
                for other_node in ast.walk(tree):
                    if (
                        isinstance(other_node, ast.FunctionDef)
                        and other_node != node
                        and other_node.lineno > node.lineno
                    ):
                        end_line = min(end_line, other_node.lineno - 1)

                # Extract function code WITH original line numbers
                function_lines = []
                for i in range(start_line, min(end_line, len(lines))):
                    # Add line number prefix: "line_number: code"
                    function_lines.append(f"{i + 1}: {lines[i]}")

                # Remove trailing empty lines
                while (
                    function_lines and not function_lines[-1].split(":", 1)[1].strip()
                ):
                    function_lines.pop()

                if function_lines:
                    functions[node.name] = "\n".join(function_lines)

    except SyntaxError as e:
        logger.warning(f"Syntax error in code, falling back to regex parsing: {e}")
        # Fallback to regex-based parsing
        functions = parse_functions_with_regex(code)

    return functions


def parse_functions_with_regex(code: str) -> Dict[str, str]:
    """
    Fallback function to parse functions using regex when AST fails, WITH line numbers.
    """
    functions = {}
    lines = code.splitlines()

    # Find all function definitions
    for i, line in enumerate(lines):
        if re.match(r"^\s*def\s+(\w+)\s*\(", line):
            match = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            if match:
                func_name = match.group(1)

                # Find the end of the function (next function or end of file)
                indent_level = len(line) - len(line.lstrip())
                func_lines = [f"{i + 1}: {line}"]  # Add line number

                for j in range(i + 1, len(lines)):
                    current_line = lines[j]

                    # If we hit another function at the same or lower indent level, stop
                    if (
                        current_line.strip()
                        and not current_line.startswith(" " * (indent_level + 1))
                        and not current_line.startswith("\t")
                        and re.match(r"^\s*def\s+", current_line)
                    ):
                        break

                    func_lines.append(f"{j + 1}: {current_line}")  # Add line number

                # Remove trailing empty lines
                while func_lines and not func_lines[-1].split(":", 1)[1].strip():
                    func_lines.pop()

                if func_lines:
                    functions[func_name] = "\n".join(func_lines)

    return functions


def find_function_calls(function_code: str) -> set:
    """
    Find all function calls within a function's code.
    Returns a set of function names that are called.
    """
    called_functions = set()

    try:
        tree = ast.parse(function_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Simple function call: func_name()
                    called_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method() - we might want to include this too
                    pass

    except SyntaxError:
        # Fallback to regex parsing
        # Look for pattern: word followed by opening parenthesis
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        matches = re.findall(pattern, function_code)
        called_functions.update(matches)

    # Filter out built-in functions and common keywords
    builtin_functions = {
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "max",
        "min",
        "sum",
        "abs",
        "round",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "map",
        "filter",
        "any",
        "all",
        "ord",
        "chr",
        "bin",
        "hex",
        "oct",
    }

    return called_functions - builtin_functions


def get_function_dependencies_recursive(
    target_function: str, all_functions: Dict[str, str]
) -> Dict[str, str]:
    """
    Recursively find all dependencies of a target function.
    Returns a dictionary with all required functions and their code.
    """
    if target_function not in all_functions:
        logger.warning(f"Function '{target_function}' not found in code")
        return {}

    dependencies = {}
    visited = set()

    def collect_dependencies(func_name: str):
        if func_name in visited or func_name not in all_functions:
            return

        visited.add(func_name)
        dependencies[func_name] = all_functions[func_name]

        # Find what functions this function calls
        called_functions = find_function_calls(all_functions[func_name])

        # Recursively collect dependencies
        for called_func in called_functions:
            if called_func in all_functions:
                collect_dependencies(called_func)

    collect_dependencies(target_function)
    return dependencies


def add_line_numbers_to_text(text: str) -> str:
    """
    Add line numbers to each line of text.
    Returns text with format: "1: line content"
    """
    lines = text.splitlines()
    numbered_lines = []

    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i}: {line}")

    return "\n".join(numbered_lines)


class GroupFunctionsAgent:
    """
    Agent that orchestrates function name extraction and code grouping using LangGraph.
    Supports both individual usage (single submission) and batch processing (multiple submissions).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = get_agent_config(config, "group_functions")
        self.llm = self._setup_llm()
        self.graph = self._build_graph()

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

    # -------------------------- LangGraph Nodes ------------------------------ #

    def extract_function_names_node(
        self, state: GroupFunctionsState
    ) -> GroupFunctionsState:
        """
        Node for extracting function names from assignment description.
        Returns the identified function names.
        """
        assignment_description = state["assignment_description"]

        logger.info("[Node] Extracting function names from assignment description")

        # Extract function names using both LLM and regex patterns
        detected_functions_llm = self._extract_functions_with_llm(
            assignment_description
        )
        detected_functions_regex = self._extract_function_names_with_regex(
            assignment_description
        )

        # Combine results
        combined_functions = set()

        # Add regex results
        for func_name in detected_functions_regex:
            combined_functions.add(func_name)

        # Add LLM results
        for func_name in detected_functions_llm:
            combined_functions.add(func_name)

        # Store as simple list of function names
        function_names = list(combined_functions)
        state["function_names"] = function_names

        logger.info(
            f"[Node] Extracted {len(function_names)} function names: {function_names}"
        )

        return state

    def _extract_functions_with_llm(self, assignment: str) -> List[str]:
        """
        Use LLM to extract function names from assignment description.
        """
        prompt = f"""
        # Instructions
        You are an autonomous agent responsible for identifying function names mentioned explicitly in an assignment description.
        You MUST analyze the assignment description and identify function names that are explicitly mentioned.
        You MUST return ONLY the function names, one per line.
        
        Function names are typically identified by:
        - Text like "Create a function called function_name"
        - Text like "def function_name():"
        - Text like "The function_name function should..."
        - Backquoted function names like `function_name`
        
        Return only the function names, nothing else. If no function names are found, return an empty response.
        
        # Assignment Description
        {assignment}
        """

        logger.info("[Node] Using LLM to extract function names...")
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Extract content properly based on LLM type
        if hasattr(response, "content"):
            content = response.content.strip()
        else:
            content = str(response).strip()

        # Parse LLM response
        function_names = []
        if content:
            lines = content.splitlines()
            for line in lines:
                func_name = line.strip()
                if func_name and func_name.isidentifier():
                    function_names.append(func_name)

        return function_names

    def _extract_function_names_with_regex(self, assignment: str) -> List[str]:
        """
        Extract function names from assignment description using regex patterns.
        """
        function_names = set()

        # Look for function names in various patterns
        function_patterns = [
            r"`([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",  # `function_name(`
            r"`([a-zA-Z_][a-zA-Z0-9_]*)`",  # `function_name`
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",  # def function_name(
            r"function\s+called\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # function called function_name
            r"función\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # función function_name
            r"The\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+function",  # The function_name function
            r"La\s+función\s+([a-zA-Z_][a-zA-Z0-9_]*)",  # La función function_name
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*\)",  # function_name()
        ]

        for pattern in function_patterns:
            matches = re.findall(pattern, assignment, re.IGNORECASE)
            for match in matches:
                func_name = match.strip()
                # Filter out common words that aren't functions
                if func_name.lower() not in {
                    "the",
                    "la",
                    "function",
                    "función",
                    "def",
                    "called",
                    "text",
                    "texto",
                    "string",
                    "input",
                }:
                    function_names.add(func_name)

        return list(function_names)

    def group_code_by_functions_node(
        self, state: GroupFunctionsState
    ) -> GroupFunctionsState:
        """
        Node for grouping code by functions and their dependencies using recursive analysis.
        Processes all submissions and groups code for each identified function.
        """
        function_names = state["function_names"]
        submissions = state["submissions"]

        if not function_names:
            logger.info("[Node] No function names found, skipping code grouping")
            state["grouped_code"] = []
            return state

        logger.info(
            f"[Node] Grouping code for {len(function_names)} functions across {len(submissions)} submissions..."
        )

        grouped_results = []

        for submission in submissions:
            submission_name = submission["name"]
            submission_code = submission["code"]

            logger.info(f"[Node] Processing submission: {submission_name}")

            # Parse all functions from the submission code
            all_functions = parse_code_and_extract_functions(submission_code)
            logger.info(
                f"[Node] Found {len(all_functions)} functions in {submission_name}: {list(all_functions.keys())}"
            )

            # For each function name, get all its dependencies
            for target_function in function_names:
                if target_function in all_functions:
                    logger.info(
                        f"[Node] Analyzing dependencies for function '{target_function}'"
                    )

                    # Get all dependencies recursively
                    dependencies = get_function_dependencies_recursive(
                        target_function, all_functions
                    )

                    if dependencies:
                        # Combine all dependency code in a logical order
                        # Put the target function last
                        ordered_functions = []
                        for func_name, func_code in dependencies.items():
                            if func_name != target_function:
                                ordered_functions.append(func_code)

                        # Add target function last
                        if target_function in dependencies:
                            ordered_functions.append(dependencies[target_function])

                        combined_code = "\n\n".join(ordered_functions)

                        grouped_result: GroupedCode = {
                            "function_name": target_function,
                            "code": combined_code,
                            "submission_name": submission_name,
                            "line_numbers": [],  # Line numbers will come from the student code
                        }
                        grouped_results.append(grouped_result)

                        logger.info(
                            f"[Node] Grouped function '{target_function}' with {len(dependencies)} dependencies"
                        )
                    else:
                        logger.warning(
                            f"[Node] No dependencies found for function '{target_function}' in {submission_name}"
                        )
                else:
                    logger.info(
                        f"[Node] Function '{target_function}' not found in {submission_name}"
                    )

        state["grouped_code"] = grouped_results
        logger.info(f"[Node] Total grouped code segments: {len(grouped_results)}")

        return state

    # -------------------------- Graph Builder ------------------------------- #

    def _build_graph(self):
        """Build a LangGraph with sequential processing"""
        graph = StateGraph(GroupFunctionsState)

        # Add nodes
        graph.add_node("extract_function_names", self.extract_function_names_node)
        graph.add_node("group_code_by_functions", self.group_code_by_functions_node)

        # Define flow
        graph.add_edge(START, "extract_function_names")
        graph.add_edge("extract_function_names", "group_code_by_functions")
        graph.add_edge("group_code_by_functions", END)

        return graph.compile()

    # -------------------------- Public API ---------------------------------- #

    def process_submissions(
        self, assignment_description: str, submissions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Process multiple submissions to extract function names and group code.

        Args:
            assignment_description: The assignment description text
            submissions: List of dictionaries with 'name' and 'code' keys

        Returns:
            Dictionary with function_names and grouped_code results
        """
        logger.info(f"Processing {len(submissions)} submissions for function grouping")

        # Create state
        state: GroupFunctionsState = {
            "assignment_description": assignment_description,
            "submissions": submissions,
            "function_names": [],
            "grouped_code": [],
        }

        # Run the graph
        result_state = self.graph.invoke(state)

        return {
            "function_names": result_state["function_names"],
            "grouped_code": result_state["grouped_code"],
        }

    def extract_function_names(self, assignment_description: str) -> List[str]:
        """
        Extract function names from assignment description only.

        Args:
            assignment_description: The assignment description text

        Returns:
            List of function names (strings)
        """
        logger.info("Extracting function names from assignment description")

        # Create minimal state
        state: GroupFunctionsState = {
            "assignment_description": assignment_description,
            "submissions": [],
            "function_names": [],
            "grouped_code": [],
        }

        # Run only the extraction node
        result_state = self.extract_function_names_node(state)

        return result_state["function_names"]

    def group_code_by_functions(
        self,
        function_names: List[str],
        submissions: List[Dict[str, str]],
    ) -> List[GroupedCode]:
        """
        Group code by specified function names.

        Args:
            function_names: List of function names (strings)
            submissions: List of dictionaries with 'name' and 'code' keys

        Returns:
            List of GroupedCode objects
        """
        logger.info(
            f"Grouping code for {len(function_names)} functions across {len(submissions)} submissions"
        )

        # Create state
        state: GroupFunctionsState = {
            "assignment_description": "",
            "submissions": submissions,
            "function_names": function_names,
            "grouped_code": [],
        }

        # Run only the grouping node
        result_state = self.group_code_by_functions_node(state)

        return result_state["grouped_code"]

    def as_graph_node(self):
        """
        Expose the agent as a LangGraph node for use in external graphs.
        Returns a function that takes a state dict and returns a state dict with the results.
        """

        def node(state):
            # Extract required fields from state
            assignment_description = state.get("assignment_description", "")
            submissions = state.get("submissions", [])

            # Process submissions
            results = self.process_submissions(assignment_description, submissions)

            # Update state with results
            state["function_names"] = results["function_names"]
            state["grouped_code"] = results["grouped_code"]

            return state

        return node
