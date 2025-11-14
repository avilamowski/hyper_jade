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

from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from src.config import get_agent_config
from src.agents.utils.reducers import keep_last, concat
from src.models import GroupedCode

# LangSmith tracing: optional
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except Exception:
    LANGSMITH_AVAILABLE = False
    def traceable(name: str = None, run_type: str = None):
        def decorator(func):
            return func
        return decorator

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
    
    # Extra field for additional metadata/data that can be loaded any time
    extra: Annotated[Optional[Dict[str, Any]], keep_last]


@traceable(name="parse_and_extract_functions", run_type="parser")
def parse_code_and_extract_functions(code: str, include_line_numbers: bool = False) -> Dict[str, str]:
    """
    Parse Python code and extract all function definitions.
    
    Args:
        code: The Python code to parse
        include_line_numbers: If True, prefix each line with "line_number: ". Default is False.
    
    Returns:
        Dictionary with function_name -> function_code (with or without line numbers)
    """
    functions = {}

    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        # Get all function nodes sorted by line number
        function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        function_nodes.sort(key=lambda n: n.lineno)

        for node in function_nodes:
            start_line = node.lineno - 1  # ast uses 1-based indexing
            
            # Get the indentation level of the function definition
            func_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            # Find the end line by looking for code at same or lower indentation level
            end_line = start_line + 1
            
            # Skip through the function body
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                
                # Skip empty lines and comments within the function
                if not line.strip() or line.strip().startswith('#'):
                    end_line = i + 1
                    continue
                
                # Get indentation of current line
                current_indent = len(line) - len(line.lstrip())
                
                # If we find a line at same or lower indentation than the function def, stop
                if current_indent <= func_indent:
                    break
                    
                end_line = i + 1

            # Extract function code
            function_lines = []
            for i in range(start_line, end_line):
                if include_line_numbers:
                    # Add line number prefix: "line_number: code"
                    function_lines.append(f"{i + 1}: {lines[i]}")
                else:
                    function_lines.append(lines[i])

            # Remove trailing empty lines
            if include_line_numbers:
                while function_lines and not function_lines[-1].split(":", 1)[1].strip():
                    function_lines.pop()
            else:
                while function_lines and not function_lines[-1].strip():
                    function_lines.pop()

            if function_lines:
                functions[node.name] = "\n".join(function_lines)

    except SyntaxError as e:
        logger.warning(f"Syntax error in code, falling back to regex parsing: {e}")
        # Fallback to regex-based parsing
        functions = parse_functions_with_regex(code, include_line_numbers)

    return functions


def parse_functions_with_regex(code: str, include_line_numbers: bool = False) -> Dict[str, str]:
    """
    Fallback function to parse functions using regex when AST fails.
    
    Args:
        code: The Python code to parse
        include_line_numbers: If True, prefix each line with "line_number: ". Default is False.
    
    Returns:
        Dictionary with function_name -> function_code (with or without line numbers)
    """
    functions = {}
    lines = code.splitlines()

    # Find all function definitions
    for i, line in enumerate(lines):
        if re.match(r"^\s*def\s+(\w+)\s*\(", line):
            match = re.match(r"^\s*def\s+(\w+)\s*\(", line)
            if match:
                func_name = match.group(1)

                # Get the indentation level of the function definition
                func_indent = len(line) - len(line.lstrip())
                if include_line_numbers:
                    func_lines = [f"{i + 1}: {line}"]
                else:
                    func_lines = [line]

                # Find the end of the function by checking indentation
                for j in range(i + 1, len(lines)):
                    current_line = lines[j]

                    # Skip empty lines and comments
                    if not current_line.strip() or current_line.strip().startswith('#'):
                        if include_line_numbers:
                            func_lines.append(f"{j + 1}: {current_line}")
                        else:
                            func_lines.append(current_line)
                        continue

                    # Get indentation of current line
                    current_indent = len(current_line) - len(current_line.lstrip())

                    # If we hit a line at same or lower indentation than the function def, stop
                    if current_indent <= func_indent:
                        break

                    if include_line_numbers:
                        func_lines.append(f"{j + 1}: {current_line}")
                    else:
                        func_lines.append(current_line)

                # Remove trailing empty lines
                if include_line_numbers:
                    while func_lines and not func_lines[-1].split(":", 1)[1].strip():
                        func_lines.pop()
                else:
                    while func_lines and not func_lines[-1].strip():
                        func_lines.pop()

                if func_lines:
                    functions[func_name] = "\n".join(func_lines)

    return functions


def find_function_calls(function_code: str, available_functions: set = None) -> set:
    """
    Find all function calls within a function's code.
    Only returns functions that are defined in available_functions (user-defined functions).
    
    Args:
        function_code: The code of the function to analyze
        available_functions: Set of user-defined function names. If None, returns all calls.
    
    Returns:
        Set of function names that are called and defined by the user
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

    # If available_functions is provided, only keep calls to user-defined functions
    if available_functions is not None:
        called_functions = called_functions & available_functions

    return called_functions


@traceable(name="find_function_dependencies", run_type="chain")
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
    available_function_names = set(all_functions.keys())

    def collect_dependencies(func_name: str):
        if func_name in visited or func_name not in all_functions:
            return

        visited.add(func_name)
        dependencies[func_name] = all_functions[func_name]

        # Find what user-defined functions this function calls
        called_functions = find_function_calls(all_functions[func_name], available_function_names)

        # Recursively collect dependencies
        for called_func in called_functions:
            if called_func in all_functions:
                collect_dependencies(called_func)

    collect_dependencies(target_function)
    return dependencies


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
    def _extract_line_numbers(self, code_with_line_numbers: str) -> List[int]:
        """
        Extract line numbers from code formatted as "line_number: code".
        Returns a list of integer line numbers.
        """
        line_numbers = []
        for line in code_with_line_numbers.split('\n'):
            # Match pattern "number: code"
            match = re.match(r'^(\d+):\s', line)
            if match:
                line_numbers.append(int(match.group(1)))
        return line_numbers

    def _convert_to_ranges(self, line_numbers: List[int]) -> List[str]:
        """
        Convert a list of line numbers to ranges.
        Example: [4, 5, 6, 7, 19, 20, 21, 58, 59] -> ["4-7", "19-21", "58-59"]
        """
        if not line_numbers:
            return []
        
        sorted_lines = sorted(set(line_numbers))
        ranges = []
        start = sorted_lines[0]
        end = sorted_lines[0]
        
        for i in range(1, len(sorted_lines)):
            if sorted_lines[i] == end + 1:
                # Continue the current range
                end = sorted_lines[i]
            else:
                # End current range and start a new one
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = sorted_lines[i]
                end = sorted_lines[i]
        
        # Add the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ranges

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
            include_line_numbers = state.get("extra", {}).get("include_line_numbers", False)
            all_functions = parse_code_and_extract_functions(submission_code, include_line_numbers)
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
                        function_line_ranges = []
                        
                        for func_name, func_code in dependencies.items():
                            if func_name != target_function:
                                ordered_functions.append(func_code)
                                # Extract line numbers and convert to ranges with function name
                                line_nums = self._extract_line_numbers(func_code)
                                ranges = self._convert_to_ranges(line_nums)
                                for range_str in ranges:
                                    function_line_ranges.append(f"{func_name}: {range_str}")

                        # Add target function last
                        if target_function in dependencies:
                            ordered_functions.append(dependencies[target_function])
                            line_nums = self._extract_line_numbers(dependencies[target_function])
                            ranges = self._convert_to_ranges(line_nums)
                            for range_str in ranges:
                                function_line_ranges.append(f"{target_function}: {range_str}")

                        combined_code = "\n\n".join(ordered_functions)

                        grouped_result: GroupedCode = {
                            "function_name": target_function,
                            "code": combined_code,
                            "submission_name": submission_name,
                            "line_numbers": function_line_ranges,  # Function names with line ranges
                        }
                        grouped_results.append(grouped_result)

                        logger.info(
                            f"[Node] Grouped function '{target_function}' with {len(dependencies)} dependencies in {len(function_line_ranges)} line ranges"
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
        graph.add_node("group_code_by_functions", self.group_code_by_functions_node)

        # Define flow
        graph.add_edge(START, "group_code_by_functions")
        graph.add_edge("group_code_by_functions", END)

        return graph.compile()

    # -------------------------- Public API ---------------------------------- #
    @traceable(name="group_code_by_functions_execution", run_type="chain")
    def group_code_by_functions(
        self,
        function_names: List[str],
        submissions: List[Dict[str, str]],
        include_line_numbers: bool = False,
    ) -> List[GroupedCode]:
        """
        Group code by specified function names.

        Args:
            function_names: List of function names (strings)
            submissions: List of dictionaries with 'name' and 'code' keys
            include_line_numbers: If True, prefix each line with "line_number: ". Default is False.

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
            "extra": {"include_line_numbers": include_line_numbers},
        }

        # Run only the grouping node
        result_state = self.group_code_by_functions_node(state)

        return result_state["grouped_code"]


