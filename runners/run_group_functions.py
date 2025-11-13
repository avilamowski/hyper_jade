#!/usr/bin/env python3

"""
Group Functions Runner

Script to run the Group Functions Agent on student submissions.
This script handles I/O operations and coordinates the grouping of functions from student code.

Usage:
    python run_group_functions.py --assignment_folder examples/ej1-2025-s1-r2 --output_folder outputs/grouped_functions
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from src.agents.group_functions import GroupFunctionsAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(
    config_path: str = "src/config/assignment_config.yaml",
) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {"provider": "ollama", "model_name": "qwen2.5:7b", "temperature": 0.1}


def load_assignment_description(assignment_folder: Path) -> str:
    """Load assignment description from consigna.txt"""
    consigna_path = assignment_folder / "consigna.txt"
    if not consigna_path.exists():
        raise FileNotFoundError(f"Assignment description not found: {consigna_path}")

    with open(consigna_path, "r", encoding="utf-8") as file:
        return file.read()


def load_student_submissions(assignment_folder: Path) -> List[Dict[str, str]]:
    """Load all Python files from assignment folder as student submissions"""
    submissions = []

    # Find all .py files in the assignment folder
    python_files = list(assignment_folder.glob("*.py"))

    for py_file in python_files:
        # Skip files that might be test files or other non-student files
        if py_file.name.startswith("test_") or py_file.name == "__init__.py":
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as file:
                code = file.read()

            submissions.append(
                {"name": py_file.stem, "code": code}  # filename without extension
            )

        except Exception as e:
            logger.warning(f"Failed to load {py_file}: {e}")
            continue

    logger.info(f"Loaded {len(submissions)} student submissions")
    return submissions


def save_grouped_code(grouped_code: List[Dict[str, str]], output_folder: Path) -> None:
    """Save grouped code to output files"""
    output_folder.mkdir(parents=True, exist_ok=True)

    # Group by submission name and function name
    for group in grouped_code:
        function_name = group["function_name"]
        submission_name = group["submission_name"]
        code = group["code"]

        # Skip empty code (function not found in student's submission)
        if not code or code.strip() == "":
            logger.info(f"Skipping empty code for {submission_name}_{function_name}")
            continue

        # Create filename: {submission_name}_{function_name}.lines
        filename = f"{submission_name}_{function_name}.lines"
        output_path = output_folder / filename

        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(code)

            logger.info(f"Saved grouped code: {filename}")

        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")


def save_function_names(
    function_names: List[str],
    output_folder: Path,
) -> None:
    """Save identified function names to text files"""
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save function names
    function_names_path = output_folder / "identified_functions.txt"

    try:
        with open(function_names_path, "w", encoding="utf-8") as file:
            if function_names:
                file.write("Identified functions:\n\n")
                for func_name in function_names:
                    file.write(f"Function: {func_name}\n")
            else:
                file.write(
                    "No function names were identified in the assignment description.\n"
                )

        logger.info(f"Saved function names list: {function_names_path}")

    except Exception as e:
        logger.error(f"Failed to save function names: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Group Functions Agent")
    parser.add_argument(
        "--assignment_folder",
        type=str,
        required=True,
        help="Path to folder containing consigna.txt and student submission files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to output folder for grouped code files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/assignment_config.yaml",
        help="Path to configuration file (default: src/config/assignment_config.yaml)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    assignment_folder = Path(args.assignment_folder)
    output_folder = Path(args.output_folder)

    # Validate input folder
    if not assignment_folder.exists():
        logger.error(f"Assignment folder does not exist: {assignment_folder}")
        sys.exit(1)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration: {config}")

        # Load assignment description
        assignment_description = load_assignment_description(assignment_folder)
        logger.info(
            f"Loaded assignment description ({len(assignment_description)} characters)"
        )

        # Load student submissions
        submissions = load_student_submissions(assignment_folder)

        if not submissions:
            logger.warning("No student submissions found")
            return

        # Initialize agent
        agent = GroupFunctionsAgent(config)
        logger.info("Initialized Group Functions Agent")

        # Process submissions
        results = agent.process_submissions(assignment_description, submissions)

        function_names = results["function_names"]
        grouped_code = results["grouped_code"]

        logger.info(f"Identified {len(function_names)} function names")
        logger.info(f"Generated {len(grouped_code)} grouped code segments")

        # Save results
        save_function_names(function_names, output_folder)
        save_grouped_code(grouped_code, output_folder)

        # Print summary
        print(f"\n=== Group Functions Results ===")
        print(f"Assignment folder: {assignment_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Submissions processed: {len(submissions)}")
        print(f"Function names identified: {len(function_names)}")
        if function_names:
            print(f"Functions: {'; '.join(function_names)}")
        print(f"Grouped code segments: {len(grouped_code)}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
