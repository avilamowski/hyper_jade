#!/usr/bin/env python3
"""
Linter Correction Agent

This module runs Ruff linter on student code and uses an LLM to filter
relevant errors. Relevant errors are converted into Correction objects
that integrate with the CodeCorrectorAgent pipeline.
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional
import subprocess
import tempfile
import json
import logging
import os
import re

from dotenv import load_dotenv, dotenv_values
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_agent_config
from src.models import Correction, Requirement, PromptType

logger = logging.getLogger(__name__)

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

# --------------------------------------------------------------------------- #
# Load .env
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
DOTENV = dotenv_values()

# --------------------------------------------------------------------------- #
# TypedDicts
# --------------------------------------------------------------------------- #

class LintError(TypedDict):
    """Represents a single linter error."""
    line: int
    column: int
    code: str       # e.g., "F821"
    message: str    # e.g., "undefined name 'random'"
    filename: str


# --------------------------------------------------------------------------- #
# LinterCorrectionAgent
# --------------------------------------------------------------------------- #

class LinterCorrectionAgent:
    """
    Agent that runs Ruff linter on student code, filters relevant errors
    using an LLM, and generates Correction objects for integration with
    the CodeCorrectorAgent.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.agent_config = get_agent_config(self.config, "code_corrector")
        self.linter_config = self.agent_config.get("linter", {})
        
        # Get template path
        self.template_path = self.linter_config.get(
            "template", 
            "linter/linter_relevance.jinja"
        )
        
        # Load categories and exclusions
        self.include_categories = self.linter_config.get("include_categories", ["E", "F", "W"])
        self.exclude_codes = self.linter_config.get("exclude_codes", [
            "E501",  # Line too long
            "W291",  # Trailing whitespace
            "W292",  # No newline at end of file
        ])
        
        # Setup LLM for filtering
        self.llm = self._setup_llm()
        
        # Load Jinja template
        self.template = self._load_template()

    def _setup_llm(self):
        """Setup the LLM for error relevance filtering."""
        provider = str(self.agent_config.get("provider", "openai")).lower().strip()
        model_name = self.agent_config.get("model_name", "gpt-4o-mini")
        temperature = float(self.agent_config.get("temperature", 0.1))

        if provider == "openai":
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
                model=model_name or "gemini-pro",
                temperature=temperature,
                google_api_key=api_key
            )

        return OllamaLLM(model=model_name, temperature=temperature)

    def _load_template(self) -> Optional[Template]:
        """Load the Jinja template for relevance filtering."""
        # Find templates directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to hyper_jade root (src/agents/code_corrector -> hyper_jade)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        template_full_path = os.path.join(root_dir, "templates", self.template_path)
        
        try:
            with open(template_full_path, "r", encoding="utf-8") as f:
                return Template(f.read())
        except FileNotFoundError:
            logger.warning(f"Linter template not found at {template_full_path}")
            return None

    def run_linter(self, code: str) -> List[LintError]:
        """
        Run Ruff linter on the provided code.
        
        Args:
            code: The Python source code to lint
            
        Returns:
            List of LintError objects with detected issues
        """
        errors: List[LintError] = []
        
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", 
            suffix=".py", 
            delete=False,
            encoding="utf-8"
        ) as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name
        
        try:
            # Run ruff check with JSON output
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse JSON output (ruff exits with 1 if errors found, that's OK)
            if result.stdout:
                try:
                    ruff_output = json.loads(result.stdout)
                    for item in ruff_output:
                        error: LintError = {
                            "line": item.get("location", {}).get("row", 0),
                            "column": item.get("location", {}).get("column", 0),
                            "code": item.get("code", ""),
                            "message": item.get("message", ""),
                            "filename": temp_path,
                        }
                        errors.append(error)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse ruff output: {e}")
                    
        except subprocess.TimeoutExpired:
            logger.error("Ruff linter timed out")
        except FileNotFoundError:
            logger.error("Ruff not found. Please install ruff: pip install ruff")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return errors

    def _filter_by_category(self, errors: List[LintError]) -> List[LintError]:
        """
        Pre-filter errors based on category and exclusion rules.
        This happens BEFORE sending to LLM to save tokens.
        """
        filtered = []
        for error in errors:
            code = error.get("code", "")
            
            # Skip excluded codes
            if code in self.exclude_codes:
                continue
            
            # Check if code starts with an included category prefix
            category_match = any(
                code.startswith(cat) 
                for cat in self.include_categories
            )
            
            if category_match:
                filtered.append(error)
        
        return filtered

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and extract text response."""
        try:
            if isinstance(self.llm, (ChatOpenAI, ChatGoogleGenerativeAI)):
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                if hasattr(resp, "content"):
                    return str(resp.content).strip()
                return str(resp).strip()
            else:
                resp = self.llm.invoke(prompt)
                return str(resp).strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    @staticmethod
    def _add_line_numbers(code: str) -> str:
        """Add line numbers to code for better LLM context."""
        lines = code.split('\n')
        max_line_num = len(lines)
        width = len(str(max_line_num))
        numbered_lines = [
            f"{i+1:>{width}}: {line}" 
            for i, line in enumerate(lines)
        ]
        return '\n'.join(numbered_lines)

    @traceable(name="linter_filter_relevant_errors", run_type="chain")
    def filter_relevant_errors(
        self, 
        errors: List[LintError], 
        code: str
    ) -> List[LintError]:
        """
        Use LLM to filter only relevant errors.
        
        Args:
            errors: List of linter errors to evaluate
            code: The original student code (for context)
            
        Returns:
            Filtered list containing only relevant errors
        """
        if not self.template:
            logger.warning("No template loaded, returning all errors as relevant")
            return errors
        
        relevant_errors = []
        
        # Add line numbers to code for better context
        code_with_lines = self._add_line_numbers(code)
        
        for error in errors:
            # Render template with error context
            rendered_prompt = self.template.render(
                code=code_with_lines,
                error_line=error["line"],
                error_message=error["message"],
                error_code=error["code"],
            )
            
            # Ask LLM if error is relevant
            response = self._call_llm(rendered_prompt)
            
            # Parse response for RESULT tag
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response, re.DOTALL | re.IGNORECASE)
            if result_match:
                result = result_match.group(1).strip().upper()
                if "ERROR FOUND" in result or "RELEVANTE" in result:
                    # Store the full response for later use
                    error["llm_response"] = response
                    relevant_errors.append(error)
                    logger.debug(f"Error {error['code']} marked as relevant: {error['message']}")
                else:
                    logger.debug(f"Error {error['code']} filtered as irrelevant: {error['message']}")
            else:
                logger.warning(f"Could not parse LLM response for error {error['code']}, skipping")
        
        return relevant_errors

    def _deduplicate_errors(self, errors: List[LintError]) -> List[LintError]:
        """
        Deduplicate errors by grouping identical code+message combinations.
        Only keeps the first occurrence but aggregates line numbers.
        """
        seen = {}
        deduplicated = []
        
        for error in errors:
            # Create key from code and message (ignoring line numbers)
            key = (error["code"], error["message"])
            
            if key not in seen:
                # First occurrence - keep it
                seen[key] = error
                deduplicated.append(error)
            else:
                # Duplicate - just note it in logs
                logger.debug(f"Skipping duplicate error {error['code']} at line {error['line']}")
        
        return deduplicated

    @traceable(name="linter_generate_corrections", run_type="chain")
    def generate_linter_corrections(self, code: str) -> List[Correction]:
        """
        Main entry point: run linter, filter errors, generate corrections.
        
        Args:
            code: The student's Python code
            
        Returns:
            List of Correction objects for relevant linter errors
        """
        # Step 1: Run linter
        all_errors = self.run_linter(code)
        logger.info(f"Linter found {len(all_errors)} total errors")
        
        # Step 2: Pre-filter by category/exclusions
        category_filtered = self._filter_by_category(all_errors)
        logger.info(f"After category filter: {len(category_filtered)} errors")
        
        # Step 2.5: Deduplicate BEFORE LLM filtering to avoid duplicate LLM calls
        deduplicated = self._deduplicate_errors(category_filtered)
        logger.info(f"After deduplication: {len(deduplicated)} unique errors")
        
        # Step 3: LLM filtering for relevance (now on deduplicated errors)
        relevant_errors = self.filter_relevant_errors(deduplicated, code)
        logger.info(f"After LLM filter: {len(relevant_errors)} relevant errors")
        
        # Step 4: Generate corrections
        corrections: List[Correction] = []
        
        for error in relevant_errors:
            # Create a synthetic requirement for linter errors
            requirement: Requirement = {
                "requirement": f"El código no debe tener errores de linting: {error['code']}",
                "function": "",
                "type": PromptType.STYLISTIC
            }
            
            # Use the LLM response if available, otherwise create a simple message
            llm_response = error.get("llm_response", "")
            if llm_response:
                result_text = llm_response
            else:
                result_text = f"<RESULT>ERROR FOUND</RESULT>\n<EXPLANATION>Error en línea {error['line']}: {error['message']} (código: {error['code']})</EXPLANATION>"
            
            correction: Correction = {
                "requirement": requirement,
                "result": result_text
            }
            corrections.append(correction)
        
        return corrections
