#!/usr/bin/env python3
"""
Error Locator Agent

This agent identifies specific code fragments (with line numbers) where errors occur.
It runs as a separate step after code correction, with its own LangSmith trace.
"""

from __future__ import annotations
from typing import Dict, Any, List
import re
import os
import logging

from dotenv import load_dotenv, dotenv_values
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langsmith import traceable

from src.config import get_agent_config
from src.models import Correction, ErrorLocation

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)
DOTENV = dotenv_values()

# Template directory
TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "templates", "error_locator"
)


class ErrorLocatorAgent:
    """
    Agent that locates specific code fragments where errors occur.
    Runs as a separate LangSmith-traced step after code correction.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.agent_config = get_agent_config(self.config, "error_locator")
        self.llm = self._setup_llm()
        self.template = self._load_template()

    def _setup_llm(self):
        """Setup the LLM provider based on configuration."""
        provider = str(self.agent_config.get("provider", "google")).lower().strip()
        model_name = self.agent_config.get("model_name", "gemini-2.0-flash")
        temperature = float(self.agent_config.get("temperature", 0.1))

        if provider == "openai":
            api_key = self.agent_config.get("api_key") or DOTENV.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            return ChatOpenAI(
                model=model_name, temperature=temperature, api_key=api_key
            )

        if provider in ("gemini", "google", "google-genai"):
            api_key = self.agent_config.get("api_key") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            return ChatGoogleGenerativeAI(
                model=model_name, temperature=temperature, google_api_key=api_key
            )

        return OllamaLLM(model=model_name, temperature=temperature)

    def _load_template(self) -> Template:
        """Load the Jinja template for error location."""
        template_path = os.path.join(TEMPLATE_DIR, "error_locator.jinja")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return Template(f.read())
        else:
            # Fallback inline template
            return Template("""
You are an expert code analyzer. Given the error description and code with line numbers,
identify the exact code fragment(s) where the error occurs.

Error: {{ explanation }}
Requirement: {{ requirement }}

Code:
{{ code_with_lines }}

Output format:
<LOCATION>
<LINES>[line_range]</LINES>
<FRAGMENT>[code]</FRAGMENT>
</LOCATION>
<EXPLANATION>[why]</EXPLANATION>
""")

    @staticmethod
    def _add_line_numbers(code: str) -> str:
        """Add line numbers to code for easier reference."""
        lines = code.split("\n")
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"{i:4d}: {line}")
        return "\n".join(numbered_lines)

    @staticmethod
    def _extract_explanation_from_result(result: str) -> str:
        """Extract the explanation from the correction result."""
        # Match <EXPLANATION>...</EXPLANATION>
        match = re.search(
            r"<EXPLANATION>(.*?)</EXPLANATION>", result, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        # Fallback: return the whole result without the RESULT tag
        cleaned = re.sub(
            r"<RESULT>.*?</RESULT>", "", result, flags=re.DOTALL | re.IGNORECASE
        )
        return cleaned.strip()

    @staticmethod
    def _is_error_found(result: str) -> bool:
        """Check if the correction result indicates an error was found."""
        return "ERROR FOUND" in result.upper()

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        try:
            if isinstance(self.llm, (ChatOpenAI, ChatGoogleGenerativeAI)):
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                if hasattr(resp, "content"):
                    content = resp.content
                    if isinstance(content, list):
                        return "".join(
                            p.get("text", str(p)) if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    return str(content)
                return str(resp)
            else:
                return str(self.llm.invoke(prompt))
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def _parse_locations(self, llm_response: str) -> tuple[List[ErrorLocation], str]:
        """Parse the LLM response to extract locations and explanation."""
        locations: List[ErrorLocation] = []

        # Find all <LOCATION>...</LOCATION> blocks
        location_pattern = r"<LOCATION>(.*?)</LOCATION>"
        location_matches = re.findall(
            location_pattern, llm_response, re.DOTALL | re.IGNORECASE
        )

        for loc_content in location_matches:
            # Extract LINES
            lines_match = re.search(
                r"<LINES>(.*?)</LINES>", loc_content, re.DOTALL | re.IGNORECASE
            )
            # Extract FRAGMENT
            fragment_match = re.search(
                r"<FRAGMENT>(.*?)</FRAGMENT>", loc_content, re.DOTALL | re.IGNORECASE
            )

            if lines_match and fragment_match:
                locations.append(
                    {
                        "lines": lines_match.group(1).strip(),
                        "fragment": fragment_match.group(1).strip(),
                    }
                )

        # Extract explanation (outside LOCATION blocks)
        expl_match = re.search(
            r"<EXPLANATION>(.*?)</EXPLANATION>", llm_response, re.DOTALL | re.IGNORECASE
        )
        explanation = expl_match.group(1).strip() if expl_match else ""

        return locations, explanation

    def _locate_single_error(
        self, correction: Correction, code_with_lines: str
    ) -> Correction:
        """Locate the error for a single correction."""
        # Extract info from correction
        requirement = correction.get("requirement", {})
        requirement_text = requirement.get("requirement", "")
        result = correction.get("result", "")
        explanation = self._extract_explanation_from_result(result)

        # Render the template
        prompt = self.template.render(
            requirement=requirement_text,
            explanation=explanation,
            code_with_lines=code_with_lines,
        )

        # Call LLM
        llm_response = self._call_llm(prompt)

        # Parse response
        locations, location_explanation = self._parse_locations(llm_response)

        # Return updated correction
        updated_correction: Correction = {
            **correction,
            "locations": locations if locations else None,
            "location_explanation": location_explanation
            if location_explanation
            else None,
        }

        return updated_correction

    @traceable(name="error_locator_batch", run_type="chain")
    def locate_errors_batch(
        self, corrections: List[Correction], code: str
    ) -> List[Correction]:
        """
        Locate error fragments for all ERROR FOUND corrections.

        Args:
            corrections: List of Correction objects from code corrector
            code: The student's source code (without line numbers)

        Returns:
            List of Correction objects with 'locations' and 'location_explanation' populated
            for ERROR FOUND corrections.
        """
        # Add line numbers to code
        code_with_lines = self._add_line_numbers(code)

        updated_corrections = []

        for correction in corrections:
            result = correction.get("result", "")

            if self._is_error_found(result):
                # Use original index from correction for proper trace matching
                original_idx = correction.get("_original_index", 0)

                # Locate the error with trace named by original correction index
                try:
                    updated = self._locate_with_trace(
                        correction, code_with_lines, original_idx
                    )
                    updated_corrections.append(updated)
                    logger.info(
                        f"Located {len(updated.get('locations', []))} fragments for error"
                    )
                except Exception as e:
                    logger.error(f"Failed to locate error: {e}")
                    updated_corrections.append(correction)
            else:
                # NO ERROR - keep as is without locations
                updated_corrections.append(correction)

        return updated_corrections

    def _locate_with_trace(
        self, correction: Correction, code_with_lines: str, error_index: int
    ) -> Correction:
        """Wrapper to create a named trace for each error."""

        # Create traceable function dynamically with the requirement name
        @traceable(name=f"locate_error_{error_index}", run_type="llm")
        def _inner():
            return self._locate_single_error(correction, code_with_lines)

        return _inner()

    def locate_error(self, correction: Correction, code: str) -> Correction:
        """
        Locate error for a single correction.

        Args:
            correction: Single Correction object
            code: The student's source code

        Returns:
            Updated Correction with locations if ERROR FOUND
        """
        results = self.locate_errors_batch([correction], code)
        return results[0] if results else correction
