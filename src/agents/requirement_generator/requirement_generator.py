#!/usr/bin/env python3
"""
Requirement Generator Agent (LangGraph single-pass)

This agent takes an assignment description (consigna) and generates individual
requirement files. Each requirement is saved as a separate .txt file.
Uses a minimal LangGraph with a single node.
"""

from __future__ import annotations
from typing import Dict, List, Any, TypedDict, Optional
from pathlib import Path
import os
import re
import logging
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.graph import StateGraph, END

from src.config import get_agent_config
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.models import PromptType, Requirement
from src.agents.utils.text_processors import clean_llm_response_tags

load_dotenv(override=False)

# Set up logger
logger = logging.getLogger(__name__)


def parse_requirements_from_xml_tags(
    content: str, max_requirements: int = 10
) -> List[Dict[str, str]]:
    """
    Parse requirements from XML format with numbered containers
    
    Expected format:
    <1>
    <type>requirement_presence</type>
    <function>es_par(num)</function>  <!-- optional -->
    <requirement>Specific description of the requirement</requirement>
    </1>

    Args:
        content: Cleaned content from LLM response
        max_requirements: Maximum number of requirements to return

    Returns:
        List of requirement dictionaries with keys: requirement, function (optional), type
    """
    requirements = []

    # Use regex to extract numbered requirement blocks: <1>...</1>, <2>...</2>, etc.
    pattern = r"<(\d+)>(.*?)</\1>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        raise ValueError("No XML requirements found in LLM response. Expected format: <1><type>...</type><requirement>...</requirement></1>")

    # Process each numbered requirement block
    for number, block_content in matches:
        requirement_dict = {}
        
        # Extract type (required)
        type_match = re.search(r"<type>(.*?)</type>", block_content, re.DOTALL)
        if not type_match:
            raise ValueError(f"Missing <type> tag in requirement {number}")
        requirement_dict["type"] = type_match.group(1).strip()
        
        # Extract function (optional)
        function_match = re.search(r"<function>(.*?)</function>", block_content, re.DOTALL)
        requirement_dict["function"] = function_match.group(1).strip() if function_match else ""
        
        # Extract requirement description (required)
        req_match = re.search(r"<requirement>(.*?)</requirement>", block_content, re.DOTALL)
        if not req_match:
            raise ValueError(f"Missing <requirement> tag in requirement {number}")
        requirement_dict["requirement"] = req_match.group(1).strip()
        
        requirements.append(requirement_dict)

    # Limit to max_requirements if more were generated
    if len(requirements) > max_requirements:
        logger.warning(
            f"Generated {len(requirements)} requirements, limiting to {max_requirements}"
        )
        requirements = requirements[:max_requirements]

    logger.info(f"Parsed {len(requirements)} requirements from XML response")

    return requirements




class RequirementGeneratorState(TypedDict):
    assignment: str
    requirements: List[Requirement]
    
    # Extra field for additional metadata/data that can be loaded any time
    extra: Optional[Dict[str, Any]]

class RequirementGeneratorAgent:
    """Agent that generates individual requirement files from an assignment description using LangGraph"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = get_agent_config(config, "requirement_generator")
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

    
    def generate_requirements(self, assignment: str) -> List[Requirement]:
        """
        Generate requirements from an assignment description using LangGraph
        
        Args:
            assignment: The assignment description text
            
        Returns:
            List of Requirement objects with requirement, function, and type fields
        """
        # Build and run the LangGraph
        app = self._build_graph()
        state_in: RequirementGeneratorState = {
            "assignment": assignment,
            "requirements": [],
            "extra": {}
        }
        result_state = app.invoke(state_in)
        
        return result_state.get("requirements", [])
    
    # -------------------------- LangGraph Node ------------------------------ #
    def _node_generate_requirements(self, state: RequirementGeneratorState) -> RequirementGeneratorState:
        """LangGraph node that generates requirements from assignment description"""
        assignment_description = state["assignment"]
        
        # Get allowed types from config (template keys)
        templates_cfg = self.config.get('agents', {}).get('prompt_generator', {}).get('templates', {})
        types = [t for t in PromptType if t.value in templates_cfg.keys()]

        template_name = self.agent_config.get("template")
        env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=select_autoescape(["jinja"])
        )
        template = env.get_template(template_name)
        prompt = template.render(
            assignment_description=assignment_description,
            types=types,
            max_requirements=self.agent_config.get("max_requirements", 10)
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw_content = getattr(response, "content", str(response)).strip()
        
        # Parse and extract requirements from LLM response using XML tags format
        requirements = self._parse_requirements_from_llm_response(raw_content)
        
        # Update state with generated requirements
        state["requirements"] = requirements
        return state
    
    def _parse_requirements_from_llm_response(self, raw_content: str) -> List[Requirement]:
        """Parse and extract requirements from LLM response using XML tags format"""
        # Clean the content by removing think tags and explanations
        content = clean_llm_response_tags(raw_content)

        # Extract content from code blocks if present
        if "```" in content:
            # Remove markdown code blocks
            lines = content.split("\n")
            content_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    content_lines.append(line)
            content = "\n".join(content_lines).strip()

        # Debug: Log the extracted content
        logger.info(f"Extracted content for parsing: {repr(content)}")

        # Check if content is empty
        if not content.strip():
            raise ValueError("LLM returned empty response")

        # Parse requirements using the XML tags format
        max_requirements = self.agent_config.get("max_requirements", 10)
        requirement_dicts = parse_requirements_from_xml_tags(content, max_requirements)

        # Convert to Requirement TypedDict with proper PromptType enum
        requirements: List[Requirement] = []
        for req_dict in requirement_dicts:
            # Convert type string to PromptType enum
            type_value = req_dict.get("type", "unknown")
            prompt_type = None
            
            # Try to find matching PromptType
            for pt in PromptType:
                if pt.value == type_value:
                    prompt_type = pt
                    break
            
            # If no match found, use a default or create one
            if prompt_type is None:
                # Try to find a reasonable default or use the first available type
                try:
                    prompt_type = list(PromptType)[0]  # Use the first available type as fallback
                except:
                    prompt_type = type_value  # Keep as string if PromptType enum is not available
            
            requirement: Requirement = {
                "requirement": req_dict.get("requirement", ""),
                "function": req_dict.get("function", ""),
                "type": prompt_type
            }
            requirements.append(requirement)

        logger.info(f"Generated {len(requirements)} requirements")
        return requirements
    
    # -------------------------- Graph Builder ------------------------------- #
    def _build_graph(self):
        """Build a minimal single-pass LangGraph for requirement generation."""
        graph = StateGraph(RequirementGeneratorState)
        graph.add_node("generate_requirements", self._node_generate_requirements)
        graph.set_entry_point("generate_requirements")
        graph.add_edge("generate_requirements", END)
        return graph.compile()
    

