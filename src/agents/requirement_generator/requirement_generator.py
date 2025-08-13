#!/usr/bin/env python3
"""
Requirement Generator Agent

This agent takes an assignment description and generates a comprehensive rubric
with specific requirements for evaluation. It uses LLM to identify important
aspects that should be considered when grading student code.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

@dataclass
class RubricItem:
    """Represents a single rubric item"""
    id: str
    title: str
    description: str
    criteria: List[str]

@dataclass
class Rubric:
    """Complete rubric for an assignment"""
    title: str
    description: str
    items: List[RubricItem]
    programming_language: str
    difficulty_level: str = "beginner"

class RequirementGeneratorAgent:
    """Agent that generates rubrics from assignment descriptions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on configuration"""
        if self.config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.config.get("model_name", "gpt-4")
            )
        else:
            return OllamaLLM(
                model=self.config.get("model_name", "qwen2.5:7b")
            )
    
    def generate_rubric(
        self,
        assignment_description: str,
        programming_language: str = "python"
    ) -> Rubric:
        """Generate a comprehensive rubric from assignment description"""
        logger.info("Generating rubric from assignment")
        
        prompt = f"""Generate a comprehensive rubric for the following programming assignment.

ASSIGNMENT:
{assignment_description}

PROGRAMMING LANGUAGE: {programming_language}

Generate a rubric with the following structure in JSON format:
{{
    "title": "Rubric for [Assignment Name]",
    "description": "Brief description of what this rubric evaluates",
    "total_score": 100,
    "programming_language": "{programming_language}",
    "difficulty_level": "beginner",
    "items": [
        {{
            "id": "correctness",
            "title": "Correctness",
            "description": "Does the code produce correct results?",
            "criteria": [
                "Produces expected output for given inputs",
                "Handles edge cases appropriately",
                "No logical errors in the implementation"
            ]
        }},
        {{
            "id": "code_quality",
            "title": "Code Quality",
            "description": "Is the code well-structured and readable?",
            "criteria": [
                "Clear variable and function names",
                "Proper indentation and formatting",
                "Logical code organization"
            ]
        }},
        {{
            "id": "documentation",
            "title": "Documentation",
            "description": "Is the code properly documented?",
            "criteria": [
                "Functions have clear docstrings",
                "Comments explain complex logic",
                "Code is self-explanatory"
            ]
        }},
        {{
            "id": "error_handling",
            "title": "Error Handling",
            "description": "Does the code handle errors gracefully?",
            "criteria": [
                "Validates input parameters",
                "Handles edge cases (empty lists, null values, etc.)",
                "Provides meaningful error messages"
            ]
        }}
    ]
}}

Focus on beginner-level expectations and ensure all criteria are specific and measurable.
Return only the JSON, no additional text.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = str(response).strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            rubric_data = json.loads(content)
            
            # Convert to Rubric object
            items = []
            for item_data in rubric_data["items"]:
                item = RubricItem(
                    id=item_data["id"],
                    title=item_data["title"],
                    description=item_data["description"],
                    criteria=item_data["criteria"]
                )
                items.append(item)
            
            rubric = Rubric(
                title=rubric_data["title"],
                description=rubric_data["description"],
                items=items,
                programming_language=rubric_data["programming_language"],
                difficulty_level=rubric_data.get("difficulty_level", "beginner")
            )
            
            logger.info(f"Generated rubric with {len(items)} items")
            return rubric
            
        except Exception as e:
            logger.error(f"Failed to generate rubric: {e}")
            # Return a fallback rubric
            return self._create_fallback_rubric(assignment_description, programming_language)
    
    def _create_fallback_rubric(self, assignment_description: str, programming_language: str) -> Rubric:
        """Create a fallback rubric when generation fails"""
        return Rubric(
            title=f"Basic Rubric for {assignment_description[:50]}...",
            description="Basic rubric for evaluation",
            programming_language=programming_language,
            difficulty_level="beginner",
            items=[
                RubricItem(
                    id="basic_functionality",
                    title="Basic Functionality",
                    description="Code runs without errors",
                    criteria=["Code executes without syntax errors", "Produces some output"]
                ),
                RubricItem(
                    id="code_quality",
                    title="Code Quality",
                    description="Code is readable and well-structured",
                    criteria=["Clear variable names", "Proper formatting"]
                ),
                RubricItem(
                    id="documentation",
                    title="Documentation",
                    description="Code has basic documentation",
                    criteria=["Functions have docstrings", "Code is self-explanatory"]
                )
            ]
        )
