#!/usr/bin/env python3
"""
Requirement Generator Agent

This agent takes an assignment description (consigna) and generates individual
requirement files. Each requirement is saved as a separate .txt file.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class RequirementGeneratorAgent:
    """Agent that generates individual requirement files from an assignment description"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on configuration"""
        if self.config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.config.get("model_name", "gpt-4"),
                temperature=self.config.get("temperature", 0.1)
            )
        else:
            return OllamaLLM(
                model=self.config.get("model_name", "qwen2.5:7b"),
                temperature=self.config.get("temperature", 0.1)
            )
    
    def generate_requirements(
        self,
        assignment_file_path: str,
        output_directory: str
    ) -> List[str]:
        """
        Generate individual requirement files from an assignment description
        
        Args:
            assignment_file_path: Path to the assignment description (.txt file)
            output_directory: Directory where requirement files will be saved
            
        Returns:
            List of paths to the generated requirement files
        """
        logger.info(f"Generating requirements from assignment: {assignment_file_path}")
        
        # Read the assignment description
        with open(assignment_file_path, 'r', encoding='utf-8') as f:
            assignment_description = f.read().strip()
        
        # Generate requirements using LLM
        requirements = self._generate_requirements_list(assignment_description)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each requirement as a separate file
        requirement_files = []
        for i, requirement in enumerate(requirements, 1):
            filename = f"requirement_{i:02d}.txt"
            file_path = output_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(requirement)
            requirement_files.append(str(file_path))
            logger.info(f"Generated requirement file: {file_path}")
        
        logger.info(f"Generated {len(requirement_files)} requirement files")
        return requirement_files
    
    def _generate_requirements_list(self, assignment_description: str) -> List[str]:
        """Generate a list of individual requirements from assignment description"""
        
        prompt = f"""Analyze the following programming assignment and generate a list of individual requirements.

ASSIGNMENT:
{assignment_description}

INSTRUCTIONS:
1. Identify all aspects that should be evaluated in the student's code
2. Break down the assignment into specific and measurable requirements
3. Each requirement should be independent and evaluable separately
4. Include requirements for functionality, code quality, error handling, etc.
5. Each requirement should be clear and specific

OUTPUT FORMAT:
Generate a JSON list with each requirement as a separate element:

[
    "Requirement 1: Specific description of the first requirement",
    "Requirement 2: Specific description of the second requirement",
    "Requirement 3: Specific description of the third requirement",
    ...
]

Each requirement must be:
- Specific and measurable
- Independent of other requirements
- Clear about what the code should do
- Objectively evaluable

Return only the JSON, no additional text.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = str(response).strip()
        
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        requirements = json.loads(content)
        
        # Validate that we got a list of strings
        if not isinstance(requirements, list):
            raise ValueError("Expected list of requirements")
        
        # Ensure each requirement is a string
        requirements = [str(req) for req in requirements if req]
        
        logger.info(f"Generated {len(requirements)} requirements")
        return requirements
