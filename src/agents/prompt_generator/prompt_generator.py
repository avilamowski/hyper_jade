#!/usr/bin/env python3
"""
Prompt Generator Agent

This agent takes a single requirement file (.txt) and an assignment description,
then generates a Jinja2 template prompt (.jinja) that will be used by the code 
corrector agent to analyze code against that specific requirement.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class PromptGeneratorAgent:
    """Agent that generates Jinja2 template prompts from individual requirements"""
    
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
    
    def generate_prompt(
        self,
        requirement_file_path: str,
        assignment_file_path: str,
        output_file_path: str
    ) -> str:
        """
        Generate a Jinja2 template prompt from a single requirement file and assignment description
        
        Args:
            requirement_file_path: Path to the requirement file (.txt)
            assignment_file_path: Path to the assignment description file (.txt)
            output_file_path: Path where the Jinja2 template will be saved
            
        Returns:
            Path to the generated Jinja2 template file
        """
        logger.info(f"Generating prompt from requirement: {requirement_file_path}")
        
        # Read the requirement
        with open(requirement_file_path, 'r', encoding='utf-8') as f:
            requirement = f.read().strip()
        
        # Read the assignment description
        with open(assignment_file_path, 'r', encoding='utf-8') as f:
            assignment_description = f.read().strip()
        
        # Generate Jinja2 template prompt
        jinja_template = self._generate_jinja_template(requirement, assignment_description)
        
        # Save the template
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(jinja_template)
        
        logger.info(f"Generated Jinja2 template: {output_path}")
        return str(output_path)
    
    def _generate_jinja_template(self, requirement: str, assignment_description: str) -> str:
        """Generate a Jinja2 template prompt from a requirement and assignment description"""
        
        prompt = f"""You are creating a static code analyzer template. This template will be used to analyze student code.

ASSIGNMENT DESCRIPTION:
{assignment_description}

REQUIREMENT:
{requirement}

Your task is to create a Jinja2 template that will analyze if student code meets this specific requirement.

The template should follow this EXACT structure:

```
You are a static code analyzer.

Task:
- Determine if the code [describe what to check based on the requirement]

Definition:
- [Define what constitutes valid/invalid code for this requirement]

Examples that should return <RESULT>YES</RESULT>:
1. [Example of valid code that meets the requirement]
2. [Another example of valid code]

Examples that should return <RESULT>NO</RESULT>:
1. [Example of invalid code that does not meet the requirement]
2. [Another example of invalid code]

Instructions:
- Return YES if the code [meets the requirement].
- Return NO otherwise.

Output format (strict and exact):
<RESULT>YES</RESULT>
or
<RESULT>NO</RESULT>

Code to analyze:
{{{{ code }}}}
```

IMPORTANT: 
- This is a TEMPLATE for analysis, NOT code to implement
- Use "{{{{ code }}}}" (with double braces) for the Jinja2 variable
- The template will be filled with student code later
- Focus on the specific requirement given
- Provide clear examples of what passes and what fails

Generate only the template, no explanations or additional text.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        template = str(response).strip()
        
        # Clean up the response if it contains markdown formatting
        if "```jinja" in template:
            template = template.split("```jinja")[1].split("```")[0].strip()
        elif "```" in template:
            template = template.split("```")[1].strip()
        
        # Ensure the template has the basic structure - be more flexible with validation
        if "{{ code }}" not in template and "{{code}}" not in template:
            # Try to fix common issues
            template = template.replace("{{ student_code }}", "{{ code }}")
            template = template.replace("{{code}}", "{{ code }}")
            
            # If still not found, add it at the end
            if "{{ code }}" not in template:
                template += "\n\nCode to analyze:\n{{ code }}"
        
        logger.info("Generated Jinja2 template successfully")
        return template
