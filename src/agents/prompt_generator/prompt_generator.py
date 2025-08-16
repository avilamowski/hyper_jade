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
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

# Import MLflow logger lazily to avoid circular imports
def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed"""
    try:
        from src.core.mlflow_utils import mlflow_logger
        return mlflow_logger
    except ImportError:
        # Return a dummy logger if MLflow is not available
        class DummyLogger:
            def start_run(self, *args, **kwargs): return None
            def end_run(self): pass
            def log_metric(self, *args, **kwargs): pass
            def log_metrics(self, *args, **kwargs): pass
            def log_artifact(self, *args, **kwargs): pass
            def log_artifacts(self, *args, **kwargs): pass
            def log_param(self, *args, **kwargs): pass
            def log_params(self, *args, **kwargs): pass
            def log_text(self, *args, **kwargs): pass
            def log_prompt_metrics(self, *args, **kwargs): pass
            def log_prompt(self, *args, **kwargs): pass
            def log_trace_step(self, *args, **kwargs): pass
            def log_agent_input_output(self, *args, **kwargs): pass
        return DummyLogger()

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
        # Start MLflow run for prompt generation
        mlflow_logger = get_mlflow_logger()
        run_id = mlflow_logger.start_run(
            run_name="prompt_generation",
            tags={
                "agent": "prompt_generator",
                "requirement_file": Path(requirement_file_path).name,
                "assignment_file": Path(assignment_file_path).name,
                "output_file": Path(output_file_path).name
            }
        )
        
        start_time = time.time()
        logger.info(f"Generating prompt from requirement: {requirement_file_path}")
        
        # Log parameters
        mlflow_logger.log_params({
            "requirement_file_path": requirement_file_path,
            "assignment_file_path": assignment_file_path,
            "output_file_path": output_file_path,
            "model_name": self.config.get("model_name", "unknown"),
            "provider": self.config.get("provider", "unknown"),
            "temperature": self.config.get("temperature", 0.1)
        })
        
        try:
            # Read the requirement
            with open(requirement_file_path, 'r', encoding='utf-8') as f:
                requirement = f.read().strip()
            
            # Read the assignment description
            with open(assignment_file_path, 'r', encoding='utf-8') as f:
                assignment_description = f.read().strip()
            
            # Log input files as artifacts
            mlflow_logger.log_text(requirement, "input_requirement.txt")
            mlflow_logger.log_text(assignment_description, "input_assignment.txt")
            
            # Log trace step: Reading inputs
            mlflow_logger.log_trace_step("read_inputs", {
                "requirement_file": requirement_file_path,
                "assignment_file": assignment_file_path,
                "requirement_length": len(requirement),
                "assignment_length": len(assignment_description)
            }, step_number=1)
            
            # Generate Jinja2 template prompt
            llm_start_time = time.time()
            jinja_template = self._generate_jinja_template(requirement, assignment_description)
            llm_time = time.time() - llm_start_time
            
            # Log trace step: Template generation
            mlflow_logger.log_trace_step("template_generation", {
                "generation_time": llm_time,
                "template_length": len(jinja_template)
            }, step_number=2)
            
            # Log LLM performance metrics
            mlflow_logger.log_metrics({
                "llm_generation_time_seconds": llm_time,
                "template_length_chars": len(jinja_template),
                "template_length_lines": len(jinja_template.split('\n'))
            })
            
            # Log agent I/O
            mlflow_logger.log_agent_input_output("prompt_generator", {
                "requirement": requirement,
                "assignment_description": assignment_description
            }, {
                "jinja_template": jinja_template,
                "generation_time": llm_time
            })
            
            # Save the template
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(jinja_template)
            
            # Log the generated template as artifact and metrics
            prompt_name = Path(output_file_path).stem
            mlflow_logger.log_text(jinja_template, f"generated_templates/{Path(output_file_path).name}")
            mlflow_logger.log_prompt_metrics(jinja_template, prompt_name)
            
            total_time = time.time() - start_time
            
            # Log final metrics
            mlflow_logger.log_metrics({
                "total_generation_time_seconds": total_time,
                "template_generation_rate": 1.0 / total_time if total_time > 0 else 0
            })
            
            logger.info(f"Generated Jinja2 template: {output_path}")
            return str(output_path)
            
        except Exception as e:
            # Log error metrics
            mlflow_logger.log_metric("error_occurred", 1.0)
            mlflow_logger.log_text(str(e), "error_log.txt")
            logger.error(f"Error generating prompt: {e}")
            raise
        finally:
            mlflow_logger.end_run()
    
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
        
        # Log the prompt being sent to LLM
        mlflow_logger = get_mlflow_logger()
        mlflow_logger.log_prompt(prompt, "prompt_generation", "llm_prompt")
        
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
