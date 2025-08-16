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
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from src.config import get_agent_config

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
            def log_requirement_metrics(self, *args, **kwargs): pass
            def log_prompt(self, *args, **kwargs): pass
            def log_trace_step(self, *args, **kwargs): pass
            def log_agent_input_output(self, *args, **kwargs): pass
        return DummyLogger()

logger = logging.getLogger(__name__)

class RequirementGeneratorAgent:
    """Agent that generates individual requirement files from an assignment description"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Get agent-specific configuration
        self.agent_config = get_agent_config(config, 'requirement_generator')
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on agent-specific configuration"""
        if self.agent_config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.agent_config.get("model_name", "gpt-4"),
                temperature=self.agent_config.get("temperature", 0.1)
            )
        else:
            return OllamaLLM(
                model=self.agent_config.get("model_name", "qwen2.5:7b"),
                temperature=self.agent_config.get("temperature", 0.1)
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
        # Start MLflow run for requirement generation
        mlflow_logger = get_mlflow_logger()
        run_id = mlflow_logger.start_run(
            run_name="requirement_generation",
            tags={
                "agent": "requirement_generator",
                "assignment_file": Path(assignment_file_path).name,
                "output_directory": output_directory
            }
        )
        
        start_time = time.time()
        logger.info(f"Generating requirements from assignment: {assignment_file_path}")
        
        # Log parameters
        mlflow_logger.log_params({
            "assignment_file_path": assignment_file_path,
            "output_directory": output_directory,
            "model_name": self.agent_config.get("model_name", "unknown"),
            "provider": self.agent_config.get("provider", "unknown"),
            "temperature": self.agent_config.get("temperature", 0.1)
        })
        
        try:
            # Read the assignment description
            with open(assignment_file_path, 'r', encoding='utf-8') as f:
                assignment_description = f.read().strip()
            
            # Log assignment content as artifact
            mlflow_logger.log_text(assignment_description, "assignment_description.txt")
            
            # Log trace step: Reading assignment
            mlflow_logger.log_trace_step("read_assignment", {
                "assignment_file": assignment_file_path,
                "assignment_length": len(assignment_description)
            }, step_number=1)
            
            # Generate requirements using LLM
            llm_start_time = time.time()
            requirements = self._generate_requirements_list(assignment_description)
            llm_time = time.time() - llm_start_time
            
            # Log LLM performance metrics
            mlflow_logger.log_metrics({
                "llm_generation_time_seconds": llm_time,
                "requirements_generated": len(requirements)
            })
            
            # Log agent I/O
            mlflow_logger.log_agent_input_output("requirement_generator", {
                "assignment_description": assignment_description,
                "model_name": self.agent_config.get("model_name", "unknown")
            }, {
                "requirements": requirements,
                "generation_time": llm_time
            })
            
            # Log trace step: Requirements generated
            mlflow_logger.log_trace_step("requirements_generated", {
                "count": len(requirements),
                "generation_time": llm_time
            }, step_number=3)
            
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
                
                # Log each requirement as artifact and metrics
                mlflow_logger.log_text(requirement, f"requirements/requirement_{i:02d}.txt")
                mlflow_logger.log_requirement_metrics(requirement, i)
            
            total_time = time.time() - start_time
            
            # Log final metrics
            mlflow_logger.log_metrics({
                "total_generation_time_seconds": total_time,
                "requirements_per_second": len(requirements) / total_time if total_time > 0 else 0
            })
            
            # Log the entire requirements directory as artifacts
            mlflow_logger.log_artifacts(output_directory, "requirements")
            
            # Log trace step: Final completion
            mlflow_logger.log_trace_step("requirement_generation_complete", {
                "files_generated": len(requirement_files),
                "total_time": total_time
            }, step_number=4)
            
            logger.info(f"Generated {len(requirement_files)} requirement files")
            return requirement_files
            
        except Exception as e:
            # Log error metrics
            mlflow_logger.log_metric("error_occurred", 1.0)
            mlflow_logger.log_text(str(e), "error_log.txt")
            logger.error(f"Error generating requirements: {e}")
            raise
        finally:
            mlflow_logger.end_run()
    
    def _generate_requirements_list(self, assignment_description: str) -> List[str]:
        """Generate a list of individual requirements from assignment description"""
        
        prompt = f"""Analyze the following programming assignment and generate a list of individual requirements.

INSTRUCTIONS:
1. Identify all aspects that should be evaluated in the student's code
2. Break down the assignment into specific requirements
3. Each requirement should be independent
4. Include requirements for functionality, code quality, error handling, etc.
5. Each requirement should be clear and specific
6. Only include up to 10 requirements

OUTPUT FORMAT:
Generate a list with each requirement starting with a dash (-):

- Requirement 1: Specific description of the first requirement
- Requirement 2: Specific description of the second requirement
- Requirement 3: Specific description of the third requirement
...

Return only the list with dashes, no additional text.

ASSIGNMENT:
{assignment_description}
"""
        
        # Log the prompt being sent to LLM
        mlflow_logger = get_mlflow_logger()
        mlflow_logger.log_prompt(prompt, "requirement_generation", "llm_prompt")
        
        # Log trace step: LLM generation
        mlflow_logger.log_trace_step("llm_generation", {
            "prompt_length": len(prompt),
            "model_name": self.agent_config.get("model_name", "unknown")
        }, step_number=2)
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = str(response).strip()
        
        # Debug: Log the raw response
        logger.info(f"Raw LLM response: {repr(content)}")
        
        # Extract content from code blocks if present
        if "```" in content:
            # Remove markdown code blocks
            lines = content.split('\n')
            content_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    content_lines.append(line)
            content = '\n'.join(content_lines).strip()
        
        # Debug: Log the extracted content
        logger.info(f"Extracted content for parsing: {repr(content)}")
        
        # Check if content is empty
        if not content.strip():
            raise ValueError("LLM returned empty response")
        
        # Parse requirements from dash-separated list
        requirements = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                # Remove the dash and any leading whitespace
                requirement = line[1:].strip()
                if requirement:  # Only add non-empty requirements
                    requirements.append(requirement)
            elif line and not line.startswith('#') and not line.startswith('*'):
                # If line doesn't start with dash but contains content, 
                # it might be a continuation or a requirement without dash
                if line and not line.startswith('Requirement'):
                    # Try to clean it up and add it
                    clean_line = line.strip()
                    if clean_line:
                        requirements.append(clean_line)
        
        # If no requirements found with dashes, try to extract from numbered list
        if not requirements:
            for line in lines:
                line = line.strip()
                if line and (line.startswith('Requirement') or 
                           any(line.startswith(f"{i}.") for i in range(1, 11)) or
                           any(line.startswith(f"{i})") for i in range(1, 11))):
                    requirements.append(line)
        
        # Validate that we got requirements
        if not requirements:
            raise ValueError("No requirements found in LLM response")
        
        logger.info(f"Parsed {len(requirements)} requirements from response")
        
        logger.info(f"Generated {len(requirements)} requirements")
        return requirements
