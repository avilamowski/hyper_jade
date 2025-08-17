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
import os
from dotenv import load_dotenv


import sys
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from src.config import get_agent_config
from jinja2 import Environment, FileSystemLoader, select_autoescape
from src.agents.utils.agent_evaluator import AgentEvaluator


load_dotenv(override=False)
# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import MLflow logger lazily to avoid circular imports
def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed"""
    try:
        from src.core.mlflow_utils import mlflow_logger
        return mlflow_logger
    except ImportError:
        logger.warning("MLflow not available - logging will be disabled")
        return None

def safe_log_call(logger_instance, method_name, *args, **kwargs):
    """Safely call a logging method, doing nothing if logger is None"""
    if logger_instance is not None and hasattr(logger_instance, method_name):
        try:
            method = getattr(logger_instance, method_name)
            method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to call {method_name}: {e}")


class RequirementGeneratorAgent:
    """Agent that generates individual requirement files from an assignment description"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Get agent-specific configuration
        self.agent_config = get_agent_config(config, 'requirement_generator')
        self.llm = self._setup_llm()
        
        # Initialize agent evaluator if enabled
        self.evaluator = None
        # Disable automatic evaluation by default - use standalone evaluator instead
        # if config.get('agents', {}).get('agent_evaluator', {}).get('enabled', False):
        #     self.evaluator = AgentEvaluator(config)
    
    def _setup_llm(self):
        """Setup LLM based on agent-specific configuration"""
        provider = self.agent_config.get("provider")
        model_name = self.agent_config.get("model_name", "gpt-4o-mini")
        temperature = self.agent_config.get("temperature", 0.1)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not found. Please add it to your .env file or environment variables."
                )
            base_url = os.getenv("OPENAI_BASE_URL")  # optional
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url if base_url else None,
            )
        else:
            return OllamaLLM(
                model=model_name or "qwen2.5:7b",
                temperature=temperature
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
        safe_log_call(mlflow_logger, "start_run",
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
        safe_log_call(mlflow_logger, "log_params", {
            "assignment_file_path": assignment_file_path,
            "output_directory": output_directory,
            "model_name": self.agent_config.get("model_name", "unknown"),
            "provider": self.agent_config.get("provider", "unknown"),
            "temperature": self.agent_config.get("temperature", 0.1)
        })
        
        try:
            # Read the assignment description
            logger.info(f"Opening file")
            with open(assignment_file_path, 'r', encoding='utf-8') as f:
                assignment_description = f.read().strip()
            
            # Log assignment content as artifact
            safe_log_call(mlflow_logger, "log_text", assignment_description, "assignment_description.txt")
            
            # Log trace step: Reading assignment
            safe_log_call(mlflow_logger, "log_trace_step", "read_assignment", {
                "assignment_file": assignment_file_path,
                "assignment_length": len(assignment_description)
            }, step_number=1)
            
            logger.info("Generating requirements using LLM")
            # Generate requirements using LLM
            llm_start_time = time.time()
            requirements = self._generate_requirements_list(assignment_description)
            llm_time = time.time() - llm_start_time
            
            # Log LLM performance metrics
            safe_log_call(mlflow_logger, "log_metrics", {
                "llm_generation_time_seconds": llm_time,
                "requirements_generated": len(requirements)
            })
            
            # Log agent I/O
            safe_log_call(mlflow_logger, "log_agent_input_output", "requirement_generator", {
                "assignment_description": assignment_description,
                "model_name": self.agent_config.get("model_name", "unknown")
            }, {
                "requirements": requirements,
                "generation_time": llm_time
            })
            
            # Log trace step: Requirements generated
            safe_log_call(mlflow_logger, "log_trace_step", "requirements_generated", {
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
                safe_log_call(mlflow_logger, "log_text", requirement, f"requirements/requirement_{i:02d}.txt")
                safe_log_call(mlflow_logger, "log_requirement_metrics", requirement, i)
            
            total_time = time.time() - start_time
            
            # Log final metrics
            safe_log_call(mlflow_logger, "log_metrics", {
                "total_generation_time_seconds": total_time,
                "requirements_per_second": len(requirements) / total_time if total_time > 0 else 0
            })
            
            # Log the entire requirements directory as artifacts
            safe_log_call(mlflow_logger, "log_artifacts", output_directory, "requirements")
            
            # Evaluate agent output if evaluator is enabled
            # Note: Evaluation is now handled by the standalone evaluator
            # if self.evaluator:
            #     try:
            #         logger.info("Evaluating requirement generator output...")
            #         evaluation_result = self.evaluator.evaluate_requirement_generator(
            #             assignment_description,
            #             requirements,
            #             output_directory
            #         )
            #         
            #         # Log evaluation metrics
            #         safe_log_call(mlflow_logger, "log_agent_evaluation_metrics", "requirement_generator", evaluation_result)
            #         
            #         logger.info(f"Requirement generator evaluation completed. Overall score: {evaluation_result.get('overall_score', 'N/A')}")
            #         
            #     except Exception as eval_error:
            #         logger.warning(f"Error during requirement generator evaluation: {eval_error}")
            
            # Log trace step: Final completion
            safe_log_call(mlflow_logger, "log_trace_step", "requirement_generation_complete", {
                "files_generated": len(requirement_files),
                "total_time": total_time,
                "evaluation_performed": False  # Evaluation now handled by standalone evaluator
            }, step_number=4)
            
            logger.info(f"Generated {len(requirement_files)} requirement files")
            return requirement_files
            
        except Exception as e:
            # Log error metrics
            safe_log_call(mlflow_logger, "log_metric", "error_occurred", 1.0)
            safe_log_call(mlflow_logger, "log_text", str(e), "error_log.txt")
            logger.error(f"Error generating requirements: {e}")
            raise
        finally:
            safe_log_call(mlflow_logger, "end_run")
    
    def _generate_requirements_list(self, assignment_description: str) -> List[str]:
        """Generate a list of individual requirements from assignment description"""
        
        from src.agents.utils.prompt_types import PromptType
        prompt_types = [f"[{t.value}]" for t in PromptType]
        type_instructions = "\n".join([
            f"- Use {tag} for requirements that are of type '{tag[1:-1]}'" for tag in prompt_types
        ])

        template_name = self.agent_config.get("template")
        env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=select_autoescape(["jinja"])
        )
        template = env.get_template(template_name)
        prompt = template.render(
            assignment_description=assignment_description,
            type_instructions=type_instructions
        )
        
        # Log the prompt being sent to LLM
        mlflow_logger = get_mlflow_logger()
        safe_log_call(mlflow_logger, "log_prompt", prompt, "requirement_generation", "llm_prompt")
        
        # Log trace step: LLM generation
        safe_log_call(mlflow_logger, "log_trace_step", "llm_generation", {
            "prompt_length": len(prompt),
            "model_name": self.agent_config.get("model_name", "unknown")
        }, step_number=2)
        
        logging.info(f"Invoking LLM with prompt of length {len(prompt)}")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        # Minimal change: read actual message content
        raw_content = getattr(response, "content", str(response)).strip()
        
        # Log the raw response as artifact
        mlflow_logger = get_mlflow_logger()
        safe_log_call(mlflow_logger, "log_text", raw_content, "raw_llm_response.txt")
        
        # Debug: Log the raw response
        logger.info(f"Raw LLM response: {repr(raw_content)}")
        
        # Clean the content by removing think tags and explanations
        content = self._clean_llm_response(raw_content)
        
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
    
    def _clean_llm_response(self, raw_content: str) -> str:
        """
        Clean LLM response by removing think tags, explanations, and other non-requirement content
        
        Args:
            raw_content: The raw response from the LLM
            
        Returns:
            Cleaned content containing only the requirements
        """
        content = raw_content
        
        # Remove think tags and their content
        import re
        
        # Remove <think>...</think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Remove <thinking>...</thinking> tags and their content
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        
        # Remove <reasoning>...</reasoning> tags and their content
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        
        # Remove <analysis>...</analysis> tags and their content
        content = re.sub(r'<analysis>.*?</analysis>', '', content, flags=re.DOTALL)
        
        # Find the first occurrence of a dash (-) and cut everything before it
        dash_index = content.find('-')
        if dash_index != -1:
            content = content[dash_index:]
        
        # If no dash found, try to find numbered requirements
        if dash_index == -1:
            # Look for "Requirement 1:" pattern
            req_match = re.search(r'Requirement\s+\d+:', content)
            if req_match:
                content = content[req_match.start():]
            else:
                # Look for numbered list (1., 2., etc.)
                num_match = re.search(r'\d+\.', content)
                if num_match:
                    content = content[num_match.start():]
        
        # Clean up any leading/trailing whitespace
        content = content.strip()
        
        return content
