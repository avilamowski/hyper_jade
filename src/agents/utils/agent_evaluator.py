#!/usr/bin/env python3
"""
Agent Evaluator - LLM as a Judge

This module provides evaluation capabilities for agent outputs using an LLM as a judge.
It evaluates the coherence and quality of agent outputs against their expected tasks.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from src.config import get_agent_config
from src.core.mlflow_utils import mlflow_logger

load_dotenv(override=False)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Evaluates agent outputs using an LLM as a judge"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluator_config = get_agent_config(config, 'agent_evaluator')
        self.llm = self._setup_llm()
        self.evaluation_criteria = self.evaluator_config.get('evaluation_criteria', {})
    
    def _extract_response_content(self, response) -> str:
        """Extract content from LLM response in a safe way"""
        try:
            if hasattr(response, 'content'):
                return str(response.content)
            elif hasattr(response, 'text'):
                return str(response.text)
            else:
                return str(response)
        except Exception:
            return str(response)
    
    def _parse_llm_response(self, response_content: str, agent_name: str) -> Dict[str, float]:
        """Parse LLM response in simple format: 'criterion: score' one per line"""
        import re
        
        # Log the raw response for debugging
        logger.info(f"Raw evaluation response for {agent_name}: {repr(response_content)}")
        
        # Get expected criteria for this agent
        expected_criteria = self.evaluation_criteria.get(agent_name, {})
        
        # Parse response line by line
        scores = {}
        lines = response_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for pattern "criterion: score" or "criterion = score"
            match = re.match(r'^([^:=\s]+)\s*[:=]\s*(\d+(?:\.\d+)?)', line, re.IGNORECASE)
            if match:
                criterion = match.group(1).strip().lower()
                score = float(match.group(2))
                
                # Validate score range
                if 1 <= score <= 5:
                    scores[criterion] = score
                else:
                    logger.warning(f"Score out of range (1-5) for {criterion}: {score}")
        
        # Validate that we got all expected criteria
        missing_criteria = set(expected_criteria.keys()) - set(scores.keys())
        if missing_criteria:
            logger.warning(f"Missing criteria for {agent_name}: {missing_criteria}")
        
        # Add default scores for missing criteria
        for criterion in missing_criteria:
            scores[criterion] = 1.0  # Default to lowest score
            logger.info(f"Using default score 1.0 for missing criterion: {criterion}")
        
        return scores
    
    def _calculate_weighted_score(self, scores: Dict[str, float], agent_name: str) -> float:
        """Calculate weighted average score based on criteria weights"""
        expected_criteria = self.evaluation_criteria.get(agent_name, {})
        
        if not scores or not expected_criteria:
            return 0.0
        
        # Calculate weighted score using criteria weights
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criterion, weight in expected_criteria.items():
            score = scores.get(criterion, 0.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        weighted_score = total_weighted_score / total_weight
        
        # Convert to 1-10 scale (multiply by 2)
        return weighted_score * 2
    
    def _setup_llm(self):
        """Setup LLM for evaluation"""
        provider = self.evaluator_config.get("provider", "openai")
        model_name = self.evaluator_config.get("model_name", "gpt-4o-mini")
        temperature = self.evaluator_config.get("temperature", 0.1)

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found")
            base_url = os.getenv("OPENAI_BASE_URL")
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
    
    def _build_evaluation_prompt(self, agent_name: str, context: str) -> Tuple[str, str]:
        """Build evaluation prompt based on agent and criteria from config"""
        criteria = self.evaluation_criteria.get(agent_name, {})
        
        if not criteria:
            raise ValueError(f"No evaluation criteria found for agent: {agent_name}")
        
        # Build criteria description
        criteria_descriptions = {
            'completeness': 'Are all aspects of the assignment covered?',
            'clarity': 'Are the requirements clear and unambiguous?',
            'specificity': 'Are the requirements specific enough to be testable?',
            'coherence': 'Do the requirements logically follow from the assignment?',
            'independence': 'Are the requirements independent of each other?',
            'template_completeness': 'Does the template include all necessary variables and placeholders?',
            'code_analysis_focus': 'Does the prompt guide toward proper code analysis?',
            'jinja2_syntax': 'Is the Jinja2 syntax correct and appropriate?',
            'requirement_coverage': 'Does the analysis address the specific requirement?',
            'analysis_depth': 'Is the analysis thorough and insightful?',
            'accuracy': 'Is the analysis technically accurate?',
            'actionability': 'Does the feedback provide actionable guidance?'
        }
        
        criteria_text = "\n".join([
            f"{i+1}. {criterion.replace('_', ' ').title()}: {criteria_descriptions.get(criterion, 'Evaluate this aspect')}"
            for i, criterion in enumerate(criteria.keys())
        ])
        
        system_prompt = f"""You are an expert evaluator for educational content generation.
Your task is to evaluate the quality and coherence of outputs from the {agent_name.replace('_', ' ')} agent.

Evaluation Criteria:
{criteria_text}

Rate each criterion from 1-5 (1=poor, 5=excellent).
Provide your evaluation in a simple format: one criterion per line with "criterion: score".

Example format:
completeness: 4
clarity: 3
specificity: 5"""
        
        human_prompt = f"""Please evaluate the {agent_name.replace('_', ' ')} output based on the criteria above.

{context}

Provide your evaluation in the following format (one criterion per line):
{chr(10).join(f"{criterion}: <score>" for criterion in criteria.keys())}"""
        
        return system_prompt, human_prompt
    
    def evaluate_requirement_generator(
        self, 
        assignment_text: str, 
        generated_requirements: List[str],
        output_directory: str
    ) -> Dict[str, Any]:
        """
        Evaluate the output of the requirement generator agent
        
        Args:
            assignment_text: Original assignment description
            generated_requirements: List of generated requirement texts
            output_directory: Directory where requirements were saved
            
        Returns:
            Dictionary with evaluation metrics
        """
        agent_name = "requirement_generator"
        
        # Start MLflow run for evaluation
        run_id = mlflow_logger.start_run(
            run_name=f"evaluate_{agent_name}",
            tags={"agent": agent_name, "evaluation_type": "llm_judge"}
        )
        
        try:
            # Build evaluation prompt
            context = f"""
ASSIGNMENT DESCRIPTION:
{assignment_text}

GENERATED REQUIREMENTS:
{chr(10).join(f"{i+1}. {req}" for i, req in enumerate(generated_requirements))}

NUMBER OF REQUIREMENTS: {len(generated_requirements)}
OUTPUT DIRECTORY: {output_directory}
"""
            
            system_prompt, human_prompt = self._build_evaluation_prompt(agent_name, context)
            
            # Log prompts to MLflow
            mlflow_logger.log_prompt(system_prompt, f"{agent_name}_system", "evaluation")
            mlflow_logger.log_prompt(human_prompt, f"{agent_name}_human", "evaluation")
            
            # Log evaluation parameters
            mlflow_logger.log_params({
                "agent_name": agent_name,
                "num_requirements": len(generated_requirements),
                "output_directory": output_directory,
                "evaluation_criteria": list(self.evaluation_criteria.get(agent_name, {}).keys())
            })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            start_time = time.time()
            response = self.llm.invoke(messages)
            evaluation_time = time.time() - start_time
            
            # Handle different response formats from different LLM providers
            response_content = self._extract_response_content(response)
            
            # Parse scores
            scores = self._parse_llm_response(response_content, agent_name)
            
            # Calculate weighted scores
            average_score = sum(scores.values()) / len(scores) if scores else 0.0
            overall_score = self._calculate_weighted_score(scores, agent_name)
            
            # Build evaluation result
            evaluation_result = {
                "scores": scores,
                "average_score": average_score,
                "overall_score": overall_score,
                "evaluation_time": evaluation_time,
                "criteria_used": list(self.evaluation_criteria.get(agent_name, {}).keys())
            }
            
            # Log metrics to MLflow
            mlflow_logger.log_metrics({
                f"{agent_name}_evaluation_time": evaluation_time,
                f"{agent_name}_average_score": average_score,
                f"{agent_name}_overall_score": overall_score
            })
            
            # Log individual criterion scores
            for criterion, score in scores.items():
                mlflow_logger.log_metric(f"{agent_name}_{criterion}_score", score)
            
            # Log evaluation result as artifact
            mlflow_logger.log_agent_evaluation_metrics(agent_name, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating requirement generator: {e}")
            mlflow_logger.log_metric(f"{agent_name}_evaluation_error", 1.0)
            return {
                "error": str(e),
                "scores": {},
                "average_score": 0.0,
                "overall_score": 0.0,
                "evaluation_time": 0.0
            }
        finally:
            mlflow_logger.end_run()
    
    def evaluate_prompt_generator(
        self,
        requirement_text: str,
        assignment_text: str,
        generated_prompt: str,
        output_file_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate the output of the prompt generator agent
        
        Args:
            requirement_text: Original requirement text
            assignment_text: Assignment description
            generated_prompt: Generated Jinja2 prompt template
            output_file_path: Path where prompt was saved
            
        Returns:
            Dictionary with evaluation metrics
        """
        agent_name = "prompt_generator"
        
        # Start MLflow run for evaluation
        run_id = mlflow_logger.start_run(
            run_name=f"evaluate_{agent_name}",
            tags={"agent": agent_name, "evaluation_type": "llm_judge"}
        )
        
        try:
            # Build evaluation prompt
            context = f"""
REQUIREMENT TEXT:
{requirement_text}

ASSIGNMENT DESCRIPTION:
{assignment_text}

GENERATED PROMPT TEMPLATE:
{generated_prompt}

OUTPUT FILE: {output_file_path}
"""
            
            system_prompt, human_prompt = self._build_evaluation_prompt(agent_name, context)
            
            # Log prompts to MLflow
            mlflow_logger.log_prompt(system_prompt, f"{agent_name}_system", "evaluation")
            mlflow_logger.log_prompt(human_prompt, f"{agent_name}_human", "evaluation")
            
            # Log evaluation parameters
            mlflow_logger.log_params({
                "agent_name": agent_name,
                "output_file": output_file_path,
                "prompt_length": len(generated_prompt),
                "evaluation_criteria": list(self.evaluation_criteria.get(agent_name, {}).keys())
            })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            start_time = time.time()
            response = self.llm.invoke(messages)
            evaluation_time = time.time() - start_time
            
            # Handle different response formats from different LLM providers
            response_content = self._extract_response_content(response)
            
            # Parse scores
            scores = self._parse_llm_response(response_content, agent_name)
            
            # Calculate weighted scores
            average_score = sum(scores.values()) / len(scores) if scores else 0.0
            overall_score = self._calculate_weighted_score(scores, agent_name)
            
            # Build evaluation result
            evaluation_result = {
                "scores": scores,
                "average_score": average_score,
                "overall_score": overall_score,
                "evaluation_time": evaluation_time,
                "criteria_used": list(self.evaluation_criteria.get(agent_name, {}).keys())
            }
            
            # Log metrics to MLflow
            mlflow_logger.log_metrics({
                f"{agent_name}_evaluation_time": evaluation_time,
                f"{agent_name}_average_score": average_score,
                f"{agent_name}_overall_score": overall_score
            })
            
            # Log individual criterion scores
            for criterion, score in scores.items():
                mlflow_logger.log_metric(f"{agent_name}_{criterion}_score", score)
            
            # Log evaluation result as artifact
            mlflow_logger.log_agent_evaluation_metrics(agent_name, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating prompt generator: {e}")
            mlflow_logger.log_metric(f"{agent_name}_evaluation_error", 1.0)
            return {
                "error": str(e),
                "scores": {},
                "average_score": 0.0,
                "overall_score": 0.0,
                "evaluation_time": 0.0
            }
        finally:
            mlflow_logger.end_run()
    
    def evaluate_code_corrector(
        self,
        requirement_text: str,
        prompt_template: str,
        student_code: str,
        analysis_result: str,
        output_file_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate the output of the code corrector agent
        
        Args:
            requirement_text: Original requirement text
            prompt_template: Jinja2 prompt template used
            student_code: Student's code that was analyzed
            analysis_result: Generated analysis result
            output_file_path: Path where analysis was saved
            
        Returns:
            Dictionary with evaluation metrics
        """
        agent_name = "code_corrector"
        
        # Start MLflow run for evaluation
        run_id = mlflow_logger.start_run(
            run_name=f"evaluate_{agent_name}",
            tags={"agent": agent_name, "evaluation_type": "llm_judge"}
        )
        
        try:
            # Build evaluation prompt
            context = f"""
REQUIREMENT TEXT:
{requirement_text}

PROMPT TEMPLATE USED:
{prompt_template}

STUDENT CODE:
{student_code}

ANALYSIS RESULT:
{analysis_result}

OUTPUT FILE: {output_file_path}
"""
            
            system_prompt, human_prompt = self._build_evaluation_prompt(agent_name, context)
            
            # Log prompts to MLflow
            mlflow_logger.log_prompt(system_prompt, f"{agent_name}_system", "evaluation")
            mlflow_logger.log_prompt(human_prompt, f"{agent_name}_human", "evaluation")
            
            # Log evaluation parameters
            mlflow_logger.log_params({
                "agent_name": agent_name,
                "output_file": output_file_path,
                "student_code_length": len(student_code),
                "analysis_length": len(analysis_result),
                "evaluation_criteria": list(self.evaluation_criteria.get(agent_name, {}).keys())
            })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            start_time = time.time()
            response = self.llm.invoke(messages)
            evaluation_time = time.time() - start_time
            
            # Handle different response formats from different LLM providers
            response_content = self._extract_response_content(response)
            
            # Parse scores
            scores = self._parse_llm_response(response_content, agent_name)
            
            # Calculate weighted scores
            average_score = sum(scores.values()) / len(scores) if scores else 0.0
            overall_score = self._calculate_weighted_score(scores, agent_name)
            
            # Build evaluation result
            evaluation_result = {
                "scores": scores,
                "average_score": average_score,
                "overall_score": overall_score,
                "evaluation_time": evaluation_time,
                "criteria_used": list(self.evaluation_criteria.get(agent_name, {}).keys())
            }
            
            # Log metrics to MLflow
            mlflow_logger.log_metrics({
                f"{agent_name}_evaluation_time": evaluation_time,
                f"{agent_name}_average_score": average_score,
                f"{agent_name}_overall_score": overall_score
            })
            
            # Log individual criterion scores
            for criterion, score in scores.items():
                mlflow_logger.log_metric(f"{agent_name}_{criterion}_score", score)
            
            # Log evaluation result as artifact
            mlflow_logger.log_agent_evaluation_metrics(agent_name, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating code corrector: {e}")
            mlflow_logger.log_metric(f"{agent_name}_evaluation_error", 1.0)
            return {
                "error": str(e),
                "scores": {},
                "average_score": 0.0,
                "overall_score": 0.0,
                "evaluation_time": 0.0
            }
        finally:
            mlflow_logger.end_run()
