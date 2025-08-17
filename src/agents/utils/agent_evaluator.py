#!/usr/bin/env python3
"""
Agent Evaluator - LLM as a Judge

This module provides evaluation capabilities for agent outputs using an LLM as a judge.
It evaluates the coherence and quality of agent outputs against their expected tasks.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from src.config import get_agent_config

load_dotenv(override=False)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Evaluates agent outputs using an LLM as a judge"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluator_config = get_agent_config(config, 'agent_evaluator')
        self.llm = self._setup_llm()
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON content"""
        import re
        
        # Log the raw response for debugging
        logger.info(f"Raw evaluation response: {repr(response_content)}")
        
        # Look for JSON content in the response
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(0)
            logger.info(f"Extracted JSON content: {json_content}")
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON: {e}")
                logger.error(f"JSON content: {json_content}")
                raise
        
        # If no JSON found, try to parse the entire response
        logger.warning("No JSON found in response, trying to parse entire response")
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            logger.error(f"Response content: {response_content}")
            raise
        
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
        system_prompt = """You are an expert evaluator for educational content generation. 
        Your task is to evaluate the quality and coherence of requirements generated from an assignment description.
        
        Evaluation Criteria:
        1. Completeness: Are all aspects of the assignment covered?
        2. Clarity: Are the requirements clear and unambiguous?
        3. Specificity: Are the requirements specific enough to be testable?
        4. Coherence: Do the requirements logically follow from the assignment?
        5. Independence: Are the requirements independent of each other?
        
        Rate each criterion from 1-5 (1=poor, 5=excellent) and provide a brief justification.
        Also provide an overall quality score from 1-10."""
        
        # Prepare evaluation context
        context = f"""
        ASSIGNMENT DESCRIPTION:
        {assignment_text}
        
        GENERATED REQUIREMENTS:
        {chr(10).join(f"{i+1}. {req}" for i, req in enumerate(generated_requirements))}
        
        NUMBER OF REQUIREMENTS: {len(generated_requirements)}
        OUTPUT DIRECTORY: {output_directory}
        """
        
        human_prompt = f"""
        Please evaluate the generated requirements based on the criteria above.
        
        {context}
        
        Provide your evaluation in the following JSON format:
        {{
            "completeness": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "clarity": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "specificity": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "coherence": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "independence": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "overall_score": <1-10>,
            "summary": "<overall assessment>",
            "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>"]
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            # Handle different response formats from different LLM providers
            if hasattr(response, 'content'):
                response_content = response.content
            elif hasattr(response, 'text'):
                response_content = response.text
            elif isinstance(response, str):
                response_content = response
            else:
                response_content = str(response)
            
            evaluation = self._parse_llm_response(response_content)
            
            # Calculate average score
            criteria_scores = [
                evaluation["completeness"]["score"],
                evaluation["clarity"]["score"],
                evaluation["specificity"]["score"],
                evaluation["coherence"]["score"],
                evaluation["independence"]["score"]
            ]
            evaluation["average_criteria_score"] = sum(criteria_scores) / len(criteria_scores)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating requirement generator: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "average_criteria_score": 0
            }
    
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
        system_prompt = """You are an expert evaluator for prompt engineering and template generation.
        Your task is to evaluate the quality and effectiveness of a Jinja2 prompt template generated for code analysis.
        
        Evaluation Criteria:
        1. Template Completeness: Does the template include all necessary variables and placeholders?
        2. Clarity: Is the prompt clear and easy to understand?
        3. Specificity: Does the prompt specifically address the requirement?
        4. Code Analysis Focus: Does the prompt guide toward proper code analysis?
        5. Jinja2 Syntax: Is the Jinja2 syntax correct and appropriate?
        
        Rate each criterion from 1-5 (1=poor, 5=excellent) and provide a brief justification.
        Also provide an overall quality score from 1-10."""
        
        context = f"""
        REQUIREMENT TEXT:
        {requirement_text}
        
        ASSIGNMENT DESCRIPTION:
        {assignment_text}
        
        GENERATED PROMPT TEMPLATE:
        {generated_prompt}
        
        OUTPUT FILE: {output_file_path}
        """
        
        human_prompt = f"""
        Please evaluate the generated prompt template based on the criteria above.
        
        {context}
        
        Provide your evaluation in the following JSON format:
        {{
            "template_completeness": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "clarity": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "specificity": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "code_analysis_focus": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "jinja2_syntax": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "overall_score": <1-10>,
            "summary": "<overall assessment>",
            "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>"]
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            # Handle different response formats from different LLM providers
            if hasattr(response, 'content'):
                response_content = response.content
            elif hasattr(response, 'text'):
                response_content = response.text
            elif isinstance(response, str):
                response_content = response
            else:
                response_content = str(response)
            
            evaluation = self._parse_llm_response(response_content)
            
            # Calculate average score
            criteria_scores = [
                evaluation["template_completeness"]["score"],
                evaluation["clarity"]["score"],
                evaluation["specificity"]["score"],
                evaluation["code_analysis_focus"]["score"],
                evaluation["jinja2_syntax"]["score"]
            ]
            evaluation["average_criteria_score"] = sum(criteria_scores) / len(criteria_scores)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating prompt generator: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "average_criteria_score": 0
            }
    
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
        system_prompt = """You are an expert evaluator for code analysis and feedback generation.
        Your task is to evaluate the quality and accuracy of code analysis results.
        
        Evaluation Criteria:
        1. Requirement Coverage: Does the analysis address the specific requirement?
        2. Analysis Depth: Is the analysis thorough and insightful?
        3. Clarity: Is the feedback clear and understandable?
        4. Accuracy: Is the analysis technically accurate?
        5. Actionability: Does the feedback provide actionable guidance?
        
        Rate each criterion from 1-5 (1=poor, 5=excellent) and provide a brief justification.
        Also provide an overall quality score from 1-10."""
        
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
        
        human_prompt = f"""
        Please evaluate the code analysis result based on the criteria above.
        
        {context}
        
        Provide your evaluation in the following JSON format:
        {{
            "requirement_coverage": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "analysis_depth": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "clarity": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "accuracy": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "actionability": {{
                "score": <1-5>,
                "justification": "<explanation>"
            }},
            "overall_score": <1-10>,
            "summary": "<overall assessment>",
            "suggestions": ["<improvement suggestion 1>", "<improvement suggestion 2>"]
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            # Handle different response formats from different LLM providers
            if hasattr(response, 'content'):
                response_content = response.content
            elif hasattr(response, 'text'):
                response_content = response.text
            elif isinstance(response, str):
                response_content = response
            else:
                response_content = str(response)
            
            evaluation = self._parse_llm_response(response_content)
            
            # Calculate average score
            criteria_scores = [
                evaluation["requirement_coverage"]["score"],
                evaluation["analysis_depth"]["score"],
                evaluation["clarity"]["score"],
                evaluation["accuracy"]["score"],
                evaluation["actionability"]["score"]
            ]
            evaluation["average_criteria_score"] = sum(criteria_scores) / len(criteria_scores)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating code corrector: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "average_criteria_score": 0
            }
