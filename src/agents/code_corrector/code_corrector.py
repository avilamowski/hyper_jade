#!/usr/bin/env python3
"""
Code Correction Agent

This agent evaluates student code against the generated correction prompts.
It analyzes the code for each rubric item and identifies errors and issues
with detailed feedback for improvement.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

@dataclass
class ErrorIdentification:
    """Identified error in the code"""
    error_type: str
    location: str
    description: str
    suggestion: str
    line_number: Optional[int] = None

@dataclass
class ItemEvaluation:
    """Evaluation result for a single rubric item"""
    rubric_item_id: str
    rubric_item_title: str
    errors_found: List[ErrorIdentification]
    overall_feedback: str
    is_passing: bool

@dataclass
class ComprehensiveEvaluation:
    """Comprehensive evaluation of the entire code"""
    correctness: str
    quality: str
    error_handling: str
    strengths: List[str]
    areas_for_improvement: List[str]
    suggestions: List[str]
    learning_resources: List[str]

@dataclass
class CorrectionResult:
    """Complete result of code correction"""
    student_code: str
    assignment_description: str
    programming_language: str
    item_evaluations: List[ItemEvaluation]
    comprehensive_evaluation: ComprehensiveEvaluation
    total_errors: int
    summary: str

class CodeCorrectorAgent:
    """Agent that evaluates student code using correction prompts"""
    
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
    
    def correct_code(
        self,
        student_code: str,
        prompt_set: Any  # PromptSet from prompt generator
    ) -> CorrectionResult:
        """Evaluate student code using the generated prompts"""
        logger.info("Evaluating student code with correction prompts")
        
        try:
            # Evaluate each rubric item
            item_evaluations = []
            all_errors = []
            
            for prompt in prompt_set.prompts:
                evaluation = self._evaluate_item(student_code, prompt)
                item_evaluations.append(evaluation)
                all_errors.extend(evaluation.errors_found)
            
            # Generate comprehensive evaluation
            comprehensive_eval = self._generate_comprehensive_evaluation(
                student_code, prompt_set, item_evaluations
            )
            
            # Count errors
            total_errors = len(all_errors)
            
            # Generate summary
            summary = self._generate_summary(item_evaluations, comprehensive_eval, total_errors)
            
            result = CorrectionResult(
                student_code=student_code,
                assignment_description=prompt_set.assignment_description,
                programming_language=prompt_set.programming_language,
                item_evaluations=item_evaluations,
                comprehensive_evaluation=comprehensive_eval,
                total_errors=total_errors,
                summary=summary
            )
            
            logger.info(f"Code evaluation completed. Found {total_errors} errors")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate code: {e}")
            return self._create_fallback_result(student_code, prompt_set)
    
    def _evaluate_item(self, student_code: str, prompt: Any) -> ItemEvaluation:
        """Evaluate a single rubric item"""
        
        evaluation_prompt = f"""Analyze the following student code and identify errors related to the rubric item.

STUDENT CODE:
```{prompt_set.programming_language}
{student_code}
```

RUBRIC ITEM: {prompt.rubric_item_title}
EVALUATION PROMPT: {prompt.prompt}

CRITERIA TO EVALUATE:
{chr(10).join(f"- {criterion}" for criterion in prompt.criteria)}

Please analyze the code and identify errors in the following XML format:

<EVALUATION>
  <ERRORS>
    <ERROR type="error_type" line="line_number">
      <LOCATION>Where the error occurs</LOCATION>
      <DESCRIPTION>Detailed description of the error</DESCRIPTION>
      <SUGGESTION>How to fix this error</SUGGESTION>
    </ERROR>
  </ERRORS>
  
  <ASSESSMENT>
    <IS_PASSING>Yes/No</IS_PASSING>
    <OVERALL_FEEDBACK>Overall assessment of this aspect</OVERALL_FEEDBACK>
  </ASSESSMENT>
</EVALUATION>

Focus on identifying specific errors and providing actionable feedback for improvement.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
            content = str(response).strip()
            
            # Parse XML response
            errors_found = []
            
            # Extract error information
            error_pattern = r'<ERROR type="([^"]+)" line="([^"]*)">\s*<LOCATION>([^<]+)</LOCATION>\s*<DESCRIPTION>([^<]+)</DESCRIPTION>\s*<SUGGESTION>([^<]+)</SUGGESTION>\s*</ERROR>'
            matches = re.findall(error_pattern, content, re.DOTALL)
            
            for match in matches:
                error_type, line_str, location, description, suggestion = match
                line_number = int(line_str) if line_str and line_str.isdigit() else None
                
                error = ErrorIdentification(
                    error_type=error_type.strip(),
                    location=location.strip(),
                    description=description.strip(),
                    suggestion=suggestion.strip(),
                    line_number=line_number
                )
                errors_found.append(error)
            
            # Extract assessment
            is_passing_match = re.search(r'<IS_PASSING>([^<]+)</IS_PASSING>', content, re.DOTALL)
            is_passing = is_passing_match.group(1).strip().lower() == "yes" if is_passing_match else False
            
            overall_feedback_match = re.search(r'<OVERALL_FEEDBACK>([^<]+)</OVERALL_FEEDBACK>', content, re.DOTALL)
            overall_feedback = overall_feedback_match.group(1).strip() if overall_feedback_match else "Evaluation completed"
            
            return ItemEvaluation(
                rubric_item_id=prompt.rubric_item_id,
                rubric_item_title=prompt.rubric_item_title,
                errors_found=errors_found,
                overall_feedback=overall_feedback,
                is_passing=is_passing
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate item {prompt.rubric_item_title}: {e}")
            # Return fallback evaluation
            return ItemEvaluation(
                rubric_item_id=prompt.rubric_item_id,
                rubric_item_title=prompt.rubric_item_title,
                errors_found=[],
                overall_feedback="Evaluation failed - manual review required",
                is_passing=False
            )
    
    def _generate_comprehensive_evaluation(
        self,
        student_code: str,
        prompt_set: Any,
        item_evaluations: List[ItemEvaluation]
    ) -> ComprehensiveEvaluation:
        """Generate comprehensive evaluation of the entire code"""
        
        comprehensive_prompt = f"""Provide a comprehensive evaluation of the student's code focusing on error identification.

STUDENT CODE:
```{prompt_set.programming_language}
{student_code}
```

ASSIGNMENT: {prompt_set.assignment_description}

GENERAL EVALUATION PROMPT: {prompt_set.general_prompt}

ITEM EVALUATIONS:
{chr(10).join(f"- {eval.rubric_item_title}: {'PASS' if eval.is_passing else 'FAIL'} ({len(eval.errors_found)} errors)" for eval in item_evaluations)}

Please provide a comprehensive evaluation in the following XML format:

<COMPREHENSIVE_EVALUATION>
  <OVERALL_ASSESSMENT>
    <CORRECTNESS>Assessment of functional correctness and errors</CORRECTNESS>
    <QUALITY>Assessment of code quality issues</QUALITY>
    <ERROR_HANDLING>Assessment of error handling and edge cases</ERROR_HANDLING>
  </OVERALL_ASSESSMENT>
  
  <DETAILED_FEEDBACK>
    <STRENGTHS>List of what the student did well</STRENGTHS>
    <AREAS_FOR_IMPROVEMENT>Specific areas that need work</AREAS_FOR_IMPROVEMENT>
    <SUGGESTIONS>Concrete suggestions for improvement</SUGGESTIONS>
  </DETAILED_FEEDBACK>
  
  <LEARNING_RESOURCES>
    <RESOURCE>Links or references for learning</RESOURCE>
  </LEARNING_RESOURCES>
</COMPREHENSIVE_EVALUATION>

Focus on identifying errors and providing constructive feedback for improvement.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=comprehensive_prompt)])
            content = str(response).strip()
            
            # Parse XML response
            correctness = self._extract_xml_tag(content, "CORRECTNESS", "Assessment pending")
            quality = self._extract_xml_tag(content, "QUALITY", "Assessment pending")
            error_handling = self._extract_xml_tag(content, "ERROR_HANDLING", "Assessment pending")
            
            strengths = self._extract_list_from_xml(content, "STRENGTHS")
            areas_for_improvement = self._extract_list_from_xml(content, "AREAS_FOR_IMPROVEMENT")
            suggestions = self._extract_list_from_xml(content, "SUGGESTIONS")
            learning_resources = self._extract_list_from_xml(content, "LEARNING_RESOURCES")
            
            return ComprehensiveEvaluation(
                correctness=correctness,
                quality=quality,
                error_handling=error_handling,
                strengths=strengths,
                areas_for_improvement=areas_for_improvement,
                suggestions=suggestions,
                learning_resources=learning_resources
            )
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive evaluation: {e}")
            return ComprehensiveEvaluation(
                correctness="Evaluation failed",
                quality="Evaluation failed",
                error_handling="Evaluation failed",
                strengths=["Code submitted for evaluation"],
                areas_for_improvement=["Manual review required"],
                suggestions=["Please review the code manually"],
                learning_resources=["General programming resources"]
            )
    
    def _extract_xml_tag(self, content: str, tag: str, default: str = "") -> str:
        """Extract content from XML tag"""
        pattern = rf'<{tag}>([^<]+)</{tag}>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else default
    
    def _extract_list_from_xml(self, content: str, tag: str) -> List[str]:
        """Extract list items from XML tag"""
        pattern = rf'<{tag}>([^<]+)</{tag}>'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return []
        
        text = match.group(1).strip()
        # Split by common list separators
        items = re.split(r'[•\-\*]|\d+\.', text)
        return [item.strip() for item in items if item.strip()]
    
    def _generate_summary(
        self,
        item_evaluations: List[ItemEvaluation],
        comprehensive_eval: ComprehensiveEvaluation,
        total_errors: int
    ) -> str:
        """Generate a summary of the evaluation"""
        
        summary = f"Error Analysis Summary:\n"
        summary += f"Total Errors Found: {total_errors}\n\n"
        
        # Add strengths
        if comprehensive_eval.strengths:
            summary += "Strengths:\n"
            for strength in comprehensive_eval.strengths:
                summary += f"• {strength}\n"
            summary += "\n"
        
        # Add areas for improvement
        if comprehensive_eval.areas_for_improvement:
            summary += "Areas for Improvement:\n"
            for area in comprehensive_eval.areas_for_improvement:
                summary += f"• {area}\n"
            summary += "\n"
        
        # Add suggestions
        if comprehensive_eval.suggestions:
            summary += "Suggestions:\n"
            for suggestion in comprehensive_eval.suggestions:
                summary += f"• {suggestion}\n"
        
        return summary
    
    def _create_fallback_result(self, student_code: str, prompt_set: Any) -> CorrectionResult:
        """Create a fallback result when evaluation fails"""
        return CorrectionResult(
            student_code=student_code,
            assignment_description=prompt_set.assignment_description,
            programming_language=prompt_set.programming_language,
            item_evaluations=[],
            comprehensive_evaluation=ComprehensiveEvaluation(
                correctness="Evaluation failed",
                quality="Evaluation failed",
                error_handling="Evaluation failed",
                strengths=["Code submitted for evaluation"],
                areas_for_improvement=["Manual review required"],
                suggestions=["Please review the code manually"],
                learning_resources=["General programming resources"]
            ),
            total_errors=0,
            summary="Evaluation failed - manual review required"
        )
