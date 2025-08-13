#!/usr/bin/env python3
"""
Code Correction Agent

This agent evaluates student code against the generated correction prompts.
It analyzes the code for each rubric item and provides detailed feedback
with scores and suggestions for improvement.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

@dataclass
class CriterionEvaluation:
    """Evaluation result for a single criterion"""
    name: str
    met: bool
    score: float
    feedback: str
    suggestion: str

@dataclass
class ItemEvaluation:
    """Evaluation result for a single rubric item"""
    rubric_item_id: str
    rubric_item_title: str
    total_score: float
    max_score: float
    overall_feedback: str
    criteria_evaluations: List[CriterionEvaluation]

@dataclass
class ComprehensiveEvaluation:
    """Comprehensive evaluation of the entire code"""
    correctness: str
    quality: str
    documentation: str
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
    total_score: float
    max_possible_score: float
    grade_percentage: float
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
            total_score = 0.0
            max_possible_score = 0.0
            
            for prompt in prompt_set.prompts:
                evaluation = self._evaluate_item(student_code, prompt)
                item_evaluations.append(evaluation)
                total_score += evaluation.total_score
                max_possible_score += evaluation.max_score
            
            # Generate comprehensive evaluation
            comprehensive_eval = self._generate_comprehensive_evaluation(
                student_code, prompt_set, item_evaluations
            )
            
            # Calculate grade percentage
            grade_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
            
            # Generate summary
            summary = self._generate_summary(item_evaluations, comprehensive_eval, grade_percentage)
            
            result = CorrectionResult(
                student_code=student_code,
                assignment_description=prompt_set.assignment_description,
                programming_language=prompt_set.programming_language,
                item_evaluations=item_evaluations,
                comprehensive_evaluation=comprehensive_eval,
                total_score=total_score,
                max_possible_score=max_possible_score,
                grade_percentage=grade_percentage,
                summary=summary
            )
            
            logger.info(f"Code evaluation completed. Score: {grade_percentage:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate code: {e}")
            return self._create_fallback_result(student_code, prompt_set)
    
    def _evaluate_item(self, student_code: str, prompt: Any) -> ItemEvaluation:
        """Evaluate a single rubric item"""
        
        evaluation_prompt = f"""Evaluate the following student code against the rubric item.

STUDENT CODE:
```python
{student_code}
```

RUBRIC ITEM: {prompt.rubric_item_title}
EVALUATION PROMPT: {prompt.prompt}

CRITERIA TO EVALUATE:
{chr(10).join(f"- {criterion}" for criterion in prompt.criteria)}

MAX SCORE: {prompt.max_score}

Please evaluate the code and provide your response in the following XML format:

<EVALUATION>
  <CRITERION name="criterion_name">
    <MET>Yes/No</MET>
    <SCORE>Points awarded (0-{prompt.max_score})</SCORE>
    <FEEDBACK>Specific feedback about this criterion</FEEDBACK>
    <SUGGESTION>How to improve this aspect</SUGGESTION>
  </CRITERION>
  
  <SUMMARY>
    <TOTAL_SCORE>Total points for this item</TOTAL_SCORE>
    <OVERALL_FEEDBACK>Overall assessment of this aspect</OVERALL_FEEDBACK>
  </SUMMARY>
</EVALUATION>

Be specific and provide actionable feedback for a beginner student.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=evaluation_prompt)])
            content = str(response).strip()
            
            # Parse XML response
            criteria_evaluations = []
            total_score = 0.0
            
            # Extract criterion evaluations
            criterion_pattern = r'<CRITERION name="([^"]+)">\s*<MET>([^<]+)</MET>\s*<SCORE>([^<]+)</SCORE>\s*<FEEDBACK>([^<]+)</FEEDBACK>\s*<SUGGESTION>([^<]+)</SUGGESTION>\s*</CRITERION>'
            matches = re.findall(criterion_pattern, content, re.DOTALL)
            
            for match in matches:
                criterion_name, met_str, score_str, feedback, suggestion = match
                met = met_str.strip().lower() == "yes"
                score = float(score_str.strip())
                
                criterion_eval = CriterionEvaluation(
                    name=criterion_name.strip(),
                    met=met,
                    score=score,
                    feedback=feedback.strip(),
                    suggestion=suggestion.strip()
                )
                criteria_evaluations.append(criterion_eval)
                total_score += score
            
            # Extract overall feedback
            overall_feedback_match = re.search(r'<OVERALL_FEEDBACK>([^<]+)</OVERALL_FEEDBACK>', content, re.DOTALL)
            overall_feedback = overall_feedback_match.group(1).strip() if overall_feedback_match else "Evaluation completed"
            
            return ItemEvaluation(
                rubric_item_id=prompt.rubric_item_id,
                rubric_item_title=prompt.rubric_item_title,
                total_score=total_score,
                max_score=prompt.max_score,
                overall_feedback=overall_feedback,
                criteria_evaluations=criteria_evaluations
            )
            
        except Exception as e:
            logger.error(f"Failed to evaluate item {prompt.rubric_item_title}: {e}")
            # Return fallback evaluation
            return ItemEvaluation(
                rubric_item_id=prompt.rubric_item_id,
                rubric_item_title=prompt.rubric_item_title,
                total_score=0.0,
                max_score=prompt.max_score,
                overall_feedback="Evaluation failed - manual review required",
                criteria_evaluations=[]
            )
    
    def _generate_comprehensive_evaluation(
        self,
        student_code: str,
        prompt_set: Any,
        item_evaluations: List[ItemEvaluation]
    ) -> ComprehensiveEvaluation:
        """Generate comprehensive evaluation of the entire code"""
        
        comprehensive_prompt = f"""Provide a comprehensive evaluation of the student's code.

STUDENT CODE:
```{prompt_set.programming_language}
{student_code}
```

ASSIGNMENT: {prompt_set.assignment_description}

GENERAL EVALUATION PROMPT: {prompt_set.general_prompt}

ITEM EVALUATIONS:
{chr(10).join(f"- {eval.rubric_item_title}: {eval.total_score}/{eval.max_score}" for eval in item_evaluations)}

Please provide a comprehensive evaluation in the following XML format:

<COMPREHENSIVE_EVALUATION>
  <OVERALL_ASSESSMENT>
    <CORRECTNESS>Assessment of functional correctness</CORRECTNESS>
    <QUALITY>Assessment of code quality</QUALITY>
    <DOCUMENTATION>Assessment of documentation</DOCUMENTATION>
    <ERROR_HANDLING>Assessment of error handling</ERROR_HANDLING>
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

Provide constructive feedback suitable for a beginner student.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=comprehensive_prompt)])
            content = str(response).strip()
            
            # Parse XML response
            correctness = self._extract_xml_tag(content, "CORRECTNESS", "Assessment pending")
            quality = self._extract_xml_tag(content, "QUALITY", "Assessment pending")
            documentation = self._extract_xml_tag(content, "DOCUMENTATION", "Assessment pending")
            error_handling = self._extract_xml_tag(content, "ERROR_HANDLING", "Assessment pending")
            
            strengths = self._extract_list_from_xml(content, "STRENGTHS")
            areas_for_improvement = self._extract_list_from_xml(content, "AREAS_FOR_IMPROVEMENT")
            suggestions = self._extract_list_from_xml(content, "SUGGESTIONS")
            learning_resources = self._extract_list_from_xml(content, "LEARNING_RESOURCES")
            
            return ComprehensiveEvaluation(
                correctness=correctness,
                quality=quality,
                documentation=documentation,
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
                documentation="Evaluation failed",
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
        grade_percentage: float
    ) -> str:
        """Generate a summary of the evaluation"""
        
        if grade_percentage >= 90:
            grade_letter = "A"
        elif grade_percentage >= 80:
            grade_letter = "B"
        elif grade_percentage >= 70:
            grade_letter = "C"
        elif grade_percentage >= 60:
            grade_letter = "D"
        else:
            grade_letter = "F"
        
        summary = f"Grade: {grade_letter} ({grade_percentage:.1f}%)\n\n"
        
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
                documentation="Evaluation failed",
                error_handling="Evaluation failed",
                strengths=["Code submitted for evaluation"],
                areas_for_improvement=["Manual review required"],
                suggestions=["Please review the code manually"],
                learning_resources=["General programming resources"]
            ),
            total_score=0.0,
            max_possible_score=100.0,
            grade_percentage=0.0,
            summary="Evaluation failed - manual review required"
        )
