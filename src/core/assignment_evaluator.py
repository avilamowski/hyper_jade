#!/usr/bin/env python3
"""
Assignment Evaluator - Main Orchestrator

This is the main orchestrator that coordinates the three specialized agents:
1. Requirement Generator Agent: generates rubrics from assignments
2. Prompt Generator Agent: generates correction prompts for each rubric item
3. Code Correction Agent: evaluates student code using the prompts

The workflow follows the flowchart:
Assignment → Rubric → Prompts → Code Evaluation → Final Report
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import mlflow
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated

from ..agents.requirement_generator import RequirementGeneratorAgent
from ..agents.prompt_generator import PromptGeneratorAgent
from ..agents.code_corrector import CodeCorrectorAgent

logger = logging.getLogger(__name__)

@dataclass
class EvaluationPipeline:
    """Complete evaluation pipeline result"""
    assignment_description: str
    programming_language: str
    rubric: Any  # Rubric from requirement generator
    prompt_set: Any  # PromptSet from prompt generator
    correction_result: Any  # CorrectionResult from code corrector
    metadata: Dict[str, Any]

class AssignmentEvaluatorState(TypedDict):
    """State for the complete assignment evaluation pipeline"""
    assignment_description: str
    programming_language: str
    student_code: str
    
    # Agent outputs
    generated_rubric: Optional[Any]
    generated_prompts: Optional[Any]
    correction_result: Optional[Any]
    
    # Evaluation scores
    rubric_evaluation_score: float
    prompt_evaluation_score: float
    
    # Final result
    final_result: Optional[EvaluationPipeline]

class AssignmentEvaluator:
    """Main orchestrator for the assignment evaluation pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize the three specialized agents
        self.requirement_generator = RequirementGeneratorAgent(config)
        self.prompt_generator = PromptGeneratorAgent(config)
        self.code_corrector = CodeCorrectorAgent(config)
        
        # Create the main workflow graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the main LangGraph workflow"""
        graph = StateGraph(AssignmentEvaluatorState)
        
        # Add nodes for each step in the pipeline
        graph.add_node("generate_requirements", self._generate_requirements)
        graph.add_node("evaluate_rubric", self._evaluate_rubric)
        graph.add_node("generate_prompts", self._generate_prompts)
        graph.add_node("evaluate_prompts", self._evaluate_prompts)
        graph.add_node("correct_code", self._correct_code)
        graph.add_node("aggregate_final_results", self._aggregate_final_results)
        
        # Set entry point
        graph.set_entry_point("generate_requirements")
        
        # Add edges following the flowchart
        graph.add_edge("generate_requirements", "evaluate_rubric")
        graph.add_edge("evaluate_rubric", "generate_prompts")
        graph.add_edge("generate_prompts", "evaluate_prompts")
        graph.add_edge("evaluate_prompts", "correct_code")
        graph.add_edge("correct_code", "aggregate_final_results")
        graph.add_edge("aggregate_final_results", END)
        
        return graph.compile()
    
    def _generate_requirements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Generate rubric from assignment (Requirement Generator Agent)"""
        logger.info("Step 1: Generating rubric from assignment")
        
        try:
            start_time = time.time()
            rubric = self.requirement_generator.generate_rubric(
                assignment_description=state["assignment_description"],
                programming_language=state["programming_language"]
            )
            end_time = time.time()
            
            mlflow.log_metric("step1_rubric_generation_time", end_time - start_time)
            mlflow.log_metric("step1_rubric_items_count", len(rubric.items))
            mlflow.log_param("step1_rubric_title", rubric.title)
            
            # Save rubric as artifact
            rubric_data = {
                "title": rubric.title,
                "description": rubric.description,
                "programming_language": rubric.programming_language,
                "difficulty_level": rubric.difficulty_level,
                "items": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "criteria": item.criteria
                    }
                    for item in rubric.items
                ]
            }
            mlflow.log_dict(rubric_data, "step1_generated_rubric.json")
            
            logger.info(f"Generated rubric with {len(rubric.items)} items")
            return {"generated_rubric": rubric}
        except Exception as e:
            logger.error(f"Failed to generate rubric: {e}")
            mlflow.log_param("step1_error", str(e))
            # Create a basic fallback rubric
            fallback_rubric = self._create_fallback_rubric()
            return {"generated_rubric": fallback_rubric}
    
    def _evaluate_rubric(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Evaluate the generated rubric (LLM as judge)"""
        logger.info("Step 2: Evaluating generated rubric")
        
        if not state.get("generated_rubric"):
            return {"rubric_evaluation_score": 0.0}
        
        # This step simulates the "LLM as judge" from the flowchart
        # In a real implementation, this would compare against teacher rubrics
        try:
            start_time = time.time()
            # Simple evaluation based on rubric completeness
            total_criteria = sum(len(item.criteria) for item in state["generated_rubric"].items)
            score = min(10.0, total_criteria / 2.0)  # Simple scoring
            end_time = time.time()
            
            mlflow.log_metric("step2_rubric_evaluation_time", end_time - start_time)
            mlflow.log_metric("step2_rubric_evaluation_score", score)
            mlflow.log_metric("step2_total_criteria", total_criteria)
            
            # Save evaluation as artifact
            evaluation_data = {
                "rubric_evaluation_score": score,
                "total_criteria": total_criteria,
                "evaluation_method": "criteria_count_based",
                "timestamp": time.time()
            }
            mlflow.log_dict(evaluation_data, "step2_rubric_evaluation.json")
            
            logger.info(f"Rubric evaluation score: {score}")
            return {"rubric_evaluation_score": score}
        except Exception as e:
            logger.error(f"Failed to evaluate rubric: {e}")
            mlflow.log_param("step2_error", str(e))
            return {"rubric_evaluation_score": 7.0}  # Default score
    
    def _generate_prompts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate correction prompts for each rubric item (Prompt Generator Agent)"""
        logger.info("Step 3: Generating correction prompts")
        
        if not state.get("generated_rubric"):
            logger.error("No rubric available for prompt generation")
            return {"generated_prompts": self._create_fallback_prompt_set()}
        
        try:
            start_time = time.time()
            prompt_set = self.prompt_generator.generate_prompts(
                assignment_description=state["assignment_description"],
                rubric=state["generated_rubric"]
            )
            end_time = time.time()
            
            mlflow.log_metric("step3_prompt_generation_time", end_time - start_time)
            mlflow.log_metric("step3_prompts_count", len(prompt_set.prompts))
            
            # Save prompts as artifact
            prompts_data = {
                "assignment_description": prompt_set.assignment_description,
                "programming_language": prompt_set.programming_language,
                "general_prompt": prompt_set.general_prompt,
                "prompts": [
                    {
                        "rubric_item_id": prompt.rubric_item_id,
                        "rubric_item_title": prompt.rubric_item_title,
                        "prompt": prompt.prompt,
                        "criteria": prompt.criteria
                    }
                    for prompt in prompt_set.prompts
                ]
            }
            mlflow.log_dict(prompts_data, "step3_generated_prompts.json")
            
            logger.info(f"Generated {len(prompt_set.prompts)} correction prompts")
            return {"generated_prompts": prompt_set}
        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}")
            mlflow.log_param("step3_error", str(e))
            fallback_prompts = self._create_fallback_prompt_set()
            return {"generated_prompts": fallback_prompts}
    
    def _evaluate_prompts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Evaluate the generated prompts (LLM as judge)"""
        logger.info("Step 4: Evaluating generated prompts")
        
        if not state.get("generated_prompts"):
            return {"prompt_evaluation_score": 0.0}
        
        # This step simulates the "LLM as judge" from the flowchart
        try:
            start_time = time.time()
            # Simple evaluation based on prompt completeness
            total_prompts = len(state["generated_prompts"].prompts)
            score = min(10.0, total_prompts * 2.0)  # Simple scoring
            end_time = time.time()
            
            mlflow.log_metric("step4_prompt_evaluation_time", end_time - start_time)
            mlflow.log_metric("step4_prompt_evaluation_score", score)
            mlflow.log_metric("step4_total_prompts", total_prompts)
            
            # Save evaluation as artifact
            evaluation_data = {
                "prompt_evaluation_score": score,
                "total_prompts": total_prompts,
                "evaluation_method": "prompt_count_based",
                "timestamp": time.time()
            }
            mlflow.log_dict(evaluation_data, "step4_prompt_evaluation.json")
            
            logger.info(f"Prompt evaluation score: {score}")
            return {"prompt_evaluation_score": score}
        except Exception as e:
            logger.error(f"Failed to evaluate prompts: {e}")
            mlflow.log_param("step4_error", str(e))
            return {"prompt_evaluation_score": 7.0}  # Default score
    
    def _correct_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Evaluate student code using the prompts (Code Correction Agent)"""
        logger.info("Step 5: Evaluating student code")
        
        if not state.get("generated_prompts"):
            logger.error("No prompts available for code correction")
            fallback_result = self._create_fallback_correction_result()
            return {"correction_result": fallback_result}
        
        try:
            start_time = time.time()
            correction_result = self.code_corrector.correct_code(
                student_code=state["student_code"],
                prompt_set=state["generated_prompts"]
            )
            end_time = time.time()
            
            mlflow.log_metric("step5_code_evaluation_time", end_time - start_time)
            mlflow.log_metric("step5_grade_percentage", correction_result.grade_percentage)
            mlflow.log_metric("step5_total_score", correction_result.total_score)
            
            # Save evaluation results as artifact
            evaluation_data = {
                "student_code": correction_result.student_code,
                "assignment_description": correction_result.assignment_description,
                "programming_language": correction_result.programming_language,
                "grade_percentage": correction_result.grade_percentage,
                "total_score": correction_result.total_score,
                "max_possible_score": correction_result.max_possible_score,
                "summary": correction_result.summary,
                "item_evaluations": [
                    {
                        "rubric_item_id": eval.rubric_item_id,
                        "rubric_item_title": eval.rubric_item_title,
                        "total_score": eval.total_score,
                        "max_score": eval.max_score,
                        "overall_feedback": eval.overall_feedback,
                        "criteria_evaluations": [
                            {
                                "name": criterion.name,
                                "met": criterion.met,
                                "score": criterion.score,
                                "feedback": criterion.feedback,
                                "suggestion": criterion.suggestion
                            }
                            for criterion in eval.criteria_evaluations
                        ]
                    }
                    for eval in correction_result.item_evaluations
                ],
                "comprehensive_evaluation": {
                    "correctness": correction_result.comprehensive_evaluation.correctness,
                    "quality": correction_result.comprehensive_evaluation.quality,
                    "documentation": correction_result.comprehensive_evaluation.documentation,
                    "error_handling": correction_result.comprehensive_evaluation.error_handling,
                    "strengths": correction_result.comprehensive_evaluation.strengths,
                    "areas_for_improvement": correction_result.comprehensive_evaluation.areas_for_improvement,
                    "suggestions": correction_result.comprehensive_evaluation.suggestions,
                    "learning_resources": correction_result.comprehensive_evaluation.learning_resources
                }
            }
            mlflow.log_dict(evaluation_data, "step5_evaluation_results.json")
            
            logger.info(f"Code evaluation completed. Score: {correction_result.grade_percentage:.1f}%")
            return {"correction_result": correction_result}
        except Exception as e:
            logger.error(f"Failed to correct code: {e}")
            mlflow.log_param("step5_error", str(e))
            fallback_result = self._create_fallback_correction_result()
            return {"correction_result": fallback_result}
    
    def _aggregate_final_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Aggregate all results into final report (Results Aggregator)"""
        logger.info("Step 6: Aggregating final results")
        
        try:
            final_result = EvaluationPipeline(
                assignment_description=state["assignment_description"],
                programming_language=state["programming_language"],
                rubric=state["generated_rubric"],
                prompt_set=state["generated_prompts"],
                correction_result=state["correction_result"],
                metadata={
                    "rubric_evaluation_score": state["rubric_evaluation_score"],
                    "prompt_evaluation_score": state["prompt_evaluation_score"],
                    "pipeline_version": "1.0",
                    "agents_used": ["requirement_generator", "prompt_generator", "code_corrector"]
                }
            )
            logger.info("Final results aggregated successfully")
            return {"final_result": final_result}
        except Exception as e:
            logger.error(f"Failed to aggregate final results: {e}")
            fallback_pipeline = self._create_fallback_pipeline(
                state["assignment_description"], state["student_code"], state["programming_language"]
            )
            return {"final_result": fallback_pipeline}
    
    def evaluate_assignment(
        self,
        assignment_description: str,
        student_code: str,
        programming_language: str = "python"
    ) -> EvaluationPipeline:
        """Main method to evaluate a student assignment"""
        
        logger.info("Starting assignment evaluation pipeline")
        
        # Start MLflow run
        with mlflow.start_run(run_name="assignment_evaluation") as run:
            mlflow.log_param("programming_language", programming_language)
            mlflow.log_param("assignment_length", len(assignment_description))
            mlflow.log_param("code_length", len(student_code))
            
            start_time = time.time()
            
            # Initialize state as a dictionary
            initial_state = {
                "assignment_description": assignment_description,
                "student_code": student_code,
                "programming_language": programming_language,
                "generated_rubric": None,
                "generated_prompts": None,
                "correction_result": None,
                "rubric_evaluation_score": 0.0,
                "prompt_evaluation_score": 0.0,
                "final_result": None
            }
            
            # Run the complete pipeline
            result = self.graph.invoke(initial_state)
            
            # Log metrics
            end_time = time.time()
            execution_time = end_time - start_time
            mlflow.log_metric("execution_time_seconds", execution_time)
            
            final_result = result.get("final_result") or self._create_fallback_pipeline(
                assignment_description, student_code, programming_language
            )
            
            # Log evaluation results
            if final_result and hasattr(final_result, 'correction_result'):
                mlflow.log_metric("grade_percentage", final_result.correction_result.grade_percentage)
                mlflow.log_metric("total_score", final_result.correction_result.total_score)
                mlflow.log_metric("max_possible_score", final_result.correction_result.max_possible_score)
            
            # Log the run ID for reference
            logger.info(f"MLflow run ID: {run.info.run_id}")
            
            return final_result
    
    def _create_fallback_rubric(self):
        """Create a fallback rubric when generation fails"""
        from ..agents.requirement_generator.requirement_generator import Rubric, RubricItem
        
        return Rubric(
            title="Fallback Rubric",
            description="Basic rubric for evaluation",
            total_score=100,
            programming_language="python",
            difficulty_level="beginner",
            items=[
                RubricItem(
                    id="basic_functionality",
                    title="Basic Functionality",
                    description="Code runs without errors",
                    max_score=50,
                    criteria=["Code executes without syntax errors", "Produces some output"]
                ),
                RubricItem(
                    id="code_quality",
                    title="Code Quality",
                    description="Code is readable and well-structured",
                    max_score=30,
                    criteria=["Clear variable names", "Proper formatting"]
                ),
                RubricItem(
                    id="documentation",
                    title="Documentation",
                    description="Code has basic documentation",
                    max_score=20,
                    criteria=["Functions have docstrings", "Code is self-explanatory"]
                )
            ]
        )
    
    def _create_fallback_prompt_set(self):
        """Create a fallback prompt set when generation fails"""
        from ..agents.prompt_generator.prompt_generator import PromptSet, CorrectionPrompt
        
        return PromptSet(
            assignment_description="Fallback assignment",
            programming_language="python",
            prompts=[
                CorrectionPrompt(
                    rubric_item_id="basic_functionality",
                    rubric_item_title="Basic Functionality",
                    prompt="Evaluate if the code runs without errors and produces output.",
                    criteria=["Code executes", "Produces output"],
                    max_score=50
                )
            ],
            general_prompt="Provide a basic evaluation of the student's code."
        )
    
    def _create_fallback_correction_result(self):
        """Create a fallback correction result when evaluation fails"""
        from ..agents.code_corrector.code_corrector import CorrectionResult, ComprehensiveEvaluation
        
        return CorrectionResult(
            student_code="Code evaluation failed",
            assignment_description="Fallback assignment",
            programming_language="python",
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
    
    def _create_fallback_pipeline(
        self,
        assignment_description: str,
        student_code: str,
        programming_language: str
    ) -> EvaluationPipeline:
        """Create a fallback pipeline when the main pipeline fails"""
        return EvaluationPipeline(
            assignment_description=assignment_description,
            programming_language=programming_language,
            rubric=self._create_fallback_rubric(),
            prompt_set=self._create_fallback_prompt_set(),
            correction_result=self._create_fallback_correction_result(),
            metadata={
                "rubric_evaluation_score": 0.0,
                "prompt_evaluation_score": 0.0,
                "pipeline_version": "1.0",
                "agents_used": ["fallback"],
                "error": "Main pipeline failed, using fallback"
            }
        )
