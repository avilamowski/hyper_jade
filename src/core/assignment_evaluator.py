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
import time
from pathlib import Path

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
        graph.add_node("generate_prompts", self._generate_prompts)
        graph.add_node("correct_code", self._correct_code)
        graph.add_node("aggregate_final_results", self._aggregate_final_results)
        
        # Set entry point
        graph.set_entry_point("generate_requirements")
        
        # Add edges following the flowchart
        graph.add_edge("generate_requirements", "generate_prompts")
        graph.add_edge("generate_prompts", "correct_code")
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
            
            logger.info(f"Generated rubric with {len(rubric.items)} items")
            return {"generated_rubric": rubric}
        except Exception as e:
            logger.error(f"Failed to generate rubric: {e}")
            # Create a basic fallback rubric
            fallback_rubric = self._create_fallback_rubric()
            return {"generated_rubric": fallback_rubric}
    
    def _generate_prompts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Generate correction prompts for each rubric item (Prompt Generator Agent)"""
        logger.info("Step 2: Generating correction prompts")
        
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
            
            logger.info(f"Generated {len(prompt_set.prompts)} correction prompts")
            return {"generated_prompts": prompt_set}
        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}")
            fallback_prompts = self._create_fallback_prompt_set()
            return {"generated_prompts": fallback_prompts}
    
    def _correct_code(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Evaluate student code using the prompts (Code Correction Agent)"""
        logger.info("Step 3: Evaluating student code")
        
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
            
            logger.info(f"Code evaluation completed")
            return {"correction_result": correction_result}
        except Exception as e:
            logger.error(f"Failed to correct code: {e}")
            fallback_result = self._create_fallback_correction_result()
            return {"correction_result": fallback_result}
    
    def _aggregate_final_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Aggregate all results into final report"""
        logger.info("Step 4: Aggregating final results")
        
        try:
            final_result = EvaluationPipeline(
                assignment_description=state["assignment_description"],
                programming_language=state["programming_language"],
                rubric=state["generated_rubric"],
                prompt_set=state["generated_prompts"],
                correction_result=state["correction_result"],
                metadata={
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
        
        start_time = time.time()
        
        # Initialize state as a dictionary
        initial_state = {
            "assignment_description": assignment_description,
            "student_code": student_code,
            "programming_language": programming_language,
            "generated_rubric": None,
            "generated_prompts": None,
            "correction_result": None,
            "final_result": None
        }
        
        # Run the complete pipeline
        result = self.graph.invoke(initial_state)
        
        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
        
        final_result = result.get("final_result") or self._create_fallback_pipeline(
            assignment_description, student_code, programming_language
        )
        
        return final_result
    
    def _create_fallback_rubric(self):
        """Create a fallback rubric when generation fails"""
        from ..agents.requirement_generator.requirement_generator import Rubric, RubricItem
        
        return Rubric(
            title="Fallback Rubric",
            description="Basic rubric for evaluation",
            programming_language="python",
            items=[
                RubricItem(
                    id="basic_functionality",
                    title="Basic Functionality",
                    description="Code runs without errors",
                    criteria=["Code executes without syntax errors", "Produces some output"]
                ),
                RubricItem(
                    id="code_quality",
                    title="Code Quality",
                    description="Code is readable and well-structured",
                    criteria=["Clear variable names", "Proper formatting"]
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
                    criteria=["Code executes", "Produces output"]
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
                error_handling="Evaluation failed",
                strengths=["Code submitted for evaluation"],
                areas_for_improvement=["Manual review required"],
                suggestions=["Please review the code manually"],
                learning_resources=["General programming resources"]
            ),
            total_errors=0,
            critical_errors=0,
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
                "pipeline_version": "1.0",
                "agents_used": ["fallback"],
                "error": "Main pipeline failed, using fallback"
            }
        )
