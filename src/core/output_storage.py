#!/usr/bin/env python3
"""
Output Storage System

This module provides functionality to save and load intermediate results
from each agent in the pipeline, enabling independent agent execution.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OutputStorage:
    """Manages storage and retrieval of agent outputs"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each agent
        (self.output_dir / "requirement_generator").mkdir(exist_ok=True)
        (self.output_dir / "prompt_generator").mkdir(exist_ok=True)
        (self.output_dir / "code_corrector").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
    
    def save_rubric(self, rubric: Any, assignment_id: str, metadata: Dict[str, Any] = None) -> str:
        """Save rubric from requirement generator"""
        timestamp = datetime.now().isoformat()
        filename = f"rubric_{assignment_id}_{timestamp}.json"
        filepath = self.output_dir / "requirement_generator" / filename
        
        # Convert rubric to serializable format
        rubric_data = {
            "title": rubric.title,
            "description": rubric.description,
            "programming_language": rubric.programming_language,
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
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rubric_data, f, indent=2)
        
        # Save metadata
        if metadata:
            self._save_metadata("requirement_generator", assignment_id, timestamp, metadata)
            
        # Log MLflow run ID if available
        if metadata and "mlflow_run_id" in metadata:
            logger.info(f"MLflow run ID: {metadata['mlflow_run_id']}")
        
        logger.info(f"Saved rubric to {filepath}")
        return str(filepath)
    
    def load_rubric(self, filepath: str) -> Any:
        """Load rubric from file"""
        from ..agents.requirement_generator.requirement_generator import Rubric, RubricItem
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = [
            RubricItem(
                id=item["id"],
                title=item["title"],
                description=item["description"],
                criteria=item["criteria"]
            )
            for item in data["items"]
        ]
        
        rubric = Rubric(
            title=data["title"],
            description=data["description"],
            programming_language=data["programming_language"],
            items=items
        )
        
        return rubric
    
    def save_prompts(self, prompt_set: Any, assignment_id: str, metadata: Dict[str, Any] = None) -> str:
        """Save prompts from prompt generator"""
        timestamp = datetime.now().isoformat()
        filename = f"prompts_{assignment_id}_{timestamp}.json"
        filepath = self.output_dir / "prompt_generator" / filename
        
        # Convert prompt set to serializable format
        prompts_data = {
            "assignment_description": prompt_set.assignment_description,
            "programming_language": prompt_set.programming_language,
            "general_prompt": prompt_set.general_prompt,
            "prompts": [
                {
                    "rubric_item_id": prompt.rubric_item_id,
                    "rubric_item_title": prompt.rubric_item_title,
                    "prompt": prompt.prompt,
                    "criteria": prompt.criteria,
                    "examples": getattr(prompt, 'examples', None),
                    "resources": getattr(prompt, 'resources', None)
                }
                for prompt in prompt_set.prompts
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts_data, f, indent=2)
        
        # Save metadata
        if metadata:
            self._save_metadata("prompt_generator", assignment_id, timestamp, metadata)
            
        # Log MLflow run ID if available
        if metadata and "mlflow_run_id" in metadata:
            logger.info(f"MLflow run ID: {metadata['mlflow_run_id']}")
        
        logger.info(f"Saved prompts to {filepath}")
        return str(filepath)
    
    def load_prompts(self, filepath: str) -> Any:
        """Load prompts from file"""
        from ..agents.prompt_generator.prompt_generator import PromptSet, CorrectionPrompt
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = [
            CorrectionPrompt(
                rubric_item_id=prompt["rubric_item_id"],
                rubric_item_title=prompt["rubric_item_title"],
                prompt=prompt["prompt"],
                criteria=prompt["criteria"],
                examples=prompt.get("examples"),
                resources=prompt.get("resources")
            )
            for prompt in data["prompts"]
        ]
        
        prompt_set = PromptSet(
            assignment_description=data["assignment_description"],
            programming_language=data["programming_language"],
            prompts=prompts,
            general_prompt=data["general_prompt"]
        )
        
        return prompt_set
    
    def save_correction_result(self, result: Any, assignment_id: str, metadata: Dict[str, Any] = None) -> str:
        """Save correction result from code corrector"""
        timestamp = datetime.now().isoformat()
        filename = f"correction_{assignment_id}_{timestamp}.json"
        filepath = self.output_dir / "code_corrector" / filename
        
        # Convert result to serializable format
        result_data = {
            "student_code": result.student_code,
            "assignment_description": result.assignment_description,
            "programming_language": result.programming_language,
            "total_errors": result.total_errors,
            "critical_errors": result.critical_errors,
            "summary": result.summary,
            "item_evaluations": [
                {
                    "rubric_item_id": eval.rubric_item_id,
                    "rubric_item_title": eval.rubric_item_title,
                    "is_passing": eval.is_passing,
                    "overall_feedback": eval.overall_feedback,
                    "errors_found": [
                        {
                            "error_type": error.error_type,
                            "location": error.location,
                            "description": error.description,
                            "severity": error.severity,
                            "suggestion": error.suggestion,
                            "line_number": error.line_number
                        }
                        for error in eval.errors_found
                    ]
                }
                for eval in result.item_evaluations
            ],
            "comprehensive_evaluation": {
                "correctness": result.comprehensive_evaluation.correctness,
                "quality": result.comprehensive_evaluation.quality,
                "error_handling": result.comprehensive_evaluation.error_handling,
                "strengths": result.comprehensive_evaluation.strengths,
                "areas_for_improvement": result.comprehensive_evaluation.areas_for_improvement,
                "suggestions": result.comprehensive_evaluation.suggestions,
                "learning_resources": result.comprehensive_evaluation.learning_resources
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        # Save metadata
        if metadata:
            self._save_metadata("code_corrector", assignment_id, timestamp, metadata)
            
        # Log MLflow run ID if available
        if metadata and "mlflow_run_id" in metadata:
            logger.info(f"MLflow run ID: {metadata['mlflow_run_id']}")
        
        logger.info(f"Saved correction result to {filepath}")
        return str(filepath)
    
    def load_correction_result(self, filepath: str) -> Any:
        """Load correction result from file"""
        from ..agents.code_corrector.code_corrector import (
            CorrectionResult, ItemEvaluation, ErrorIdentification, ComprehensiveEvaluation
        )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        item_evaluations = [
            ItemEvaluation(
                rubric_item_id=eval_data["rubric_item_id"],
                rubric_item_title=eval_data["rubric_item_title"],
                is_passing=eval_data["is_passing"],
                overall_feedback=eval_data["overall_feedback"],
                errors_found=[
                    ErrorIdentification(
                        error_type=error["error_type"],
                        location=error["location"],
                        description=error["description"],
                        severity=error["severity"],
                        suggestion=error["suggestion"],
                        line_number=error.get("line_number")
                    )
                    for error in eval_data["errors_found"]
                ]
            )
            for eval_data in data["item_evaluations"]
        ]
        
        comprehensive_eval = ComprehensiveEvaluation(
            correctness=data["comprehensive_evaluation"]["correctness"],
            quality=data["comprehensive_evaluation"]["quality"],
            error_handling=data["comprehensive_evaluation"]["error_handling"],
            strengths=data["comprehensive_evaluation"]["strengths"],
            areas_for_improvement=data["comprehensive_evaluation"]["areas_for_improvement"],
            suggestions=data["comprehensive_evaluation"]["suggestions"],
            learning_resources=data["comprehensive_evaluation"]["learning_resources"]
        )
        
        result = CorrectionResult(
            student_code=data["student_code"],
            assignment_description=data["assignment_description"],
            programming_language=data["programming_language"],
            item_evaluations=item_evaluations,
            comprehensive_evaluation=comprehensive_eval,
            total_errors=data["total_errors"],
            critical_errors=data["critical_errors"],
            summary=data["summary"]
        )
        
        return result
    
    def _save_metadata(self, agent_name: str, assignment_id: str, timestamp: str, metadata: Dict[str, Any]):
        """Save metadata for an agent run"""
        metadata_file = self.output_dir / "metadata" / f"{agent_name}_{assignment_id}_{timestamp}.json"
        metadata["timestamp"] = timestamp
        metadata["assignment_id"] = assignment_id
        metadata["agent"] = agent_name
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def list_outputs(self, agent_name: Optional[str] = None) -> Dict[str, list]:
        """List all available outputs"""
        outputs = {}
        
        if agent_name:
            agent_dirs = [self.output_dir / agent_name]
        else:
            agent_dirs = [
                self.output_dir / "requirement_generator",
                self.output_dir / "prompt_generator", 
                self.output_dir / "code_corrector"
            ]
        
        for agent_dir in agent_dirs:
            if agent_dir.exists():
                agent_name = agent_dir.name
                outputs[agent_name] = []
                for file in agent_dir.glob("*.json"):
                    outputs[agent_name].append(str(file))
        
        return outputs
    
    def get_latest_output(self, agent_name: str, assignment_id: str) -> Optional[str]:
        """Get the latest output file for a specific agent and assignment"""
        agent_dir = self.output_dir / agent_name
        if not agent_dir.exists():
            return None
        
        # Find files matching the assignment_id pattern
        pattern = f"*{assignment_id}*.json"
        files = list(agent_dir.glob(pattern))
        
        if not files:
            return None
        
        # Return the most recent file
        return str(max(files, key=lambda f: f.stat().st_mtime))

