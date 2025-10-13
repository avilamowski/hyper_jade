"""
Supervised Evaluator

Compares generated corrections with reference corrections using LangSmith's evaluation framework.
"""

from __future__ import annotations
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import load_config, get_agent_config
from ..models import Correction, Submission, Requirement

logger = logging.getLogger(__name__)


class SupervisedEvaluator:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None):
        self.config = config or load_config()
        self.evaluator_config = get_agent_config(self.config, "agent_evaluator")
        
        # Hardcoded criteria for supervised evaluation
        self.criteria = {
            "completeness": 1.0,
            "trigger_happy": 1.0,
            "false_positives": 1.0,
            "content_similarity": 1.0,
            "internal_content_correctness": 1.0,
            "external_content_correctness": 1.0
        }
        
        # Allow passing an existing LLM instance so tracing context is shared
        if llm is not None:
            self.llm = llm
        else:
            self.llm = self._setup_llm()
        
        repo_root = Path(__file__).resolve().parents[2]
        templates_dir = repo_root / "templates" / "evaluators"
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["jinja", "html", "txt"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        self.template = self.jinja_env.get_template("supervised_evaluator.jinja")
    
    def _setup_llm(self):
        provider = self.evaluator_config["provider"]
        model_name = self.evaluator_config["model_name"]
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            import os
            return ChatOpenAI(
                model=model_name,
                temperature=self.evaluator_config["temperature"],
                # max_tokens=self.evaluator_config["max_tokens"],
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        elif provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(
                model=model_name,
                temperature=self.evaluator_config["temperature"]
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _correction_to_text(self, correction: Correction) -> str:
        requirement_text = correction["requirement"]["requirement"]
        function_name = correction["requirement"]["function"]
        result_text = correction["result"]
        
        return f"Requirement: {requirement_text}\nFunction: {function_name}\nFeedback: {result_text}"
    
    def _render_evaluation_prompt(self, generated_text: str, reference_text: str, submission_code: str) -> tuple[str, str]:
        rendered = self.template.render(
            generated_correction=generated_text,
            reference_correction=reference_text,
            student_code=submission_code
        )
        
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        else:
            raise ValueError("Template must contain '---HUMAN---' separator")
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        scores = {}
        explanations = {}
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and any(criterion in line.lower() for criterion in self.criteria.keys()):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0].strip().lower().replace(' ', '_')
                    value_part = parts[1].strip()
                    
                    score_str = value_part
                    explanation = ""
                    
                    if ' - ' in value_part:
                        score_str, explanation = value_part.split(' - ', 1)
                        explanation = explanation.strip()
                    
                    try:
                        score = float(score_str.strip())
                        for criterion in self.criteria.keys():
                            if criterion in key_part or key_part in criterion:
                                scores[criterion] = score
                                if explanation:
                                    explanations[criterion] = explanation
                                break
                    except ValueError:
                        continue
        
        return {"scores": scores, "explanations": explanations}

    def evaluate_corrections(
        self,
        generated_corrections: List[List[Correction]],
        reference_corrections: List[str],
        submissions: List[Submission],
        requirements: List[Requirement]
    ) -> Dict[str, Any]:
        
        if len(generated_corrections) != len(reference_corrections):
            raise ValueError("Number of generated corrections must match reference corrections")
        
        if len(generated_corrections) != len(submissions):
            raise ValueError("Number of generated corrections must match submissions")
        
        evaluation_results = []
        total_scores = []
        
        for i, (gen_corr, ref_corr, submission) in enumerate(
            zip(generated_corrections, reference_corrections, submissions)
        ):
            try:
                result = self._evaluate_single_correction(gen_corr, ref_corr, submission)
                result["submission_index"] = i
                evaluation_results.append(result)
                
                if "overall_score" in result:
                    total_scores.append(result["overall_score"])
                    
            except Exception as e:
                logger.error(f"Failed to evaluate submission {i+1}: {e}")
                error_result = {
                    "submission_index": i,
                    "error": str(e),
                    "overall_score": 0.0,
                    "scores": {}
                }
                evaluation_results.append(error_result)
                total_scores.append(0.0)
        
        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        
        criterion_averages = {}
        if evaluation_results:
            for criterion in self.criteria.keys():
                criterion_scores = [
                    result["scores"].get(criterion, 0.0) 
                    for result in evaluation_results 
                    if "scores" in result and criterion in result["scores"]
                ]
                if criterion_scores:
                    criterion_averages[criterion] = sum(criterion_scores) / len(criterion_scores)
        
        return {
            "overall_score": avg_score,
            "criterion_averages": criterion_averages,
            "individual_evaluations": evaluation_results,
            "metadata": {
                "total_submissions": len(submissions),
                "total_requirements": len(requirements),
                "evaluation_timestamp": time.time(),
                "evaluator_model": self.evaluator_config["model_name"],
                "evaluator_provider": self.evaluator_config["provider"],
                "criteria_used": list(self.criteria.keys())
            }
        }
    
    def _evaluate_single_correction(
        self, 
        generated_correction: List[Correction], 
        reference_correction: str,
        submission: Submission
    ) -> Dict[str, Any]:
        
        generated_text_parts = []
        for correction in generated_correction:
            generated_text_parts.append(self._correction_to_text(correction))
        
        generated_text = "\n\n".join(generated_text_parts)
        
        system_prompt, human_prompt = self._render_evaluation_prompt(
            generated_text, reference_correction, submission["code"]
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        evaluation_time = time.time() - start_time
        
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response)
        
        parsed_result = self._parse_evaluation_response(response_content)
        scores = parsed_result["scores"]
        explanations = parsed_result["explanations"]
        
        if scores:
            weighted_sum = sum(score * self.criteria.get(criterion, 1.0) for criterion, score in scores.items())
            total_weight = sum(self.criteria.get(criterion, 1.0) for criterion in scores.keys())
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = 0.0
        
        return {
            "scores": scores,
            "explanations": explanations,
            "overall_score": overall_score,
            "evaluation_time": evaluation_time,
            "generated_text_length": len(generated_text),
            "reference_text_length": len(reference_correction),
            "raw_response": response_content
        }


def evaluate_supervised_correction(inputs: Dict[str, str], outputs: Dict[str, str], config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    
    student_code = inputs.get("student_code", "")
    reference_correction = inputs.get("reference_correction", "")
    generated_correction = outputs.get("generated_correction", "")
    
    evaluator = SupervisedEvaluator(config_path and load_config(config_path))
    
    submission: Submission = {"code": student_code}
    
    mock_correction: Correction = {
        "requirement": {
            "requirement": "General feedback evaluation",
            "function": "",
            "type": None
        },
        "result": generated_correction
    }
    
    try:
        result = evaluator._evaluate_single_correction([mock_correction], reference_correction, submission)
        
        langsmith_results = []
        for criterion, score in result.get("scores", {}).items():
            explanation = result.get("explanations", {}).get(criterion, f"Evaluated {criterion.replace('_', ' ')} aspect")
            langsmith_results.append({
                "key": criterion,
                "score": float(score),
                "max_score": 5.0,
                "rationale": explanation
            })
        
        langsmith_results.append({
            "key": "overall_score",
            "score": float(result.get("overall_score", 0.0)),
            "max_score": 5.0,
            "rationale": "Weighted average of all criteria"
        })
        
        return langsmith_results
        
    except Exception as e:
        logger.error(f"Error in supervised evaluation: {e}")
        return [{
            "key": "error",
            "score": 0.0,
            "max_score": 5.0,
            "rationale": f"Evaluation failed: {str(e)}"
        }]