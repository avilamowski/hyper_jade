"""
Two-step Supervised Evaluator

Step 1: compute auxiliary metrics (counts) via an LLM call using a dedicated template.
Step 2: compute final evaluation metrics using the auxiliary metrics as an extra input via a second LLM call.
"""
from __future__ import annotations
import logging
import time
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import load_config
from ..models import Correction, Submission, Requirement

logger = logging.getLogger(__name__)


class SupervisedEvaluator2Step:
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm: Optional[Any] = None):
        if config is not None:
            self.config = config
        else:
            # Load evaluator-specific config
            self.config = load_config("src/config/evaluator_config.yaml")

        # For evaluator config, settings are at the top level, not under agents
        self.evaluator_config = {
            'model_name': self.config['model_name'],
            'provider': self.config['provider'],
            'temperature': self.config['temperature'],
        }

        # same criteria/weights as the single-step evaluator
        self.criteria = {
            "completeness": 1.0,
            "restraint": 1.0,
            "precision": 1.0,
            "content_similarity": 1.0,
            "internal_correctness": 1.0,
            "external_correctness": 1.0,
        }

        self._synonym_map = {
            "trigger_happy": "restraint",
            "restraint": "restraint",
            "false_positives": "precision",
            "precision": "precision",
            "internal_content_correctness": "internal_correctness",
            "internal_correctness": "internal_correctness",
            "external_content_correctness": "external_correctness",
            "external_correctness": "external_correctness",
            "content_similarity": "content_similarity",
            "completeness": "completeness",
        }

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

        # Load templates
        self.aux_template = self.jinja_env.get_template("supervised_evaluator2step-aux.jinja")
        self.main_template = self.jinja_env.get_template("supervised_evaluator2step.jinja")

    def _setup_llm(self):
        provider = self.evaluator_config["provider"]
        model_name = self.evaluator_config["model_name"]

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            import os
            return ChatOpenAI(
                model=model_name,
                temperature=self.evaluator_config["temperature"],
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        elif provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(model=model_name, temperature=self.evaluator_config["temperature"])
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _extract_result_text(self, raw_response) -> str:
        if hasattr(raw_response, 'content'):
            text = raw_response.content
        elif isinstance(raw_response, dict):
            choices = raw_response.get('choices')
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                text = (first.get('message', {}) or {}).get('content') or first.get('text') or str(raw_response)
            else:
                text = raw_response.get('text') or str(raw_response)
        else:
            text = str(raw_response)

        m = re.search(r"<RESULT>(.*?)</RESULT>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(0).strip()
        return text.strip()

    def _parse_auxiliary_metrics(self, aux_text: str) -> Dict[str, Any]:
        """Parse the three auxiliary metric lines produced by the aux template.

        Expected lines like:
        missing_reference_issues: 2 - ... Cases: "..."
        extra_ai_issues: 3 - ...
        false_positive_issues: 1 - ...
        """
        metrics: Dict[str, Any] = {}
        lines = [l.strip() for l in aux_text.strip().splitlines() if l.strip()]
        for line in lines:
            if ':' not in line:
                continue
            key, rest = line.split(':', 1)
            key = key.strip()
            # value may have ' - ' and explanation
            num = None
            try:
                num_str = rest.strip().split(' ', 1)[0]
                num = int(num_str)
            except Exception:
                # try to find first integer
                m = re.search(r"(\d+)", rest)
                if m:
                    num = int(m.group(1))

            explanation = ''
            if ' - ' in rest:
                explanation = rest.split(' - ', 1)[1].strip()

            metrics[key] = {"count": num if num is not None else 0, "explanation": explanation}

        return metrics

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        # Reuse parsing logic similar to single-step evaluator for XML or key: value lines
        scores = {}
        explanations = {}
        import xml.etree.ElementTree as ElementTree

        m = re.search(r"<RESULT>(.*?)</RESULT>", response_text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            xml_text = m.group(1)
            try:
                root = ElementTree.fromstring(f"<root>{xml_text}</root>")
                for child in root:
                    tag = (child.tag.split('}')[-1]).strip().lower().replace(' ', '_')
                    score_text = None
                    explanation_text = None
                    for sub in list(child):
                        subtag = (sub.tag.split('}')[-1]).strip().lower()
                        if subtag == 'score' and sub.text:
                            score_text = sub.text.strip()
                        elif subtag == 'explanation' and sub.text:
                            explanation_text = sub.text.strip()

                    if score_text is None:
                        if child.text and child.text.strip():
                            score_text = child.text.strip()

                    if score_text is not None:
                        try:
                            val = float(score_text)
                        except ValueError:
                            continue
                        canonical = self._synonym_map.get(tag, tag)
                        if canonical == 'overall_score':
                            continue
                        if canonical in self.criteria:
                            scores[canonical] = val
                            if explanation_text:
                                explanations[canonical] = explanation_text
                if scores:
                    return {"scores": scores, "explanations": explanations}
            except ElementTree.ParseError:
                pass

        # fallback to simple line parsing
        for line in response_text.strip().splitlines():
            line = line.strip()
            if ':' not in line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            raw_key = parts[0].strip().lower()
            key_part = raw_key.replace('-', ' ').replace('  ', ' ').strip().replace(' ', '_')
            value_part = parts[1].strip()

            score_str = value_part
            explanation = ''
            if ' - ' in value_part:
                score_str, explanation = value_part.split(' - ', 1)
                explanation = explanation.strip()

            canonical = None
            for syn, canon in self._synonym_map.items():
                if syn in key_part or key_part in syn:
                    canonical = canon
                    break

            if canonical is None:
                for criterion in self.criteria.keys():
                    if criterion in key_part or key_part in criterion:
                        canonical = criterion
                        break

            if canonical is None:
                continue

            try:
                score = float(score_str.strip())
                scores[canonical] = score
                if explanation:
                    explanations[canonical] = explanation
            except ValueError:
                continue

        return {"scores": scores, "explanations": explanations}

    def evaluate_single(
        self,
        generated_correction: List[Correction],
        reference_correction: str,
        submission: Submission,
        assignment: Optional[str] = None,
        requirements: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the two-step evaluation for a single submission/correction.

        Returns a dict containing parsed scores, explanations, auxiliary metrics, timing, and raw responses.
        """
        # Build generated text similar to single-step evaluator
        generated_text_parts = []
        for correction in generated_correction:
            result_text = (correction.get('result') or '').strip()
            
            # Extract RESULT tag content for filtering
            result_match = re.search(r"<RESULT>\s*(.+?)\s*</RESULT>", result_text, flags=re.IGNORECASE | re.DOTALL)
            if result_match:
                result_content = result_match.group(1).strip().upper()
                if result_content == 'NO ERROR':
                    continue  # Skip corrections with no errors
            
            # Extract EXPLANATION tag content (this is the feedback)
            explanation_match = re.search(r"<EXPLANATION>\s*(.+?)\s*</EXPLANATION>", result_text, flags=re.IGNORECASE | re.DOTALL)
            if explanation_match:
                explanation_text = explanation_match.group(1).strip()
            else:
                # Fallback: use full result_text if no EXPLANATION tag found (backward compatibility)
                explanation_text = result_text
            
            requirement_text = correction['requirement']['requirement']
            function_name = correction['requirement'].get('function', '')
            generated_text_parts.append(f"Requirement: {requirement_text}\nFunction: {function_name}\nFeedback: {explanation_text}")

        generated_text = "\n\n".join(generated_text_parts)

        # Render aux template and call LLM
        aux_rendered = self.aux_template.render(
            assignment=assignment or "",
            requirements=requirements or "",
            student_code=submission.get('code', ''),
            reference_correction=reference_correction,
            generated_correction=generated_text
        )

        # split system/human if template uses separator (aux template is only human portion)
        messages_aux = [SystemMessage(content="Auxiliary metrics computation"), HumanMessage(content=aux_rendered)]

        t0 = time.time()
        raw_aux_response = self.llm.invoke(messages_aux)
        aux_time = time.time() - t0

        aux_text = self._extract_result_text(raw_aux_response)
        aux_metrics = self._parse_auxiliary_metrics(aux_text)

        # Now render main template including auxiliary metrics block
        main_rendered = self.main_template.render(
            assignment=assignment or "",
            requirements=requirements or "",
            student_code=submission.get('code', ''),
            reference_correction=reference_correction,
            generated_correction=generated_text,
            auxiliary_metrics=aux_text
        )

        messages_main = [SystemMessage(content="Final evaluation"), HumanMessage(content=main_rendered)]
        t1 = time.time()
        raw_main_response = self.llm.invoke(messages_main)
        main_time = time.time() - t1

        main_text = self._extract_result_text(raw_main_response)
        parsed = self._parse_evaluation_response(main_text)

        scores = parsed.get('scores', {})
        explanations = parsed.get('explanations', {})

        if scores:
            weighted_sum = sum(score * self.criteria.get(criterion, 1.0) for criterion, score in scores.items())
            total_weight = sum(self.criteria.get(criterion, 1.0) for criterion in scores.keys())
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = 0.0

        return {
            'scores': scores,
            'explanations': explanations,
            'overall_score': overall_score,
            'auxiliary_metrics': aux_metrics,
            'timing': {'aux_call': aux_time, 'main_call': main_time},
            'raw_aux_response': aux_text,
            'raw_main_response': main_text,
            'extra': {
                'auxiliary_metrics_raw': aux_text,
                'aux_timing': aux_time,
                'main_timing': main_time
            }
        }

    def evaluate_corrections(
        self,
        generated_corrections: List[List[Correction]],
        reference_corrections: List[str],
        submissions: List[Submission],
        requirements: List[Requirement]
    ) -> Dict[str, Any]:
        """Batch evaluate multiple submissions/corrections using the 2-step flow.

        Returns a dict with overall_score, criterion_averages, individual_evaluations, and metadata.
        """
        if len(generated_corrections) != len(reference_corrections):
            raise ValueError("Number of generated corrections must match reference corrections")
        if len(generated_corrections) != len(submissions):
            raise ValueError("Number of generated corrections must match submissions")

        evaluation_results = []
        total_scores = []

        for i, (gen_corr, ref_corr, submission) in enumerate(zip(generated_corrections, reference_corrections, submissions)):
            try:
                result = self.evaluate_single(gen_corr, ref_corr, submission)
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
                    result.get("scores", {}).get(criterion, 0.0)
                    for result in evaluation_results
                    if isinstance(result, dict)
                ]
                # filter out missing zeros where the metric wasn't provided
                actual_scores = [s for s in criterion_scores if s is not None]
                if actual_scores:
                    criterion_averages[criterion] = sum(actual_scores) / len(actual_scores)

        # Aggregate extra data from all evaluations
        aggregated_extra = {}
        if evaluation_results:
            all_aux_metrics_raw = [result.get("extra", {}).get("auxiliary_metrics_raw", "") for result in evaluation_results if "extra" in result]
            if all_aux_metrics_raw:
                aggregated_extra["all_auxiliary_metrics_raw"] = all_aux_metrics_raw
            
            total_aux_time = sum([result.get("extra", {}).get("aux_timing", 0) for result in evaluation_results if "extra" in result])
            total_main_time = sum([result.get("extra", {}).get("main_timing", 0) for result in evaluation_results if "extra" in result])
            aggregated_extra["total_timing"] = {"aux_calls": total_aux_time, "main_calls": total_main_time}

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
            },
            "extra": aggregated_extra
        }


def evaluate_supervised_correction_2step(inputs: Dict[str, str], outputs: Dict[str, str], config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    student_code = inputs.get('student_code', '')
    reference_correction = inputs.get('reference_correction', '')
    generated_correction = outputs.get('generated_correction', '')

    # If config_path is provided, use it; otherwise use default evaluator config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config("src/config/evaluator_config.yaml")

    evaluator = SupervisedEvaluator2Step(config)

    submission: Submission = {'code': student_code}

    mock_correction: Correction = {
        'requirement': {'requirement': 'General feedback evaluation', 'function': '', 'type': None},
        'result': generated_correction
    }

    try:
        result = evaluator.evaluate_single([mock_correction], reference_correction, submission)

        langsmith_results = []
        for raw_criterion, score in result.get('scores', {}).items():
            try:
                canonical = evaluator._synonym_map.get(raw_criterion, raw_criterion)
            except Exception:
                canonical = raw_criterion

            explanation = result.get('explanations', {}).get(raw_criterion, result.get('explanations', {}).get(canonical, f"Evaluated {canonical.replace('_', ' ')} aspect"))
            langsmith_results.append({
                'key': canonical,
                'score': float(score),
                'max_score': 5.0,
                'rationale': explanation
            })

        langsmith_results.append({
            'key': 'overall_score',
            'score': float(result.get('overall_score', 0.0)),
            'max_score': 5.0,
            'rationale': 'Weighted average of all criteria (2-step)'
        })

        return langsmith_results

    except Exception as e:
        logger.error(f"Error in supervised 2-step evaluation: {e}")
        return [{
            'key': 'error',
            'score': 0.0,
            'max_score': 5.0,
            'rationale': f"Evaluation failed: {str(e)}"
        }]
