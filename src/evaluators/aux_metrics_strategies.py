"""
Auxiliary Metrics Strategies Module

Provides different strategies for computing auxiliary metrics (match, missing, extra).
Each strategy implements a different approach to classification, with varying
trade-offs between consistency, speed, and cost.

Available strategies:
- IndependentStrategy: Current behavior, runs all three metrics in parallel independently
- PerCorrectionStrategy: Classifies each correction individually, then aggregates
- DependentStrategy: Runs match first, then injects results into missing/extra prompts
"""

from __future__ import annotations
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

if TYPE_CHECKING:
    from .supervised_evaluator_aux import AuxiliaryMetricsEvaluator, AuxiliaryMetricsState

logger = logging.getLogger(__name__)


class AuxMetricsStrategy(ABC):
    """Abstract base class for auxiliary metrics computation strategies."""
    
    def __init__(self, evaluator: 'AuxiliaryMetricsEvaluator'):
        self.evaluator = evaluator
        self.llm = evaluator.llm
        self.jinja_env = evaluator.jinja_env
    
    @abstractmethod
    def compute(
        self,
        generated_text: str,
        reference_text: Union[str, Dict],
        student_code: str,
        assignment: str,
        requirements: List[Dict],
    ) -> Dict[str, str]:
        """
        Compute auxiliary metrics.
        
        Returns:
            Dict with keys 'match', 'missing', 'extra', each containing
            text output with embedded <AUXILIARY_METRIC_COUNT> blocks.
        """
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class IndependentStrategy(AuxMetricsStrategy):
    """
    Original strategy: runs match, missing, and extra in parallel independently.
    
    This is the default strategy for backward compatibility.
    Each metric is computed by a separate LLM call without knowledge of the others.
    """
    
    def compute(
        self,
        generated_text: str,
        reference_text: Union[str, Dict],
        student_code: str,
        assignment: str,
        requirements: List[Dict],
    ) -> Dict[str, str]:
        """Delegate to the evaluator's existing LangGraph-based implementation."""
        # This strategy uses the evaluator's existing graph
        # The evaluator will call this, but for independent strategy,
        # we just return None to signal "use default graph"
        # This is handled specially in the evaluator
        raise NotImplementedError(
            "IndependentStrategy uses the evaluator's built-in graph. "
            "This should be handled by the evaluator directly."
        )


class PerCorrectionStrategy(AuxMetricsStrategy):
    """
    Per-correction strategy: classifies each correction individually.
    
    Process:
    1. Parse human corrections from reference text
    2. Parse AI corrections from generated text
    3. For each human correction, prompt: "Does any AI correction match this?"
    4. AI corrections not matched are marked as EXTRA
    5. Aggregate results into expected output format
    
    Benefits:
    - More consistent classification (each correction judged in isolation)
    - No cross-contamination between match/missing/extra decisions
    
    Trade-offs:
    - More LLM calls (one per correction)
    - Slower execution
    """
    
    def __init__(self, evaluator: 'AuxiliaryMetricsEvaluator'):
        super().__init__(evaluator)
        # Get template from config
        self.template_config = evaluator.aux_metric_configs.get("classify", {})
        self.template_path = self.template_config.get("template", "evaluators/per_correction/aux_classify_correction.jinja")
    
    def compute(
        self,
        generated_text: str,
        reference_text: Union[str, Dict],
        student_code: str,
        assignment: str,
        requirements: List[Dict],
    ) -> Dict[str, str]:
        """Classify each correction individually, then aggregate."""
        
        # Parse corrections from inputs
        human_corrections = self._parse_human_corrections(reference_text)
        ai_corrections = self._parse_ai_corrections(generated_text, requirements)
        
        logger.info(f"PerCorrectionStrategy: {len(human_corrections)} human corrections, "
                   f"{len(ai_corrections)} AI corrections")
        
        if not human_corrections and not ai_corrections:
            return self._format_empty_results()
        
        # Track results
        matches = []  # {"requirement": ..., "human": ..., "ai": ..., "function": ..., "type": ...}
        missing = []  # {"requirement": ..., "human": ..., "function": ..., "type": ...}
        matched_ai_indices = set()
        
        # Load template from config
        template = self.jinja_env.get_template(self.template_path)
        
        # Pass 1: Classify each human correction
        for human_corr in human_corrections:
            classification = self._classify_human_correction(
                human_corr,
                ai_corrections,
                student_code,
                template
            )
            
            if classification["is_match"]:
                # Get the matched AI correction to extract its function/requirement
                matched_ai = classification.get("matched_ai_corr", {})
                matches.append({
                    # Use AI's requirement (from config) if available, else use human text
                    "requirement": matched_ai.get("requirement", human_corr.get("text", "")),
                    "function": matched_ai.get("function", "unknown"),
                    "type": matched_ai.get("type", "error_presence"),
                    "human": human_corr.get("text", str(human_corr)),
                    "ai": classification.get("matched_ai_text", ""),
                    "match_quality": classification.get("match_quality", "FULL")
                })
                if classification.get("matched_ai_index") is not None:
                    matched_ai_indices.add(classification["matched_ai_index"])
            else:
                missing.append({
                    "requirement": human_corr.get("requirement", human_corr.get("text", "")),
                    "function": human_corr.get("function", "unknown"),
                    "type": human_corr.get("type", "error_presence"),
                    "human": human_corr.get("text", str(human_corr)),
                    "impact_severity": classification.get("impact_severity", "MEDIUM")
                })
        
        # Pass 2: Find extra AI corrections (not matched)
        extras = []
        for i, ai_corr in enumerate(ai_corrections):
            if i not in matched_ai_indices:
                extras.append({
                    "ai": ai_corr.get("text", str(ai_corr)),
                    "relevance": "RELEVANT"  # Default, could be classified
                })
        
        # Aggregate into expected format
        return {
            "match": self._format_match_output(matches),
            "missing": self._format_missing_output(missing),
            "extra": self._format_extra_output(extras)
        }
    
    def _parse_human_corrections(self, reference_text: Union[str, Dict]) -> List[Dict]:
        """Parse human corrections from reference text."""
        corrections = []
        
        if isinstance(reference_text, dict) and 'corrections' in reference_text:
            # ReferenceCorrection format
            for corr in reference_text['corrections']:
                corrections.append({
                    "text": corr if isinstance(corr, str) else str(corr),
                    "requirement": corr if isinstance(corr, str) else str(corr),
                    "function": "unknown",
                    "type": "error_presence"
                })
        elif isinstance(reference_text, str):
            # Plain text - split by lines or markers
            # Look for <human> tags first
            human_matches = re.findall(r'<human>(.*?)</human>', reference_text, re.DOTALL | re.IGNORECASE)
            if human_matches:
                for text in human_matches:
                    text = text.strip()
                    if text:
                        corrections.append({
                            "text": text,
                            "requirement": text,
                            "function": "unknown",
                            "type": "error_presence"
                        })
            else:
                # Split by double newlines or numbered items
                parts = re.split(r'\n\n+', reference_text.strip())
                for part in parts:
                    part = part.strip()
                    if part:
                        corrections.append({
                            "text": part,
                            "requirement": part,
                            "function": "unknown",
                            "type": "error_presence"
                        })
        
        return corrections
    
    def _parse_ai_corrections(self, generated_text: str, requirements: List[Dict]) -> List[Dict]:
        """
        Parse AI corrections from generated text.
        
        IMPORTANT: Only includes corrections with ERROR FOUND result.
        Corrections with NO ERROR are excluded since they're not actual errors.
        
        Returns list of dicts with keys: text, function, requirement
        """
        corrections = []
        
        if not generated_text:
            return corrections
        
        # Build a lookup dict from requirements by function name
        req_by_function = {}
        for req in requirements:
            func = req.get("function", "")
            if func:
                req_by_function[func.lower()] = req
        
        # Look for <generated> tags
        gen_matches = re.findall(r'<generated>(.*?)</generated>', generated_text, re.DOTALL | re.IGNORECASE)
        if gen_matches:
            for text in gen_matches:
                text = text.strip()
                if text:
                    # Only include if it's an ERROR FOUND correction
                    if self._is_error_correction(text):
                        func_name = self._extract_function_name(text)
                        req = req_by_function.get(func_name.lower(), {})
                        corrections.append({
                            "text": text,
                            "function": func_name,
                            "requirement": req.get("requirement", func_name),
                            "type": req.get("type", "error_presence")
                        })
        else:
            # Split by ## headers (function names)
            sections = re.split(r'##\s+', generated_text.strip())
            for section in sections:
                section = section.strip()
                if section:
                    # Only include if it's an ERROR FOUND correction
                    if self._is_error_correction(section):
                        func_name = self._extract_function_name(section)
                        req = req_by_function.get(func_name.lower(), {})
                        # Extract requirement from <requirement> tag if present
                        req_match = re.search(r'<requirement>(.*?)</requirement>', section, re.DOTALL | re.IGNORECASE)
                        if req_match:
                            requirement_text = req_match.group(1).strip()
                        else:
                            requirement_text = req.get("requirement", func_name)
                        corrections.append({
                            "text": section,
                            "function": func_name,
                            "requirement": requirement_text,
                            "type": req.get("type", "error_presence")
                        })
        
        return corrections
    
    def _extract_function_name(self, text: str) -> str:
        """Extract function name from the first line of AI correction text."""
        # The AI correction typically starts with the function name on the first line
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Remove any markdown formatting
            first_line = re.sub(r'^#+\s*', '', first_line)
            # Take only the first word (function name)
            words = first_line.split()
            if words:
                return words[0]
        return "unknown"
    
    def _is_error_correction(self, text: str) -> bool:
        """
        Check if the AI correction text contains ERROR FOUND.
        Returns False if it contains NO ERROR, True otherwise.
        """
        # Look for <RESULT> tag
        result_match = re.search(r'<RESULT>(.*?)</RESULT>', text, re.IGNORECASE | re.DOTALL)
        if result_match:
            result = result_match.group(1).strip().upper()
            # Exclude NO ERROR corrections
            if 'NO ERROR' in result or 'NO_ERROR' in result:
                return False
            # Include ERROR FOUND corrections
            if 'ERROR' in result:
                return True
        # If no <RESULT> tag, assume it's a valid error (for backward compatibility)
        return True
    
    def _classify_human_correction(
        self,
        human_corr: Dict,
        ai_corrections: List[Dict],
        student_code: str,
        template
    ) -> Dict:
        """Classify a single human correction as MATCH or MISSING."""
        
        human_text = human_corr.get("text", str(human_corr))
        
        # Format all AI corrections for the prompt
        ai_texts = "\n---\n".join([
            f"[AI Correction {i+1}]: {ai.get('text', str(ai))}"
            for i, ai in enumerate(ai_corrections)
        ]) if ai_corrections else "No AI corrections found."
        
        # Render template
        rendered = template.render(
            human_correction=human_text,
            ai_corrections=ai_texts,
            student_code=student_code
        )
        
        # Split system/human if separator exists
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = "Classify this correction as MATCH or MISSING"
            human_prompt = rendered
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        # Invoke LLM
        raw_response = self.llm.invoke(messages)
        
        # Extract text
        if hasattr(raw_response, 'content'):
            response_text = raw_response.content
        else:
            response_text = str(raw_response)
        
        # Parse JSON result
        result = self._parse_classification_result(response_text, ai_corrections)
        return result
    
    def _parse_classification_result(self, response_text: str, ai_corrections: List[Dict]) -> Dict:
        """Parse the classification result from LLM response (XML format)."""
        result = {
            "is_match": False,
            "matched_ai_index": None,
            "matched_ai_text": "",
            "match_quality": "FULL",
            "impact_severity": "MEDIUM",
            "explanation": ""
        }
        
        # Look for <RESULT> block with XML tags
        result_match = re.search(r'<RESULT>(.*?)</RESULT>', response_text, re.DOTALL | re.IGNORECASE)
        if result_match:
            result_content = result_match.group(1)
            
            # Parse XML tags
            class_match = re.search(r'<CLASSIFICATION>(.*?)</CLASSIFICATION>', result_content, re.IGNORECASE)
            if class_match:
                classification = class_match.group(1).strip().upper()
                result["is_match"] = classification == "MATCH"
            
            index_match = re.search(r'<MATCHED_AI_INDEX>(.*?)</MATCHED_AI_INDEX>', result_content, re.IGNORECASE)
            if index_match:
                try:
                    idx = int(index_match.group(1).strip())
                    if idx >= 0:  # -1 means no match
                        result["matched_ai_index"] = idx
                except ValueError:
                    pass
            
            quality_match = re.search(r'<MATCH_QUALITY>(.*?)</MATCH_QUALITY>', result_content, re.IGNORECASE)
            if quality_match:
                result["match_quality"] = quality_match.group(1).strip().upper()
            
            severity_match = re.search(r'<IMPACT_SEVERITY>(.*?)</IMPACT_SEVERITY>', result_content, re.IGNORECASE)
            if severity_match:
                result["impact_severity"] = severity_match.group(1).strip().upper()
            
            expl_match = re.search(r'<EXPLANATION>(.*?)</EXPLANATION>', result_content, re.IGNORECASE | re.DOTALL)
            if expl_match:
                result["explanation"] = expl_match.group(1).strip()
            
            # Get the matched AI correction (text and full object)
            if result["matched_ai_index"] is not None and ai_corrections:
                idx = result["matched_ai_index"]
                if 0 <= idx < len(ai_corrections):
                    matched_corr = ai_corrections[idx]
                    result["matched_ai_text"] = matched_corr.get("text", "")
                    result["matched_ai_corr"] = matched_corr  # Full object with function/requirement
        
        return result
    
    def _format_match_output(self, matches: List[Dict]) -> str:
        """Format matches into expected output format."""
        if not matches:
            return """NO MATCHED REQUIREMENTS FOUND

<AUXILIARY_METRIC_COUNT>
MATCH_COUNT: 0
</AUXILIARY_METRIC_COUNT>"""
        
        lines = []
        for i, m in enumerate(matches, 1):
            lines.append(f"""**{i}.**
**Requirement:** <requirement>{m['requirement']}</requirement>
**Function:** {m['function']}
**Type:** {m['type']}
**Human correction:** <human>{m['human']}</human>
**AI correction:** <generated>{m['ai']}</generated>
**Match quality:** {m.get('match_quality', 'FULL')}""")
        
        lines.append(f"""
<AUXILIARY_METRIC_COUNT>
MATCH_COUNT: {len(matches)}
</AUXILIARY_METRIC_COUNT>""")
        
        return "\n\n".join(lines)
    
    def _format_missing_output(self, missing: List[Dict]) -> str:
        """Format missing items into expected output format."""
        output_lines = ["### MISSING REQUIREMENTS (AI failed to identify these):"]
        
        if not missing:
            output_lines.append("\nNO MISSING REQUIREMENTS - AI IDENTIFIED ALL HUMAN ISSUES")
        else:
            for i, m in enumerate(missing, 1):
                output_lines.append(f"""
**{i}.**
**Requirement:** <requirement>{m['requirement']}</requirement>
**Function:** {m['function']}
**Type:** {m['type']}
**Human correction:** <human>{m['human']}</human>
**Impact severity:** {m.get('impact_severity', 'MEDIUM')}""")
        
        output_lines.append(f"""
<AUXILIARY_METRIC_COUNT>
MISSING_COUNT: {len(missing)}
</AUXILIARY_METRIC_COUNT>""")
        
        return "\n".join(output_lines)
    
    def _format_extra_output(self, extras: List[Dict]) -> str:
        """Format extra items into expected output format."""
        if not extras:
            return """NO EXTRA REQUIREMENTS - AI STAYED FOCUSED ON HUMAN ISSUES

<AUXILIARY_METRIC_COUNT>
EXTRA_COUNT: 0
</AUXILIARY_METRIC_COUNT>"""
        
        lines = []
        for i, e in enumerate(extras, 1):
            lines.append(f"""**{i}.**
**AI correction:** <generated>{e['ai']}</generated>
**Relevance:** {e.get('relevance', 'RELEVANT')}""")
        
        lines.append(f"""
<AUXILIARY_METRIC_COUNT>
EXTRA_COUNT: {len(extras)}
</AUXILIARY_METRIC_COUNT>""")
        
        return "\n\n".join(lines)
    
    def _format_empty_results(self) -> Dict[str, str]:
        """Return empty results when no corrections to classify."""
        return {
            "match": """NO MATCHED REQUIREMENTS FOUND

<AUXILIARY_METRIC_COUNT>
MATCH_COUNT: 0
</AUXILIARY_METRIC_COUNT>""",
            "missing": """### MISSING REQUIREMENTS (AI failed to identify these):

NO MISSING REQUIREMENTS - NO HUMAN CORRECTIONS PROVIDED

<AUXILIARY_METRIC_COUNT>
MISSING_COUNT: 0
</AUXILIARY_METRIC_COUNT>""",
            "extra": """NO EXTRA REQUIREMENTS - NO AI CORRECTIONS FOUND

<AUXILIARY_METRIC_COUNT>
EXTRA_COUNT: 0
</AUXILIARY_METRIC_COUNT>"""
        }


class DependentStrategy(AuxMetricsStrategy):
    """
    Dependent strategy: runs match first, then injects results into missing/extra.
    
    Process:
    1. Run match classification to identify matched corrections
    2. Run missing with matched corrections injected ("do not flag these as missing")
    3. Run extra with matched corrections injected ("do not flag these as extra")
    
    Benefits:
    - Prevents hallucination where matched items appear as missing or extra
    - More consistent overall classification
    
    Trade-offs:
    - Sequential execution (cannot parallelize)
    - Slightly slower than independent
    """
    
    def __init__(self, evaluator: 'AuxiliaryMetricsEvaluator'):
        super().__init__(evaluator)
        # Get template paths from config
        self.match_template_path = evaluator.aux_metric_configs.get("match", {}).get(
            "template", "evaluators/dependent/aux_match.jinja"
        )
        self.missing_template_path = evaluator.aux_metric_configs.get("missing", {}).get(
            "template", "evaluators/dependent/aux_missing_with_context.jinja"
        )
        self.extra_template_path = evaluator.aux_metric_configs.get("extra", {}).get(
            "template", "evaluators/dependent/aux_extra_with_context.jinja"
        )
    
    def compute(
        self,
        generated_text: str,
        reference_text: Union[str, Dict],
        student_code: str,
        assignment: str,
        requirements: List[Dict],
    ) -> Dict[str, str]:
        """Run match first, then inject results into missing/extra prompts."""
        
        # Format inputs
        requirements_xml = self._format_requirements_as_xml(requirements)
        human_correction_xml = self._format_reference_correction(reference_text)
        generated_correction_xml = self._format_generated_correction(generated_text)
        
        # Step 1: Run MATCH
        logger.info("DependentStrategy: Running MATCH classification")
        match_template = self.jinja_env.get_template(self.match_template_path)
        match_result = self._run_metric(
            match_template,
            assignment=assignment,
            requirements=requirements_xml,
            student_code=student_code,
            reference_correction=human_correction_xml,
            generated_correction=generated_correction_xml
        )
        
        # Extract matched corrections for injection
        matched_corrections_text = self._extract_matched_corrections(match_result)
        
        # Step 2: Run MISSING with matched corrections injected
        logger.info("DependentStrategy: Running MISSING classification with context")
        missing_template = self.jinja_env.get_template(self.missing_template_path)
        missing_result = self._run_metric(
            missing_template,
            assignment=assignment,
            requirements=requirements_xml,
            student_code=student_code,
            reference_correction=human_correction_xml,
            generated_correction=generated_correction_xml,
            matched_corrections=matched_corrections_text
        )
        
        # Step 3: Run EXTRA with matched corrections injected
        logger.info("DependentStrategy: Running EXTRA classification with context")
        extra_template = self.jinja_env.get_template(self.extra_template_path)
        extra_result = self._run_metric(
            extra_template,
            assignment=assignment,
            requirements=requirements_xml,
            student_code=student_code,
            reference_correction=human_correction_xml,
            generated_correction=generated_correction_xml,
            matched_corrections=matched_corrections_text
        )
        
        return {
            "match": match_result,
            "missing": missing_result,
            "extra": extra_result
        }
    
    def _run_metric(self, template, **context) -> str:
        """Run a single metric template and return the result."""
        rendered = template.render(**context)
        
        # Split system/human if separator exists
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = "Compute auxiliary metric"
            human_prompt = rendered
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        raw_response = self.llm.invoke(messages)
        
        # Extract text
        if hasattr(raw_response, 'content'):
            text = raw_response.content
        else:
            text = str(raw_response)
        
        # Check for <OUTPUT> tags and extract only that content
        output_match = re.search(
            r'<OUTPUT>(.*?)</OUTPUT>',
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if output_match:
            return output_match.group(1).strip()
        else:
            return text.strip()
    
    def _extract_matched_corrections(self, match_result: str) -> str:
        """Extract matched corrections from match result for injection."""
        # Build a summary of matched items
        lines = []
        
        # Parse numbered items from match result
        items = re.split(r'\*\*(\d+)\.\*\*', match_result)
        
        for i in range(1, len(items), 2):
            if i + 1 < len(items):
                item_content = items[i + 1]
                
                # Extract requirement
                req_match = re.search(r'<requirement>(.*?)</requirement>', item_content, re.DOTALL | re.IGNORECASE)
                if req_match:
                    lines.append(f"- {req_match.group(1).strip()}")
        
        if not lines:
            return "No matched corrections found."
        
        return "\n".join(lines)
    
    def _format_requirements_as_xml(self, requirements: List[Dict]) -> str:
        """Format requirements as XML tags."""
        if not requirements:
            return ""
        
        xml_parts = []
        for req in requirements:
            req_type = req['type'].value if hasattr(req['type'], 'value') else str(req['type'])
            xml_parts.append(
                f'<requirement function="{req["function"]}" type="{req_type}">\n'
                f'{req["requirement"]}\n'
                f'</requirement>'
            )
        
        return "\n".join(xml_parts)
    
    def _format_reference_correction(self, reference_correction: Union[str, Dict]) -> str:
        """Format reference correction with <human> tags."""
        if isinstance(reference_correction, str):
            if not reference_correction.strip():
                return ""
            return f"<human>\n{reference_correction}\n</human>"
        elif isinstance(reference_correction, dict) and 'corrections' in reference_correction:
            corrections = reference_correction['corrections']
            if not corrections:
                return ""
            wrapped = [f"<human>\n{c.strip()}\n</human>" for c in corrections if c.strip()]
            return "\n\n".join(wrapped)
        else:
            return f"<human>\n{str(reference_correction)}\n</human>"
    
    def _format_generated_correction(self, generated_text: str) -> str:
        """Format generated correction with <generated> tags."""
        if not generated_text.strip():
            return ""
        
        # Split by double newlines to separate individual corrections
        corrections = generated_text.strip().split("\n\n")
        wrapped = []
        for correction in corrections:
            if correction.strip():
                wrapped.append(f"<generated>\n{correction.strip()}\n</generated>")
        
        return "\n\n".join(wrapped)


# Strategy registry for easy lookup
STRATEGIES = {
    "independent": IndependentStrategy,
    "per_correction": PerCorrectionStrategy,
    "dependent": DependentStrategy,
}


def get_strategy(name: str, evaluator: 'AuxiliaryMetricsEvaluator') -> AuxMetricsStrategy:
    """Get a strategy instance by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](evaluator)
