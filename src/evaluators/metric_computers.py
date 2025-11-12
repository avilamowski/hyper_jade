"""
Metric Computers Module

This module provides Python functions to compute evaluation metrics from auxiliary metrics.
These functions can be used as alternatives to LLM-based template evaluation.

The design allows for interchangeable use of:
- LLM-based templates
- Python-based computations

All functions in this module follow a consistent interface:
- Input: auxiliary_metrics dict (with keys like 'match', 'missing', 'extra')
- Output: tuple of (score: float, explanation: str) where score is between 0.0 and 1.0
"""

import re
import logging
import os
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


def parse_auxiliary_metric_count(text: str, metric_type: str) -> int:
    """
    Parse an auxiliary metric text to extract the count from structured format.
    
    Args:
        text: The raw text output from an auxiliary metric template
        metric_type: Which count to look for ('match', 'missing', 'extra')
        
    Returns:
        The count of items found, or 0 if not found
    """
    if not text:
        return 0
    
    count_patterns = {
        'match': r'MATCH_COUNT:\s*(\d+)',
        'missing': r'MISSING_COUNT:\s*(\d+)',
        'extra': r'EXTRA_COUNT:\s*(\d+)',
    }
    
    pattern = count_patterns.get(metric_type.lower())
    if not pattern:
        return 0
    
    count_block_match = re.search(
        r'<AUXILIARY_METRIC_COUNT>(.*?)</AUXILIARY_METRIC_COUNT>',
        text,
        re.DOTALL | re.IGNORECASE
    )
    
    if count_block_match:
        count_block = count_block_match.group(1)
        match = re.search(pattern, count_block, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0


def compute_completeness(auxiliary_metrics: Dict[str, str]) -> Tuple[float, str]:
    """
    Compute the completeness score based on auxiliary metrics.
    
    Completeness measures how completely the AI identified issues from the human reference.
    Score is calculated as: 1 - (MISSING / (MATCH + MISSING))
    
    Args:
        auxiliary_metrics: Dictionary with keys 'match' and 'missing' containing
                          the text outputs from auxiliary metric templates
    
    Returns:
        Tuple of (score: float, explanation: str) where score is between 0.0 and 1.0
    """
    match_text = auxiliary_metrics.get('match', '')
    missing_text = auxiliary_metrics.get('missing', '')
    
    match_count = parse_auxiliary_metric_count(match_text, metric_type='match')
    missing_count = parse_auxiliary_metric_count(missing_text, metric_type='missing')
    
    total_human_issues = match_count + missing_count
    
    # Handle edge cases
    if total_human_issues == 0:
        # No human issues identified - perfect completeness (nothing to miss)
        return 1.0, "No human-identified issues found. Completeness is perfect (nothing to miss)."
    
    if missing_count == 0:
        # AI found all human issues
        return 1.0, f"AI identified all {total_human_issues} human-identified issues (MATCH={match_count}, MISSING={missing_count}). Missing ratio: 0.0 (0%)."
    
    # Calculate completeness score: 1 - missing_ratio
    missing_ratio = missing_count / total_human_issues
    score = 1.0 - missing_ratio
    
    explanation = (
        f"AI identified {match_count} out of {total_human_issues} human-identified issues "
        f"(MATCH={match_count}, MISSING={missing_count}, Total={total_human_issues}). "
        f"Missing ratio: {missing_count}/{total_human_issues} = {missing_ratio:.3f} ({missing_ratio*100:.1f}%)."
    )
    
    return score, explanation


def compute_restraint(auxiliary_metrics: Dict[str, str]) -> Tuple[float, str]:
    """
    Compute the restraint score based on auxiliary metrics.
    
    Restraint measures how restrained the AI is in reporting extra issues.
    Score is calculated as: 1 - (extra / (match + extra))
    
    Lower extra ratio means better restraint (more restrained).
    
    Args:
        auxiliary_metrics: Dictionary with keys 'match' and 'extra' containing
                          the text outputs from auxiliary metric templates
    
    Returns:
        Tuple of (score: float, explanation: str) where score is between 0.0 and 1.0
    """
    match_text = auxiliary_metrics.get('match', '')
    extra_text = auxiliary_metrics.get('extra', '')
    
    match_count = parse_auxiliary_metric_count(match_text, metric_type='match')
    extra_count = parse_auxiliary_metric_count(extra_text, metric_type='extra')
    
    total_ai_issues = match_count + extra_count
    
    # Handle edge cases
    if total_ai_issues == 0:
        # No AI issues identified - perfect restraint (nothing extra)
        return 1.0, "No AI-identified issues found. Restraint is perfect (no extra issues)."
    
    if extra_count == 0:
        # No extra issues
        return 1.0, f"AI identified {match_count} issues, all matched with human reference (MATCH={match_count}, EXTRA={extra_count}). Extra ratio: 0.0 (0%)."
    
    # Calculate restraint score: 1 - extra_ratio
    extra_ratio = extra_count / total_ai_issues
    score = 1.0 - extra_ratio
    
    explanation = (
        f"AI identified {total_ai_issues} issues total (MATCH={match_count}, EXTRA={extra_count}). "
        f"Extra ratio: {extra_count}/{total_ai_issues} = {extra_ratio:.3f} ({extra_ratio*100:.1f}%)."
    )
    
    return score, explanation


def compute_content_similarity(llm_response: str) -> Tuple[float, str]:
    """
    Compute the content similarity score by parsing individual item scores from LLM response
    and calculating the mean in Python (not by the LLM).
    
    This is a hybrid approach:
    - LLM evaluates each matched item individually and assigns scores (0.0-1.0)
    - Python parses those scores and computes the average
    
    Expected XML format from LLM:
    <RESULT>
    <ITEM>
    <ID>1</ID>
    <REQUIREMENT>label</REQUIREMENT>
    <SCORE>0.85</SCORE>
    <JUSTIFICATION>justification</JUSTIFICATION>
    </ITEM>
    <ITEM>
    <ID>2</ID>
    <REQUIREMENT>label</REQUIREMENT>
    <SCORE>0.60</SCORE>
    <JUSTIFICATION>justification</JUSTIFICATION>
    </ITEM>
    </RESULT>
    
    Args:
        llm_response: The raw text output from the content_similarity template,
                     which should contain individual item scores in XML format
    
    Returns:
        Tuple of (score: float, explanation: str) where score is the mean of item scores
    """
    from xml.etree import ElementTree as ET
    
    if not llm_response:
        return 0.0, "No LLM response provided for content similarity evaluation."
    
    # Try to parse XML format first
    try:
        # Extract RESULT block
        result_match = re.search(r'<RESULT>(.*?)</RESULT>', llm_response, re.DOTALL | re.IGNORECASE)
        if result_match:
            result_content = result_match.group(1).strip()
            
            # Check if RESULT says no matches INSIDE the result block
            if "NO MATCHED REQUIREMENTS FOUND" in result_content.upper():
                return 0.0, "No matched requirements found. Content similarity cannot be calculated."
            
            result_xml = f"<RESULT>{result_match.group(1)}</RESULT>"
            root = ET.fromstring(result_xml)
            
            # Extract all ITEM scores
            items = root.findall('.//ITEM')
            if items:
                item_scores = []
                for item in items:
                    score_elem = item.find('SCORE')
                    if score_elem is not None and score_elem.text:
                        score_val = float(score_elem.text)
                        item_scores.append(score_val)
                
                if item_scores:
                    mean_score = sum(item_scores) / len(item_scores)
                    
                    # Build explanation with Python-computed mean
                    explanation = (
                        f"Evaluated {len(item_scores)} matched items. "
                        f"Individual scores: {', '.join(f'{s:.2f}' for s in item_scores)}. "
                        f"Mean (computed in Python): {mean_score:.3f}."
                    )
                    
                    return mean_score, explanation
    except Exception as e:
        logger.warning(f"XML parsing failed for content_similarity: {e}")
    
    # Fallback: try old text format "Item N (labfel): score -"
    item_pattern = r'Item \d+ \([^)]+\):\s*(\d+\.?\d*)\s*-'
    matches = re.findall(item_pattern, llm_response)
    
    if matches:
        item_scores = [float(score) for score in matches]
        mean_score = sum(item_scores) / len(item_scores)
        
        explanation = (
            f"Evaluated {len(item_scores)} matched items. "
            f"Individual scores: {', '.join(f'{s:.2f}' for s in item_scores)}. "
            f"Mean (computed in Python): {mean_score:.3f}."
        )
        
        return mean_score, explanation
    
    # No valid format found
    return 0.0, "Could not parse item scores from LLM response."

def parse_extra_corrections(extra_text: str) -> List[str]:
    """
    Parse the extra auxiliary metric text to extract individual AI corrections.
    
    Args:
        extra_text: The raw text output from aux_extra template
        
    Returns:
        List of AI correction strings
    """
    if not extra_text:
        return []
    
    corrections = []
    seen = set()
    
    # Pattern to match numbered items: **N.** ... **AI correction:** "text" or text
    # Split by numbered items first
    items = re.split(r'\*\*(\d+)\.\*\*', extra_text)
    
    for i in range(1, len(items), 2):  # Skip number, get content
        if i + 1 < len(items):
            item_content = items[i + 1]
            
            # Look for AI correction field
            ai_corr_match = re.search(r'\*\*AI correction:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
            if ai_corr_match:
                correction_text = ai_corr_match.group(1).strip()
                
                # Remove quotes if present
                if correction_text.startswith('"') and correction_text.endswith('"'):
                    correction_text = correction_text[1:-1]
                
                # Clean up
                correction_text = correction_text.strip()
                
                if correction_text and correction_text not in seen:
                    seen.add(correction_text)
                    corrections.append(correction_text)
    
    return corrections


def compute_precision(
    auxiliary_metrics: Dict[str, str],
    student_code: str,
    assignment: str = "",
    requirements: str = "",
    llm: Optional[Any] = None,
    precision_llm_config: Optional[Dict[str, Any]] = None
) -> Tuple[float, str]:
    """
    Compute the precision score based on auxiliary metrics.
    
    Precision measures how accurate the AI's extra corrections are.
    It evaluates each "extra" correction individually to determine if it's correct.
    Score is calculated as: correct_extras / total_extras
    
    Args:
        auxiliary_metrics: Dictionary with key 'extra' containing the text output
        student_code: The student's code to evaluate corrections against
        assignment: Optional assignment description
        requirements: Optional requirements/rubric
        llm: Optional LLM instance to use (if None, will create one from config)
        precision_llm_config: Optional config dict with 'model_name', 'provider', 'temperature'
                            for the precision evaluation LLM (uses more powerful model)
    
    Returns:
        Tuple of (score: float, explanation: str) where score is between 0.0 and 1.0
    """
    extra_text = auxiliary_metrics.get('extra', '')
    
    # Parse individual corrections from extra text
    corrections = parse_extra_corrections(extra_text)
    
    if not corrections:
        return 1.0, "No extra corrections found. Precision is perfect (no false positives)."
    
    # Setup LLM for precision evaluation (use more powerful model)
    if llm is None:
        if precision_llm_config is None:
            # Default to gpt-4o for precision (more powerful)
            precision_llm_config = {
                'model_name': 'gpt-4o',
                'provider': 'openai',
                'temperature': 0.1
            }
        
        provider = precision_llm_config.get('provider', 'openai')
        model_name = precision_llm_config.get('model_name', 'gpt-4o')
        temperature = precision_llm_config.get('temperature', 0.1)
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        elif provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            llm = OllamaLLM(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Setup template for evaluating each correction
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "templates"
    
    jinja_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["jinja", "html", "txt"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    template = jinja_env.get_template("evaluators/individual/eval_precision_item.jinja")
    
    # Evaluate each correction
    correct_count = 0
    total_count = len(corrections)
    
    for correction in corrections:
        rendered = template.render(
            assignment=assignment,
            requirements=requirements,
            student_code=student_code,
            ai_correction=correction
        )
        
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = "Evaluate precision of AI correction"
            human_prompt = rendered
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Extract text
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse result
        is_correct_match = re.search(r'<IS_CORRECT>(.*?)</IS_CORRECT>', response_text, re.IGNORECASE | re.DOTALL)
        if is_correct_match:
            is_correct = is_correct_match.group(1).strip().upper()
            if is_correct == "CORRECT":
                correct_count += 1
    
    # Calculate precision score
    if total_count == 0:
        score = 1.0
    else:
        score = correct_count / total_count
    
    explanation = (
        f"Evaluated {total_count} extra corrections. "
        f"{correct_count} were correct, {total_count - correct_count} were incorrect. "
        f"Precision: {correct_count}/{total_count} = {score:.3f} ({score*100:.1f}%)."
    )
    
    return score, explanation

