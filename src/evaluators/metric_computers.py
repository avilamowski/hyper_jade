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
from typing import Dict, Any, Tuple, List, Optional, Union
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


def compute_content_similarity(
    auxiliary_metrics: Dict[str, str],
    student_code: str,
    assignment: str = "",
    requirements: Union[List[Dict[str, Any]], str] = "",
    llm: Optional[Any] = None,
    content_similarity_llm_config: Optional[Dict[str, Any]] = None
) -> Tuple[float, str]:
    """
    Compute the content similarity score by making individual LLM calls for each matched item.
    
    This function:
    1. Parses matched items from the aux_match output
    2. Makes a separate LLM call for each matched item
    3. Calculates the mean score in Python
    
    Args:
        auxiliary_metrics: Dictionary with key 'match' containing the text output from aux_match template
        student_code: The student's code to evaluate corrections against
        assignment: Optional assignment description
        requirements: Optional requirements/rubric as List[Requirement] or string
        llm: Optional LLM instance to use (if None, will create one from config)
        content_similarity_llm_config: Optional config dict with 'model_name', 'provider', 'temperature'
    
    Returns:
        Tuple of (score: float, explanation: str) where score is the mean of individual item scores
    """
    from xml.etree import ElementTree as ET
    
    match_text = auxiliary_metrics.get('match', '')
    
    # Parse matched items
    matched_items = parse_matched_items(match_text)
    
    if not matched_items:
        return 0.0, "No matched requirements found. Content similarity cannot be calculated."
    
    # Setup LLM for content similarity evaluation
    if llm is None:
        if content_similarity_llm_config is None:
            # Default to llama3.2:latest for content_similarity
            content_similarity_llm_config = {
                'model_name': 'llama3.2:latest',
                'provider': 'ollama',
                'temperature': 0.1
            }
        
        provider = content_similarity_llm_config.get('provider', 'ollama')
        model_name = content_similarity_llm_config.get('model_name', 'llama3.2:latest')
        temperature = content_similarity_llm_config.get('temperature', 0.1)
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
            }
            if api_key:
                llm_kwargs["api_key"] = api_key
            if base_url:
                llm_kwargs["base_url"] = base_url
            llm = ChatOpenAI(**llm_kwargs)
        elif provider == "ollama":
            from langchain_ollama.llms import OllamaLLM
            llm = OllamaLLM(model=model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Setup template for evaluating each match
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "templates"
    
    jinja_env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["jinja", "html", "txt"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    template = jinja_env.get_template("evaluators/individual/eval_content_similarity_item.jinja")
    
    # Evaluate each matched item individually
    item_scores = []
    item_explanations = []
    
    for idx, match_item in enumerate(matched_items, 1):
        requirement_text = match_item.get('requirement', '')
        function = match_item.get('function', 'unknown')
        req_type = match_item.get('type', 'unknown')
        human_correction = match_item.get('human_correction', '')
        ai_correction = match_item.get('ai_correction', '')
        
        if not requirement_text:
            logger.warning(f"Skipping match {idx}: missing requirement text")
            continue
        
        # Render template for this individual match
        rendered = template.render(
            assignment=assignment,
            requirement=requirement_text,
            function=function,
            type=req_type,
            student_code=student_code,
            human_correction=human_correction,
            ai_correction=ai_correction
        )
        
        # Split system/human if separator exists
        parts = rendered.split("---HUMAN---", 1)
        if len(parts) == 2:
            system_prompt = parts[0].strip()
            human_prompt = parts[1].strip()
        else:
            system_prompt = "Evaluate content similarity for matched requirement"
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
        try:
            result_match = re.search(r'<RESULT>(.*?)</RESULT>', response_text, re.DOTALL | re.IGNORECASE)
            if result_match:
                result_xml = f"<RESULT>{result_match.group(1)}</RESULT>"
                root = ET.fromstring(result_xml)
                
                score_elem = root.find('SCORE')
                justif_elem = root.find('JUSTIFICATION')
                
                if score_elem is not None and score_elem.text:
                    # Score comes as integer 1-5, store it as-is for now
                    score_val = int(float(score_elem.text.strip()))
                    if score_val < 1 or score_val > 5:
                        logger.warning(f"Score {score_val} out of range [1-5] for match {idx}, clamping")
                        score_val = max(1, min(5, score_val))
                    
                    item_scores.append(score_val)
                    
                    justification = justif_elem.text.strip() if justif_elem is not None and justif_elem.text else ""
                    item_explanations.append(f"Item {idx} (score: {score_val}/5): {justification}")
                else:
                    logger.warning(f"Could not parse score for match {idx}")
            else:
                logger.warning(f"Could not find RESULT block for match {idx}")
        except Exception as e:
            logger.warning(f"Failed to parse response for match {idx}: {e}")
    
    # Calculate mean score
    if not item_scores:
        return 0.0, "Could not evaluate any matched items. Content similarity cannot be calculated."
    
    # Scores are in range 1-5, normalize to 0-1
    # Formula: (score - 1) / 4 maps [1,5] to [0,1]
    normalized_scores = [(s - 1) / 4.0 for s in item_scores]
    mean_score = sum(normalized_scores) / len(normalized_scores)
    
    # Build explanation with both raw scores (1-5) and normalized mean
    explanation = (
        f"Evaluated {len(item_scores)} matched items individually. "
        f"Individual scores (1-5 scale): {', '.join(f'{s}/5' for s in item_scores)}. "
        f"Normalized mean (0-1 scale): {mean_score:.3f}."
    )
    
    return mean_score, explanation

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


def parse_matched_items(match_text: str) -> List[Dict[str, str]]:
    """
    Parse the match auxiliary metric text to extract individual matched items.
    
    Each match contains:
    - requirement: The requirement text (from XML tag)
    - function: The function attribute
    - type: The type attribute
    - human_correction: What the human teacher said
    - ai_correction: What the AI said
    - match_quality: FULL/HIGH/PARTIAL/POOR
    
    Args:
        match_text: The raw text output from aux_match template
        
    Returns:
        List of dictionaries, each containing the match information
    """
    if not match_text:
        return []
    
    # Check for "NO MATCHED REQUIREMENTS FOUND"
    if "NO MATCHED REQUIREMENTS FOUND" in match_text.upper():
        return []
    
    matches = []
    
    # Split by numbered items: **N.**
    items = re.split(r'\*\*(\d+)\.\*\*', match_text)
    
    for i in range(1, len(items), 2):  # Skip number, get content
        if i + 1 < len(items):
            item_content = items[i + 1]
            
            match_dict = {}
            
            # Extract requirement (from <requirement> tag)
            req_match = re.search(r'\*\*Requirement:\*\*\s*<requirement>(.*?)</requirement>', item_content, re.DOTALL | re.IGNORECASE)
            if req_match:
                match_dict['requirement'] = req_match.group(1).strip()
            else:
                # Fallback: try without tags
                req_match = re.search(r'\*\*Requirement:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
                if req_match:
                    match_dict['requirement'] = req_match.group(1).strip()
            
            # Extract function
            func_match = re.search(r'\*\*Function:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
            if func_match:
                match_dict['function'] = func_match.group(1).strip()
            
            # Extract type
            type_match = re.search(r'\*\*Type:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
            if type_match:
                match_dict['type'] = type_match.group(1).strip()
            
            # Extract human correction (from <human> tag)
            human_match = re.search(r'\*\*Human correction:\*\*\s*<human>(.*?)</human>', item_content, re.DOTALL | re.IGNORECASE)
            if human_match:
                match_dict['human_correction'] = human_match.group(1).strip()
            else:
                # Fallback: try without tags
                human_match = re.search(r'\*\*Human correction:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
                if human_match:
                    match_dict['human_correction'] = human_match.group(1).strip()
            
            # Extract AI correction (from <generated> tag)
            ai_match = re.search(r'\*\*AI correction:\*\*\s*<generated>(.*?)</generated>', item_content, re.DOTALL | re.IGNORECASE)
            if ai_match:
                match_dict['ai_correction'] = ai_match.group(1).strip()
            else:
                # Fallback: try without tags
                ai_match = re.search(r'\*\*AI correction:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
                if ai_match:
                    match_dict['ai_correction'] = ai_match.group(1).strip()
            
            # Extract match quality
            quality_match = re.search(r'\*\*Match quality:\*\*\s*(.+?)(?:\*\*|$)', item_content, re.DOTALL)
            if quality_match:
                match_dict['match_quality'] = quality_match.group(1).strip()
            
            # Only add if we have at least requirement and corrections
            if match_dict.get('requirement') and (match_dict.get('human_correction') or match_dict.get('ai_correction')):
                matches.append(match_dict)
    
    return matches


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
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
            }
            if api_key:
                llm_kwargs["api_key"] = api_key
            if base_url:
                llm_kwargs["base_url"] = base_url
            llm = ChatOpenAI(**llm_kwargs)
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

