"""
Metric Computers Module

This module provides Python functions to compute evaluation metrics from auxiliary metrics.
These functions can be used as alternatives to LLM-based template evaluation.

The design allows for interchangeable use of:
- LLM-based templates
- Python-based computations

All functions in this module follow a consistent interface:
- Input: auxiliary_metrics dict (with keys like 'match', 'missing', 'extra')
- Output: tuple of (score: int, explanation: str)
"""

import re
import logging
from typing import Dict, Any, Tuple

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


def compute_completeness(auxiliary_metrics: Dict[str, str]) -> Tuple[int, str]:
    """
    Compute the completeness score based on auxiliary metrics.
    
    Completeness measures how completely the AI identified issues from the human reference.
    It's calculated as: 1 - (MISSING / (MATCH + MISSING))
    
    Scoring guide:
    - 5: Missing ratio = 0 (AI found all human issues, 0% missed)
    - 4: Missing ratio ≤ 0.15 (AI missed ≤15% of issues, 1 minor issue)
    - 3: Missing ratio ≤ 0.35 (AI missed ≤35% of issues, 2-3 issues)
    - 2: Missing ratio ≤ 0.55 (AI missed ≤55% of issues, 4-5 issues)
    - 1: Missing ratio > 0.55 (AI missed >55% of issues)
    - 0: Missing ratio ≥ 0.80 or critical functional requirements missed
    
    Args:
        auxiliary_metrics: Dictionary with keys 'match' and 'missing' containing
                          the text outputs from auxiliary metric templates
    
    Returns:
        Tuple of (score: int, explanation: str)
    """
    match_text = auxiliary_metrics.get('match', '')
    missing_text = auxiliary_metrics.get('missing', '')
    
    match_count = parse_auxiliary_metric_count(match_text, metric_type='match')
    missing_count = parse_auxiliary_metric_count(missing_text, metric_type='missing')
    
    total_human_issues = match_count + missing_count
    
    # Handle edge cases
    if total_human_issues == 0:
        # No human issues identified - perfect completeness (nothing to miss)
        return 5, "No human-identified issues found. Completeness is perfect (nothing to miss)."
    
    if missing_count == 0:
        # AI found all human issues
        return 5, f"AI identified all {total_human_issues} human-identified issues (MATCH={match_count}, MISSING={missing_count}). Missing ratio: 0.0 (0%)."
    
    # Calculate missing ratio
    missing_ratio = missing_count / total_human_issues
    
    # Determine score based on ratio
    if missing_ratio >= 0.80:
        score = 0
    elif missing_ratio > 0.55:
        score = 1
    elif missing_ratio > 0.35:
        score = 2
    elif missing_ratio > 0.15:
        score = 3
    else:
        score = 4
    
    explanation = (
        f"AI identified {match_count} out of {total_human_issues} human-identified issues "
        f"(MATCH={match_count}, MISSING={missing_count}, Total={total_human_issues}). "
        f"Missing ratio: {missing_count}/{total_human_issues} = {missing_ratio:.3f} ({missing_ratio*100:.1f}%)."
    )
    
    return score, explanation

