"""
Code Correction Agent

This agent uses the generated correction prompts to evaluate student code
and provide detailed feedback and corrections.
"""

from .code_corrector import CodeCorrectorAgent
from .linter_correction_agent import LinterCorrectionAgent

__all__ = ["CodeCorrectorAgent", "LinterCorrectionAgent"]
