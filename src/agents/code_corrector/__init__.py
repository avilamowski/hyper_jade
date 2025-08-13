"""
Code Correction Agent

This agent uses the generated correction prompts to evaluate student code
and provide detailed feedback and corrections.
"""

from .code_corrector import CodeCorrectorAgent

__all__ = ["CodeCorrectorAgent"]
