"""
AI Agents for assignment evaluation.

This module contains specialized agents for the assignment evaluation pipeline:
- Requirement Generator Agent
- Prompt Generator Agent  
- Code Correction Agent
- Agent Evaluator (LLM as a Judge)
"""

from .requirement_generator import RequirementGeneratorAgent
from .prompt_generator import PromptGeneratorAgent
from .code_corrector import CodeCorrectorAgent
from .utils.agent_evaluator import AgentEvaluator

__all__ = ["RequirementGeneratorAgent", "PromptGeneratorAgent", "CodeCorrectorAgent", "AgentEvaluator"]
