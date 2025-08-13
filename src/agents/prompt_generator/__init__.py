"""
Prompt Generator Agent

This agent generates correction prompts for each rubric item and the assignment,
which will be used by the code correction agent.
"""

from .prompt_generator import PromptGeneratorAgent

__all__ = ["PromptGeneratorAgent"]
