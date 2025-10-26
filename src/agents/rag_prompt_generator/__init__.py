"""
RAG Prompt Generator Module

This module provides RAG-enhanced prompt generation functionality that can be used
as an alternative to the standard prompt generator when USE_RAG is enabled.

Main Components:
- RAGSystem: Vector database and retrieval system
- RAGPromptGeneratorAgent: RAG-enhanced prompt generator agent
- CodeExampleGenerator: Generates code examples using course theory
- Config: Independent configuration for RAG functionality
"""

from .rag_prompt_generator import RAGPromptGeneratorAgent
from .rag_system import RAGSystem
from .code_generator import CodeExampleGenerator

__all__ = [
    'RAGPromptGeneratorAgent',
    'RAGSystem', 
    'CodeExampleGenerator'
]
