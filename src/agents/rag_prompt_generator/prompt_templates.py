"""
Module for managing language-specific prompt templates using Jinja2.
"""
import os
from typing import Dict, Any
from enum import Enum
from jinja2 import Environment, FileSystemLoader, Template


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    HASKELL = "haskell"


class PromptTemplates:
    """Manages language-specific prompt templates for code generation using Jinja2."""
    
    def __init__(self):
        # Get the directory where this module is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 3 levels to getting to the project root (src/agents/rag_prompt_generator -> src/agents -> src -> hyper_jade)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # Point to the root templates directory
        templates_dir = os.path.join(project_root, "templates")
        
        # Initialize Jinja2 environment
        # NOTE: lstrip_blocks is NOT used to preserve code indentation in templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=False  # Keep False to preserve code indentation
        )
        
        # Cache for loaded templates
        self._template_cache = {}
    
    def _get_template(self, language: Language, template_name: str) -> Template:
        """Get a Jinja2 template. Tries:
        1. rag/{language}/{template_name}.j2 (RAG specific)
        2. {template_name}.jinja (Shared standard)
        """
        cache_key = f"{language.value}_{template_name}"
        
        if cache_key not in self._template_cache:
            # Try RAG specific path first
            rag_path = f"rag/{language.value}/{template_name}.j2"
            # Try shared path second
            shared_path = f"{template_name}.jinja"
            
            try:
                # Try loading RAG template first
                try:
                    template = self.jinja_env.get_template(rag_path)
                except Exception:
                    # Fallback to shared template
                    template = self.jinja_env.get_template(shared_path)
                    
                self._template_cache[cache_key] = template
            except Exception as e:
                raise ValueError(f"Template '{template_name}' not found at {rag_path} or {shared_path}: {e}")
        
        return self._template_cache[cache_key]
    
    def get_template(self, language: Language, template_name: str) -> str:
        """Get a specific template for a given language (for backward compatibility)."""
        # This method is kept for backward compatibility but is not recommended
        # Use format_template instead for better functionality
        return self.format_template(language, template_name)
    
    def format_template(self, language: Language, template_name: str, **kwargs) -> str:
        """Get and format a template with the provided arguments."""
        template = self._get_template(language, template_name)
        return template.render(**kwargs)
    
    def get_system_message(self, language: Language, template_name: str) -> str:
        """Get the system message for a specific template and language."""
        system_template_name = f"{template_name}_system"
        return self.format_template(language, system_template_name)
