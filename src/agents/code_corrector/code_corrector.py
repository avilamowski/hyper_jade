#!/usr/bin/env python3
"""
Code Correction Agent

This agent takes a Jinja2 template prompt and a Python code file, then analyzes
how well the code satisfies the requirement specified in the prompt.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from jinja2 import Template

logger = logging.getLogger(__name__)

class CodeCorrectorAgent:
    """Agent that analyzes code against a specific requirement using Jinja2 templates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on configuration"""
        if self.config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.config.get("model_name", "gpt-4"),
                temperature=self.config.get("temperature", 0.1)
            )
        else:
            return OllamaLLM(
                model=self.config.get("model_name", "qwen2.5:7b"),
                temperature=self.config.get("temperature", 0.1)
            )
    
    def analyze_code(
        self,
        prompt_template_path: str,
        code_file_path: str,
        output_file_path: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Analyze code against a requirement using a Jinja2 template prompt
        
        Args:
            prompt_template_path: Path to the Jinja2 template prompt (.jinja file)
            code_file_path: Path to the Python code file (.py or .txt)
            output_file_path: Optional path to save the analysis result
            additional_context: Optional additional context for the analysis
            
        Returns:
            The analysis result as a string
        """
        logger.info(f"Analyzing code: {code_file_path} against prompt: {prompt_template_path}")
        
        # Read the Jinja2 template
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            template_content = f.read().strip()
        
        # Read the code file
        with open(code_file_path, 'r', encoding='utf-8') as f:
            student_code = f.read().strip()
        
        # Render the template with the code
        template = Template(template_content)
        rendered_prompt = template.render(
            code=student_code
        )
        
        # Generate analysis using LLM
        analysis = self._generate_analysis(rendered_prompt, student_code)
        
        # Save analysis if output path is provided
        if output_file_path:
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
            
            logger.info(f"Analysis saved to: {output_path}")
        
        return analysis
    
    def _generate_analysis(self, rendered_prompt: str, student_code: str) -> str:
        """Generate analysis using the rendered prompt and LLM"""
        
        response = self.llm.invoke([HumanMessage(content=rendered_prompt)])
        analysis = str(response).strip()
        
        # Clean up the response if needed
        if "```" in analysis:
            # Remove markdown code blocks if present
            lines = analysis.split('\n')
            cleaned_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    cleaned_lines.append(line)
            
            analysis = '\n'.join(cleaned_lines).strip()
        
        logger.info("Analysis generated successfully")
        return analysis
    
    def batch_analyze(
        self,
        prompt_template_path: str,
        code_directory: str,
        output_directory: str,
        additional_context: Optional[str] = None
    ) -> List[str]:
        """
        Analyze multiple code files against the same requirement
        
        Args:
            prompt_template_path: Path to the Jinja2 template prompt
            code_directory: Directory containing code files to analyze
            output_directory: Directory to save analysis results
            additional_context: Optional additional context for all analyses
            
        Returns:
            List of paths to the generated analysis files
        """
        logger.info(f"Batch analyzing code files in: {code_directory}")
        
        code_path = Path(code_directory)
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all Python and text files
        code_files = list(code_path.glob("*.py")) + list(code_path.glob("*.txt"))
        
        analysis_files = []
        for code_file in code_files:
            # Generate output filename
            output_filename = f"analysis_{code_file.stem}.txt"
            output_file_path = output_path / output_filename
            
            # Analyze the code
            analysis = self.analyze_code(
                prompt_template_path=str(prompt_template_path),
                code_file_path=str(code_file),
                output_file_path=str(output_file_path),
                additional_context=additional_context
            )
            
            analysis_files.append(str(output_file_path))
            logger.info(f"Analyzed: {code_file.name} -> {output_filename}")
        
        logger.info(f"Batch analysis completed. Processed {len(analysis_files)} files")
        return analysis_files
