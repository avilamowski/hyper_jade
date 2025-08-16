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
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from jinja2 import Template

# Import MLflow logger lazily to avoid circular imports
def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed"""
    try:
        from src.core.mlflow_utils import mlflow_logger
        return mlflow_logger
    except ImportError:
        # Return a dummy logger if MLflow is not available
        class DummyLogger:
            def start_run(self, *args, **kwargs): return None
            def end_run(self): pass
            def log_metric(self, *args, **kwargs): pass
            def log_metrics(self, *args, **kwargs): pass
            def log_artifact(self, *args, **kwargs): pass
            def log_artifacts(self, *args, **kwargs): pass
            def log_param(self, *args, **kwargs): pass
            def log_params(self, *args, **kwargs): pass
            def log_text(self, *args, **kwargs): pass
            def log_code_analysis_metrics(self, *args, **kwargs): pass
            def log_prompt(self, *args, **kwargs): pass
            def log_trace_step(self, *args, **kwargs): pass
            def log_agent_input_output(self, *args, **kwargs): pass
        return DummyLogger()

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
        # Start MLflow run for code analysis
        mlflow_logger = get_mlflow_logger()
        run_id = mlflow_logger.start_run(
            run_name="code_analysis",
            tags={
                "agent": "code_corrector",
                "prompt_template": Path(prompt_template_path).name,
                "code_file": Path(code_file_path).name,
                "output_file": Path(output_file_path).name if output_file_path else "none"
            }
        )
        
        start_time = time.time()
        logger.info(f"Analyzing code: {code_file_path} against prompt: {prompt_template_path}")
        
        # Log parameters
        mlflow_logger.log_params({
            "prompt_template_path": prompt_template_path,
            "code_file_path": code_file_path,
            "output_file_path": output_file_path,
            "additional_context": additional_context is not None,
            "model_name": self.config.get("model_name", "unknown"),
            "provider": self.config.get("provider", "unknown"),
            "temperature": self.config.get("temperature", 0.1)
        })
        
        try:
            # Read the Jinja2 template
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
            
            # Read the code file
            with open(code_file_path, 'r', encoding='utf-8') as f:
                student_code = f.read().strip()
            
            # Log input files as artifacts
            mlflow_logger.log_text(template_content, "input_prompt_template.jinja")
            mlflow_logger.log_text(student_code, "input_student_code.py")
            
            # Log trace step: Reading inputs
            mlflow_logger.log_trace_step("read_inputs", {
                "prompt_template_path": prompt_template_path,
                "code_file_path": code_file_path,
                "template_length": len(template_content),
                "code_length": len(student_code)
            }, step_number=1)
            
            if additional_context:
                mlflow_logger.log_text(additional_context, "input_additional_context.txt")
            
            # Render the template with the code
            template_render_start = time.time()
            template = Template(template_content)
            rendered_prompt = template.render(
                code=student_code
            )
            template_render_time = time.time() - template_render_start
            
            # Log template rendering metrics
            mlflow_logger.log_metrics({
                "template_render_time_seconds": template_render_time,
                "rendered_prompt_length_chars": len(rendered_prompt),
                "rendered_prompt_length_lines": len(rendered_prompt.split('\n'))
            })
            
            # Log rendered prompt as artifact
            mlflow_logger.log_text(rendered_prompt, "rendered_prompt.txt")
            
            # Log trace step: Template rendering
            mlflow_logger.log_trace_step("template_rendering", {
                "render_time": template_render_time,
                "rendered_prompt_length": len(rendered_prompt)
            }, step_number=2)
            
            # Generate analysis using LLM
            llm_start_time = time.time()
            analysis = self._generate_analysis(rendered_prompt, student_code)
            llm_time = time.time() - llm_start_time
            
            # Log LLM performance metrics
            mlflow_logger.log_metrics({
                "llm_analysis_time_seconds": llm_time,
                "analysis_length_chars": len(analysis),
                "analysis_length_lines": len(analysis.split('\n'))
            })
            
            # Log analysis as artifact and metrics
            mlflow_logger.log_text(analysis, "generated_analysis.txt")
            
            # Log specific code analysis metrics
            requirement_name = Path(prompt_template_path).stem
            mlflow_logger.log_code_analysis_metrics(analysis, requirement_name)
            
            # Log trace step: LLM analysis
            mlflow_logger.log_trace_step("llm_analysis", {
                "analysis_time": llm_time,
                "analysis_length": len(analysis),
                "requirement_name": requirement_name
            }, step_number=3)
            
            # Save analysis if output path is provided
            if output_file_path:
                output_path = Path(output_file_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(analysis)
                
                # Log the saved analysis file as artifact
                mlflow_logger.log_artifact(str(output_path), "output_analysis")
                
                logger.info(f"Analysis saved to: {output_path}")
            
            # Log agent I/O
            mlflow_logger.log_agent_input_output("code_corrector", {
                "prompt_template": template_content,
                "student_code": student_code,
                "rendered_prompt": rendered_prompt
            }, {
                "analysis": analysis,
                "analysis_time": llm_time
            })
            
            total_time = time.time() - start_time
            
            # Log final metrics
            mlflow_logger.log_metrics({
                "total_analysis_time_seconds": total_time,
                "analysis_rate": 1.0 / total_time if total_time > 0 else 0
            })
            
            # Log trace step: Analysis complete
            mlflow_logger.log_trace_step("analysis_complete", {
                "total_time": total_time,
                "output_file": output_file_path
            }, step_number=4)
            
            return analysis
            
        except Exception as e:
            # Log error metrics
            mlflow_logger.log_metric("error_occurred", 1.0)
            mlflow_logger.log_text(str(e), "error_log.txt")
            logger.error(f"Error analyzing code: {e}")
            raise
        finally:
            mlflow_logger.end_run()
    
    def _generate_analysis(self, rendered_prompt: str, student_code: str) -> str:
        """Generate analysis using the rendered prompt and LLM"""
        
        # Log the prompt being sent to LLM
        mlflow_logger = get_mlflow_logger()
        mlflow_logger.log_prompt(rendered_prompt, "code_analysis", "llm_prompt")
        
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
        # Start MLflow run for batch analysis
        mlflow_logger = get_mlflow_logger()
        run_id = mlflow_logger.start_run(
            run_name="batch_code_analysis",
            tags={
                "agent": "code_corrector",
                "operation": "batch_analysis",
                "prompt_template": Path(prompt_template_path).name,
                "code_directory": Path(code_directory).name,
                "output_directory": Path(output_directory).name
            }
        )
        
        start_time = time.time()
        logger.info(f"Batch analyzing code files in: {code_directory}")
        
        # Log parameters
        mlflow_logger.log_params({
            "prompt_template_path": prompt_template_path,
            "code_directory": code_directory,
            "output_directory": output_directory,
            "additional_context": additional_context is not None,
            "model_name": self.config.get("model_name", "unknown"),
            "provider": self.config.get("provider", "unknown"),
            "temperature": self.config.get("temperature", 0.1)
        })
        
        try:
            code_path = Path(code_directory)
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find all Python and text files
            code_files = list(code_path.glob("*.py")) + list(code_path.glob("*.txt"))
            
            # Log batch metrics
            mlflow_logger.log_metrics({
                "total_files_to_analyze": len(code_files),
                "python_files": len(list(code_path.glob("*.py"))),
                "text_files": len(list(code_path.glob("*.txt")))
            })
            
            analysis_files = []
            successful_analyses = 0
            failed_analyses = 0
            
            for i, code_file in enumerate(code_files):
                try:
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
                    successful_analyses += 1
                    logger.info(f"Analyzed: {code_file.name} -> {output_filename}")
                    
                    # Log progress metrics
                    mlflow_logger.log_metric("successful_analyses", successful_analyses, step=i+1)
                    mlflow_logger.log_metric("failed_analyses", failed_analyses, step=i+1)
                    mlflow_logger.log_metric("completion_percentage", (i+1) / len(code_files) * 100, step=i+1)
                    
                except Exception as e:
                    failed_analyses += 1
                    logger.error(f"Failed to analyze {code_file.name}: {e}")
                    mlflow_logger.log_metric("failed_analyses", failed_analyses, step=i+1)
            
            total_time = time.time() - start_time
            
            # Log final batch metrics
            mlflow_logger.log_metrics({
                "total_batch_time_seconds": total_time,
                "successful_analyses": successful_analyses,
                "failed_analyses": failed_analyses,
                "success_rate": successful_analyses / len(code_files) if code_files else 0,
                "average_time_per_file": total_time / len(code_files) if code_files else 0
            })
            
            # Log the entire output directory as artifacts
            mlflow_logger.log_artifacts(output_directory, "batch_analysis_results")
            
            logger.info(f"Batch analysis completed. Processed {len(analysis_files)} files")
            return analysis_files
            
        except Exception as e:
            # Log error metrics
            mlflow_logger.log_metric("error_occurred", 1.0)
            mlflow_logger.log_text(str(e), "batch_error_log.txt")
            logger.error(f"Error in batch analysis: {e}")
            raise
        finally:
            mlflow_logger.end_run()
