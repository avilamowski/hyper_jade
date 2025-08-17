# src/agents/code_corrector/code_corrector.py
#!/usr/bin/env python3
"""
Code Correction Agent (LangGraph single-pass)

Evaluates a student code file against a Jinja2 prompt template exactly once
and returns the analysis text. Uses a minimal LangGraph with a single node.

Public API (stable):
- CodeCorrectorAgent(config: dict)
- analyze_code(prompt_template_path, code_file_path, output_file_path=None,
               additional_context=None) -> str
- batch_analyze(prompt_template_path, code_directory, output_directory,
               additional_context=None) -> list[str]
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, TypedDict
from pathlib import Path
import logging
import time
import re

from dotenv import load_dotenv, dotenv_values
from jinja2 import Template
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage

# LangGraph
from langgraph.graph import StateGraph, END

from src.config import get_agent_config
from src.agents.utils.agent_evaluator import AgentEvaluator

# --------------------------------------------------------------------------- #
# Load .env ASAP and keep values in-memory
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
DOTENV = dotenv_values()


# --------------------------------------------------------------------------- #
# MLflow logger (optional)
# --------------------------------------------------------------------------- #
def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed."""
    try:
        from src.core.mlflow_utils import mlflow_logger  # type: ignore
        return mlflow_logger
    except ImportError:
        logger.warning("MLflow not available - logging will be disabled")
        return None

def safe_log_call(logger_instance, method_name, *args, **kwargs):
    """Safely call a logging method, doing nothing if logger is None"""
    if logger_instance is not None and hasattr(logger_instance, method_name):
        try:
            method = getattr(logger_instance, method_name)
            method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to call {method_name}: {e}")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# --------------------------------------------------------------------------- #
# LangGraph State
# --------------------------------------------------------------------------- #
class AnalysisState(TypedDict, total=False):
    prompt_template: str
    student_code: str
    additional_context: Optional[str]
    rendered_prompt: str
    analysis: str
    timings: Dict[str, float]
    error: Optional[str]


# --------------------------------------------------------------------------- #
# CodeCorrectorAgent
# --------------------------------------------------------------------------- #
class CodeCorrectorAgent:
    """Agent that analyzes code against a requirement using a single-pass LangGraph."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.agent_config = get_agent_config(self.config, "code_corrector")
        self.llm = self._setup_llm()
        self.ml = get_mlflow_logger()
        
        # Initialize agent evaluator if enabled
        self.evaluator = None
        if config.get('agents', {}).get('agent_evaluator', {}).get('enabled', False):
            self.evaluator = AgentEvaluator(config)

    # -------------------------- LLM Setup ----------------------------------- #
    def _setup_llm(self):
        provider = str(self.agent_config.get("provider", "openai")).lower().strip()
        model_name = self.agent_config.get("model_name", "gpt-4o-mini")
        temperature = float(self.agent_config.get("temperature", 0.1))

        if provider == "openai":
            # Prefer explicit api_key from config, else pull from .env dict
            api_key = self.agent_config.get("api_key") or DOTENV.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY missing from .env and no api_key provided in config."
                )
            logger.info(
                f"[CodeCorrector] Using OpenAI model={model_name} temp={temperature}"
            )
            return ChatOpenAI(
                model=model_name, temperature=temperature, api_key=api_key
            )

        logger.info(
            f"[CodeCorrector] Using Ollama model={model_name} temp={temperature}"
        )
        return OllamaLLM(model=model_name, temperature=temperature)

    # -------------------------- Utilities ----------------------------------- #
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        return re.sub(r"```[a-zA-Z]*\n|```", "", text).strip()

    def _call_llm(self, rendered_prompt: str) -> str:
        try:
            if isinstance(self.llm, ChatOpenAI):
                resp = self.llm.invoke([HumanMessage(content=rendered_prompt)])
                out = getattr(resp, "content", str(resp))
            else:
                out = self.llm.invoke(rendered_prompt)
            return self._strip_code_fences(str(out))
        except Exception as e:
            logger.error(f"[CodeCorrector] LLM invocation failed: {e}")
            raise

    # -------------------------- Graph Node ---------------------------------- #
    def _node_analyze_once(self, state: AnalysisState) -> AnalysisState:
        """Single node that renders template and invokes the LLM exactly once."""
        ml = self.ml
        t0 = time.time()
        prompt_template = state.get("prompt_template", "")
        code = state.get("student_code", "")
        add_ctx = state.get("additional_context")

        # Render template
        tr0 = time.time()
        rendered = Template(prompt_template).render(code=code, context=add_ctx)
        tr1 = time.time()

        # Log prompt before sending
        safe_log_call(ml, "log_prompt",
            rendered, prompt_name="code_corrector", step="rendered_prompt"
        )

        # Call LLM
        tl0 = time.time()
        analysis = self._call_llm(rendered)
        tl1 = time.time()

        # Log timings
        timings = {
            "node_total_sec": time.time() - t0,
            "template_render_sec": tr1 - tr0,
            "llm_invoke_sec": tl1 - tl0,
        }

        # Trace steps
        safe_log_call(ml, "log_trace_step",
            "template_rendering",
            {
                "render_time": timings["template_render_sec"],
                "rendered_len": len(rendered),
            },
            step_number=1,
        )
        safe_log_call(ml, "log_trace_step",
            "llm_analysis",
            {
                "analysis_time": timings["llm_invoke_sec"],
                "analysis_len": len(analysis),
            },
            step_number=2,
        )

        # Artifacts
        safe_log_call(ml, "log_text", rendered, "rendered_prompt.txt")
        safe_log_call(ml, "log_text", analysis, "generated_analysis.txt")

        # Code-analysis metrics (semantic parser is inside your mlflow utils)
        safe_log_call(ml, "log_code_analysis_metrics",
            analysis, requirement_name="code_corrector_requirement"
        )

        state["rendered_prompt"] = rendered
        state["analysis"] = analysis
        state["timings"] = timings
        return state

    # -------------------------- Graph Builder ------------------------------- #
    def _build_graph(self):
        """Build a minimal single-pass LangGraph."""
        graph = StateGraph(AnalysisState)
        graph.add_node("analyze_once", self._node_analyze_once)
        graph.set_entry_point("analyze_once")
        graph.add_edge("analyze_once", END)
        return graph.compile()

    # -------------------------- Public API ---------------------------------- #
    def analyze_code(
        self,
        prompt_template_path: str,
        code_file_path: str,
        output_file_path: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Analyze code against a Jinja2 prompt ONCE using LangGraph.
        """
        ml = self.ml
        start = time.time()

        # Start run + params
        safe_log_call(ml, "start_run",
            run_name="code_analysis",
            tags={
                "agent": "code_corrector",
                "prompt_template": Path(prompt_template_path).name,
                "code_file": Path(code_file_path).name,
                "output_file": (
                    Path(output_file_path).name if output_file_path else "none"
                ),
            },
        )
        safe_log_call(ml, "log_params",
            {
                "prompt_template_path": prompt_template_path,
                "code_file_path": code_file_path,
                "output_file_path": output_file_path or "",
                "has_additional_context": bool(additional_context),
                "model_name": self.agent_config.get("model_name", "unknown"),
                "provider": self.agent_config.get("provider", "unknown"),
                "temperature": self.agent_config.get("temperature", 0.1),
            }
        )

        try:
            # Read inputs
            template_content = Path(prompt_template_path).read_text(encoding="utf-8")
            student_code = Path(code_file_path).read_text(encoding="utf-8")

            # Log inputs as artifacts
            safe_log_call(ml, "log_text", template_content, "input_prompt_template.jinja")
            safe_log_call(ml, "log_text", student_code, "input_student_code.py")
            if additional_context:
                safe_log_call(ml, "log_text", str(additional_context), "input_additional_context.txt")

            # Trace step: inputs read
            safe_log_call(ml, "log_trace_step",
                "read_inputs",
                {
                    "template_length": len(template_content),
                    "code_length": len(student_code),
                    "has_additional_context": bool(additional_context),
                },
                step_number=0,
            )

            # Build and run graph
            app = self._build_graph()
            state_in: AnalysisState = {
                "prompt_template": template_content,
                "student_code": student_code,
                "additional_context": additional_context,
            }
            result_state = app.invoke(state_in)

            analysis = result_state.get("analysis", "")

            # Save output if requested
            if output_file_path:
                out = Path(output_file_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(analysis, encoding="utf-8")
                safe_log_call(ml, "log_artifact", str(out), "output_analysis")

            # Log agent I/O
            safe_log_call(ml, "log_agent_input_output",
                "code_corrector",
                {
                    "prompt_template": template_content,
                    "student_code": student_code,
                    "rendered_prompt": result_state.get("rendered_prompt", ""),
                },
                {
                    "analysis": analysis,
                    "timings": result_state.get("timings", {}),
                },
            )

            # Evaluate agent output if evaluator is enabled
            if self.evaluator:
                try:
                    logger.info("Evaluating code corrector output...")
                    # Get requirement text from prompt template (simplified approach)
                    requirement_text = "Code analysis requirement"  # This could be enhanced
                    
                    evaluation_result = self.evaluator.evaluate_code_corrector(
                        requirement_text,
                        template_content,
                        student_code,
                        analysis,
                        output_file_path or "no_output_file"
                    )
                    
                    # Log evaluation metrics
                    safe_log_call(ml, "log_agent_evaluation_metrics", "code_corrector", evaluation_result)
                    
                    logger.info(f"Code corrector evaluation completed. Overall score: {evaluation_result.get('overall_score', 'N/A')}")
                    
                except Exception as eval_error:
                    logger.warning(f"Error during code corrector evaluation: {eval_error}")

            total_time = time.time() - start
            safe_log_call(ml, "log_metrics",
                {
                    "total_analysis_time_seconds": total_time,
                    "analysis_length_chars": len(analysis),
                    "analysis_length_lines": len(analysis.splitlines()),
                }
            )
            safe_log_call(ml, "log_trace_step",
                "analysis_complete",
                {"total_time": total_time, "output_file": output_file_path, "evaluation_performed": self.evaluator is not None},
                step_number=3,
            )

            logger.info(f"[CodeCorrector] Analysis complete in {total_time:.3f}s")
            return analysis

        except Exception as e:
            safe_log_call(ml, "log_metric", "error_occurred", 1.0)
            safe_log_call(ml, "log_text", str(e), "error_log.txt")
            logger.exception("[CodeCorrector] Error during analyze_code")
            raise
        finally:
            safe_log_call(ml, "end_run")

    def batch_analyze(
        self,
        prompt_template_path: str,
        code_directory: str,
        output_directory: str,
        additional_context: Optional[str] = None,
    ) -> List[str]:
        """
        Analyze all .py and .txt files in a directory using the same prompt.
        Returns a list of generated output file paths.
        """
        ml = self.ml
        start = time.time()

        safe_log_call(ml, "start_run",
            run_name="batch_code_analysis",
            tags={
                "agent": "code_corrector",
                "operation": "batch_analysis",
                "prompt_template": Path(prompt_template_path).name,
                "code_directory": Path(code_directory).name,
                "output_directory": Path(output_directory).name,
            },
        )
        safe_log_call(ml, "log_params",
            {
                "prompt_template_path": prompt_template_path,
                "code_directory": code_directory,
                "output_directory": output_directory,
                "has_additional_context": bool(additional_context),
                "model_name": self.agent_config.get("model_name", "unknown"),
                "provider": self.agent_config.get("provider", "unknown"),
                "temperature": self.agent_config.get("temperature", 0.1),
            }
        )

        try:
            code_path = Path(code_directory)
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)

            py_files = list(code_path.glob("*.py"))
            txt_files = list(code_path.glob("*.txt"))
            files: List[Path] = sorted(py_files + txt_files)

            safe_log_call(ml, "log_metrics",
                {
                    "total_files_to_analyze": len(files),
                    "python_files": len(py_files),
                    "text_files": len(txt_files),
                }
            )

            outputs: List[str] = []
            success = 0
            fail = 0

            for i, f in enumerate(files, start=1):
                try:
                    target = out_dir / f"{f.stem}.analysis.txt"
                    self.analyze_code(
                        prompt_template_path=prompt_template_path,
                        code_file_path=str(f),
                        output_file_path=str(target),
                        additional_context=additional_context,
                    )
                    outputs.append(str(target))
                    success += 1
                except Exception:
                    fail += 1
                finally:
                    safe_log_call(ml, "log_metric", "successful_analyses", success, step=i)
                    safe_log_call(ml, "log_metric", "failed_analyses", fail, step=i)
                    safe_log_call(ml, "log_metric",
                        "completion_percentage",
                        (i / max(1, len(files))) * 100.0,
                        step=i,
                    )

            total_time = time.time() - start
            safe_log_call(ml, "log_metrics",
                {
                    "total_batch_time_seconds": total_time,
                    "successful_analyses": success,
                    "failed_analyses": fail,
                    "success_rate": (success / len(files)) if files else 0.0,
                    "average_time_per_file": (
                        (total_time / len(files)) if files else 0.0
                    ),
                }
            )
            safe_log_call(ml, "log_artifacts", output_directory, "batch_analysis_results")

            logger.info(
                f"[CodeCorrector] Batch complete: {success}/{len(files)} succeeded "
                f"in {total_time:.3f}s"
            )
            return outputs

        except Exception as e:
            safe_log_call(ml, "log_metric", "error_occurred", 1.0)
            safe_log_call(ml, "log_text", str(e), "batch_error_log.txt")
            logger.exception("[CodeCorrector] Error during batch_analyze")
            raise
        finally:
            safe_log_call(ml, "end_run")
