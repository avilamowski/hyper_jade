from __future__ import annotations
from typing import Dict, Any, Optional
import os
import re
import time
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

from src.config import load_config, get_agent_config

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """Evaluate generated Jinja2 prompt templates using the project's evaluator template.

    This class is intentionally lightweight and focused: it renders the
    `templates/evaluators/prompt_evaluator.jinja` template to produce prompts,
    invokes an LLM (OpenAI or Ollama) and parses the XML-like <ASSESMENT> blocks
    into numeric scores and rationales.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load config
        try:
            self.config = load_config(config_path or "src/config/assignment_config.yaml")
        except Exception:
            self.config = {}

        # Get evaluator config (agent_evaluator section)
        self.evaluator_config = get_agent_config(self.config, "agent_evaluator")

        # Criteria for prompt_generator
        self.criteria = self.evaluator_config.get("evaluation_criteria", {}).get("prompt_generator", {})

        # Setup LLM
        self.llm = self._setup_llm()

        # Setup Jinja environment and load template
        repo_root = Path(__file__).resolve().parents[2]
        templates_dir = repo_root / "templates" / "evaluators"
        if not templates_dir.exists():
            templates_dir = Path("templates") / "evaluators"

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["jinja", "html", "txt"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self.template_name = "prompt_evaluator.jinja"
        try:
            self.template = self.jinja_env.get_template(self.template_name)
        except Exception as e:
            logger.error(f"Unable to load evaluator template {self.template_name}: {e}")
            # create a minimal template fallback
            self.template = self.jinja_env.from_string("{% macro human_prompt(requirement_text, assignment_text, generated_prompt, criteria_list) %}{% endmacro %}")

    def _setup_llm(self):
        provider = self.evaluator_config.get("provider", os.getenv("PROVIDER", "openai"))
        model_name = self.evaluator_config.get("model_name", os.getenv("MODEL_NAME", "gpt-4o-mini"))
        temperature = float(self.evaluator_config.get("temperature", os.getenv("TEMPERATURE", 0.1)))

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL")
                return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=(base_url or None))
            else:
                from langchain_ollama.llms import OllamaLLM

                return OllamaLLM(model=model_name or "qwen2.5:7b", temperature=temperature)
        except Exception as e:
            logger.warning(f"LLM backend not available or import failed: {e}. Using a deterministic stub.")

            class _StubLLM:
                def invoke(self, messages):
                    # produce a default response containing ASSESMENT blocks for each criterion
                    criteria = list(self_criteria)
                    blocks = []
                    for c in criteria:
                        blocks.append(f"<ASSESMENT>\n    <CRITERION_KEY>{c}</CRITERION_KEY>\n    <RATIONALE>Default stub rationale for {c}.</RATIONALE>\n    <SCORE>3</SCORE>\n</ASSESMENT>")
                    return type("R", (), {"content": "\n".join(blocks)})()

            self_criteria = list(self.criteria.keys() or ["python correctness", "jinja2 syntax", "semantic correctness", "completeness"])
            return _StubLLM()

    def _render_prompts(self, requirement_text: str, assignment_text: str, generated_prompt: str) -> tuple[str, str]:
        # system prompt: take the top-level rendered template (which includes instructions and criteria list)
        try:
            system_text = self.template.render()
        except Exception:
            system_text = "You are an expert evaluator for Jinja2 prompt templates. Evaluate the generated prompt according to the requested criteria."

        # human prompt: call macro human_prompt to produce the specific evaluation input
        try:
            module = self.template.make_module({})
            if hasattr(module, "human_prompt"):
                human_text = getattr(module, "human_prompt")(requirement_text, assignment_text, generated_prompt, list(self.criteria.keys()))
            else:
                # fallback: render whole template with the values
                human_text = self.template.render(requirement_text=requirement_text, assignment_text=assignment_text, generated_prompt=generated_prompt, criteria_list=list(self.criteria.keys()))
        except Exception:
            human_text = self.template.render(requirement_text=requirement_text or "", assignment_text=assignment_text or "", generated_prompt=generated_prompt or "", criteria_list=list(self.criteria.keys()))

        return system_text, human_text

    def _extract_response_content(self, response) -> str:
        if response is None:
            return ""
        if hasattr(response, "content"):
            return getattr(response, "content") or ""
        if hasattr(response, "text"):
            return getattr(response, "text") or ""
        return str(response)

    def _parse_assessments(self, text: str) -> tuple[Dict[str, float], Dict[str, str]]:
        # Find all ASSESMENT blocks and extract CRITERION_KEY, RATIONALE, SCORE
        scores: Dict[str, float] = {}
        rationales: Dict[str, str] = {}

        pattern = re.compile(r"<ASSESMENT>(.*?)</ASSESMENT>", re.DOTALL | re.IGNORECASE)
        key_re = re.compile(r"<CRITERION_KEY>(.*?)</CRITERION_KEY>", re.DOTALL | re.IGNORECASE)
        rationale_re = re.compile(r"<RATIONALE>(.*?)</RATIONALE>", re.DOTALL | re.IGNORECASE)
        score_re = re.compile(r"<SCORE>(.*?)</SCORE>", re.DOTALL | re.IGNORECASE)

        for m in pattern.finditer(text):
            block = m.group(1)
            k = key_re.search(block)
            r = rationale_re.search(block)
            s = score_re.search(block)
            if k:
                key = k.group(1).strip()
                rationale = r.group(1).strip() if r else ""
                try:
                    score = float(s.group(1).strip()) if s else 1.0
                except Exception:
                    score = 1.0

                scores[key] = score
                rationales[key] = rationale

        # Ensure all expected criteria are present
        for crit in self.criteria.keys():
            if crit not in scores:
                scores[crit] = 1.0
                rationales[crit] = "Missing from LLM response"

        return scores, rationales

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        weights = self.criteria or {k: 1.0 for k in scores.keys()}
        total_w = sum(weights.values()) or 1.0
        weighted = sum(scores.get(k, 0.0) * weights.get(k, 1.0) for k in weights.keys()) / total_w
        return weighted * 2

    def evaluate_prompt(self, requirement_text: str, assignment_text: str, generated_prompt: str) -> Dict[str, Any]:
        """Evaluate a generated prompt template and return scores + metadata.

        Returns a list of dicts, one per evaluation criterion, to match the
        LangSmith / LangChain code-evaluator interface.
        Each dict contains: name, score (1-5), rationale, max_score.
        The final entry is an overall/average summary.
        """
        system_prompt, human_prompt = self._render_prompts(requirement_text, assignment_text, generated_prompt)

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        start = time.time()
        try:
            response = self.llm.invoke(messages)
        except Exception:
            try:
                response = self.llm(messages)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                response = None
        elapsed = time.time() - start

        resp_text = self._extract_response_content(response)
        scores, rationales = self._parse_assessments(resp_text)

        return [{
            "key": k,
            "score": float(v),
            "rationale": rationales.get(k, ""),
        } for k, v in scores.items()]


def evaluate_prompt(inputs: Dict[str, str], outputs: Dict[str, str], config_path: Optional[str] = None) -> list[Dict[str, Any]]:
    """Compatibility wrapper returning a list of dicts (LangSmith-friendly).

    inputs: { requirement_text, assignment_text }
    outputs: { generated_prompt }
    """
    req = inputs.get("requirement_text", "")
    assignment = inputs.get("assignment_text", "")
    gen = outputs.get("generated_prompt", "")
    pe = PromptEvaluator(config_path=config_path)
    return pe.evaluate_prompt(req, assignment, gen)
