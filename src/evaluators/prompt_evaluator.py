from __future__ import annotations
from typing import Dict, Any, Optional
import os
import re
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import load_config, get_agent_config

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """Evaluate generated Jinja2 prompt templates."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path or "src/config/assignment_config.yaml")
        self.evaluator_config = get_agent_config(self.config, "agent_evaluator")
        self.criteria = self.evaluator_config["evaluation_criteria"]["prompt_generator"]
        self.llm = self._setup_llm()

        repo_root = Path(__file__).resolve().parents[2]
        templates_dir = repo_root / "templates" / "evaluators"
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["jinja", "html", "txt"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self.template = self.jinja_env.get_template("prompt_evaluator.jinja")

    def _setup_llm(self):
        provider = self.evaluator_config["provider"]
        model_name = self.evaluator_config["model_name"]
        temperature = float(self.evaluator_config["temperature"])

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        else:
            from langchain_ollama.llms import OllamaLLM
            return OllamaLLM(model=model_name, temperature=temperature)

    def _render_prompts(self, requirement_text: str, assignment_text: str, generated_prompt: str) -> tuple[str, str]:
        system_text = self.template.render()
        
        module = self.template.make_module({})
        human_text = module.human_prompt(requirement_text, assignment_text, generated_prompt, list(self.criteria.keys()))
        
        return system_text, human_text

    def _extract_response_content(self, response) -> str:
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        return str(response)

    def _parse_assessments(self, text: str) -> tuple[Dict[str, float], Dict[str, str]]:
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
                score = float(s.group(1).strip()) if s else 1.0

                scores[key] = score
                rationales[key] = rationale

        return scores, rationales

    def evaluate_prompt(self, requirement_text: str, assignment_text: str, generated_prompt: str) -> list[Dict[str, Any]]:
        """Evaluate a generated prompt template and return scores + metadata."""
        system_prompt, human_prompt = self._render_prompts(requirement_text, assignment_text, generated_prompt)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        response = self.llm.invoke(messages)
        resp_text = self._extract_response_content(response)
        scores, rationales = self._parse_assessments(resp_text)

        return [{
            "key": k,
            "score": float(v),
            "rationale": rationales.get(k, ""),
        } for k, v in scores.items()]


def evaluate_prompt(inputs: Dict[str, str], outputs: Dict[str, str], config_path: Optional[str] = None) -> list[Dict[str, Any]]:
    """Compatibility wrapper returning a list of dicts (LangSmith-friendly)."""
    req = inputs["requirement_text"]
    assignment = inputs["assignment_text"]
    gen = outputs["generated_prompt"]
    pe = PromptEvaluator(config_path=config_path)
    return pe.evaluate_prompt(req, assignment, gen)
