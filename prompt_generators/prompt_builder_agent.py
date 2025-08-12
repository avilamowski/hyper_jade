from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any
from pathlib import Path
import logging
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Paths & config
# ----------------------------
BASE_PATH = Path(__file__).parent
TEMPLATES_PATH = BASE_PATH / "prompts"

with open(BASE_PATH / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MODEL_NAME = CONFIG.get("model", "qwen2.5:7b")
LLM_KWARGS = CONFIG.get("llm_kwargs", {}) or {}

# ----------------------------
# LLM (Ollama)
# ----------------------------
llm = OllamaLLM(model=MODEL_NAME, **LLM_KWARGS)

# ----------------------------
# Jinja
# ----------------------------
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_PATH)),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(relpath: str, context: Dict[str, Any]) -> str:
    return jinja_env.get_template(relpath).render(**context)


# ----------------------------
# LangGraph state
# ----------------------------
class PromptBuilderState(TypedDict, total=False):
    rubric_yaml: str
    assignment_md: str
    additional_policies: Optional[str]
    max_tokens_hint: Optional[int]
    refine: bool

    draft_prompt: str
    final_prompt: str


# ----------------------------
# Nodes
# ----------------------------
def build_draft_node(state: PromptBuilderState) -> PromptBuilderState:
    logger.info("Building tag-based grading prompt from rubric + assignment...")
    context = {
        "rubric_yaml": state["rubric_yaml"],
        "assignment_md": state["assignment_md"],
        "additional_policies": state.get("additional_policies", ""),
    }
    draft = render_template("grading/build_from_rubric_and_assignment.jinja", context)
    return {"draft_prompt": draft}


def refine_draft_node(state: PromptBuilderState) -> PromptBuilderState:
    logger.info("Refining draft prompt with LLM...")
    sys_msg = render_template("refine/system.jinja", {})
    user_msg = render_template(
        "refine/user.jinja",
        {
            "draft_prompt": state["draft_prompt"],
            "max_tokens_hint": state.get("max_tokens_hint", 1800),
        },
    )
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]
    refined = llm.invoke(messages)
    return {"final_prompt": str(refined).strip()}


def should_refine(state: PromptBuilderState) -> bool:
    return bool(state.get("refine", True))


# ----------------------------
# Graph
# ----------------------------
graph = StateGraph(PromptBuilderState)
graph.add_node("build_draft", build_draft_node)
graph.add_node("refine_draft", refine_draft_node)
graph.set_entry_point("build_draft")
graph.add_conditional_edges(
    "build_draft", should_refine, {True: "refine_draft", False: END}
)
graph.add_edge("refine_draft", END)
compiled_graph = graph.compile()


# ----------------------------
# Public API
# ----------------------------
def generate_prompt_from_rubric_and_assignment(
    rubric_yaml: str,
    assignment_md: str,
    *,
    additional_policies: Optional[str] = None,
    max_tokens_hint: Optional[int] = 1800,
    refine: bool = True,
) -> str:
    init: PromptBuilderState = {
        "rubric_yaml": rubric_yaml,
        "assignment_md": assignment_md,
        "additional_policies": additional_policies,
        "max_tokens_hint": max_tokens_hint,
        "refine": refine,
    }
    result = compiled_graph.invoke(init)
    return result.get("final_prompt") or result["draft_prompt"]


def generate_prompt_from_files(
    rubric_path: str | Path,
    assignment_path: str | Path,
    *,
    additional_policies: Optional[str] = None,
    max_tokens_hint: Optional[int] = 1800,
    refine: bool = True,
) -> str:
    rubric_text = Path(rubric_path).read_text(encoding="utf-8")
    assignment_text = Path(assignment_path).read_text(encoding="utf-8")

    if not str(rubric_path).lower().endswith((".yaml", ".yml")):
        logger.warning("Expected rubric file to be .yaml/.yml (got %s)", rubric_path)
    if not str(assignment_path).lower().endswith(".md"):
        logger.warning("Expected assignment file to be .md (got %s)", assignment_path)

    return generate_prompt_from_rubric_and_assignment(
        rubric_yaml=rubric_text,
        assignment_md=assignment_text,
        additional_policies=additional_policies,
        max_tokens_hint=max_tokens_hint,
        refine=refine,
    )


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    examples_dir = BASE_PATH.parent / "examples"
    rubric_file = examples_dir / "rubric.yaml"
    assignment_file = examples_dir / "assignment.md"

    prompt = generate_prompt_from_files(
        rubric_path=rubric_file,
        assignment_path=assignment_file,
        refine=True,
        additional_policies="Do not execute code. Perform static analysis only.",
    )
    print("\n" + "=" * 80 + "\nFINAL PROMPT (for the grader agent):\n" + "=" * 80)
    print(prompt)
