"""Composite LLM wrapper and small backend factory.

This module provides a small CompositeLLM that holds multiple backend clients
and returns small stage-bound objects that expose an .invoke(...) method so
existing agents can keep calling `.invoke(...)` without modification.

Supported providers out of the box: "openai" and "ollama". Other providers
can be added by implementing the `_create_backend_from_config` function.

The composite pattern lets the runner create one shared CompositeLLM and
bind per-stage clients (prompt_generation / code_corrector / evaluation).
Using a single composite instance makes it easier to create a single
instrumentation/entrypoint for tracing if you want to extend it later.
"""
from __future__ import annotations
from typing import Any, Dict
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM


def _create_backend_from_config(agent_config: Dict[str, Any]) -> Any:
    """Create a backend client from an agent config dict.

    agent_config should contain at least `provider` and `model_name` keys.
    """
    provider = str(agent_config.get("provider", "openai")).lower().strip()
    model_name = agent_config.get("model_name")
    temperature = float(agent_config.get("temperature", 0.1))

    if provider == "openai":
        # ChatOpenAI accepts api_key param but will also read from env if needed
        api_key = agent_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name or "gpt-4", temperature=temperature, api_key=api_key)

    if provider in ("ollama", "gpt-oss", "local-ollama"):
        return OllamaLLM(model=model_name or "qwen2.5:7b", temperature=temperature)

    # Placeholder for other providers (deepseek, llama_cpp, etc.)
    raise NotImplementedError(f"Provider '{provider}' is not implemented in composite_llm. Add an adapter.")


class _StageBoundLLM:
    """Simple object returned to agents that binds a composite and a stage.

    This keeps agents unchanged: they call `.invoke(...)` and the bound object
    routes to the correct backend inside the CompositeLLM.
    """

    def __init__(self, composite: "CompositeLLM", stage: str):
        self._composite = composite
        self._stage = stage

    def invoke(self, messages_or_text):
        return self._composite.invoke_for_stage(self._stage, messages_or_text)


class CompositeLLM:
    """Holds backend LLM clients and routes calls by stage name.

    backends: dict(stage_name -> backend_client)
    Each backend_client should implement an `invoke(...)` method (Chat-like) or
    at least be callable in a similar way. Agents already tolerate different
    return types (they check for `.content`), so minimal normalization is done.
    """

    def __init__(self, backends: Dict[str, Any]):
        self.backends = dict(backends)

    @classmethod
    def from_agent_configs(cls, mapping: Dict[str, Dict[str, Any]]):
        """Create a CompositeLLM from a mapping stage -> agent_config."""
        backends: Dict[str, Any] = {}
        for stage, cfg in mapping.items():
            backends[stage] = _create_backend_from_config(cfg)
        return cls(backends)

    def get_bound(self, stage: str) -> _StageBoundLLM:
        return _StageBoundLLM(self, stage)

    def get_shared_bound(self, initial_stage: str = None):
        """Return a single shared bound object whose `stage` can be updated
        at runtime. Use this when you want a single Python object to be used
        by multiple agents but route calls to different backends by changing
        the `stage` attribute before invoking.
        """
        return SharedStageBoundLLM(self, initial_stage)

    def invoke_for_stage(self, stage: str, messages_or_text):
        backend = self.backends.get(stage)
        if backend is None:
            raise RuntimeError(f"No backend configured for stage '{stage}'")

        # Many LangChain LLMs expose .invoke and accept either a list of
        # HumanMessage or raw text. Try to call .invoke and let exceptions
        # propagate; agents already handle different return types.
        if hasattr(backend, "invoke"):
            try:
                return backend.invoke(messages_or_text)
            except TypeError:
                # Some backends expect a single string instead of a list
                return backend.invoke(str(messages_or_text))

        # Fallback: if backend is a simple callable
        if callable(backend):
            return backend(messages_or_text)

        raise RuntimeError(f"Backend for stage '{stage}' is not callable nor implements 'invoke'.")


class SharedStageBoundLLM:
    """A mutable bound object that holds an active `stage` and delegates
    invoke calls to the composite's backend for that stage.

    Use case: create one SharedStageBoundLLM and assign it to multiple agents
    so they all call the same Python object (good for shared tracing). The
    runner should set `shared.stage = 'code_correction'` (or other) before
    invoking an agent that expects that backend.
    """

    def __init__(self, composite: CompositeLLM, initial_stage: str | None = None):
        self._composite = composite
        self.stage = initial_stage

    def invoke(self, messages_or_text):
        if not self.stage:
            raise RuntimeError("SharedStageBoundLLM.stage not set. Set it to a stage name before invoking.")
        return self._composite.invoke_for_stage(self.stage, messages_or_text)
