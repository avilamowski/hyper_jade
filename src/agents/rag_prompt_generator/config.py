"""
RAG Prompt Generator Configuration

Configuration for RAG functionality - reads from YAML and environment variables.
Environment variables are only required when RAG is enabled (use_rag: true).
"""

import os
import yaml
from typing import Optional
from pathlib import Path

def _require_env_var(var_name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} environment variable is required but not set")
    return value

# Load RAG configuration from YAML file FIRST
def load_rag_config():
    """
    Load RAG configuration from YAML file.
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """ 
    import logging
    logger = logging.getLogger(__name__)
    
    config_path = Path(__file__).parent.parent.parent / "config" / "rag_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"RAG config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError(f"RAG config file is empty: {config_path}")
    
    return config

# Load configuration (will raise if file missing or invalid)
_rag_config = load_rag_config()

# Main Control Flag - from YAML config only (required)
if "use_rag" not in _rag_config:
    raise ValueError("'use_rag' key is required in rag_config.yaml")
USE_RAG = _rag_config["use_rag"]

# Raise error if 'rag' section is missing and USE_RAG is True
if USE_RAG and "rag" not in _rag_config:
    raise ValueError("'rag' section is required in rag_config.yaml when use_rag is True")

# Get RAG section (required if USE_RAG is True)
rag_section = _rag_config.get("rag", {})

# Read configuration from YAML (when RAG is enabled)
if USE_RAG:
    # Weaviate Configuration - from YAML
    if "weaviate" not in rag_section:
        raise ValueError("'rag.weaviate' section is required in rag_config.yaml")
    WEAVIATE_URL = rag_section["weaviate"]["url"]
    
    # AI Provider will be inherited from main config via get_main_llm_config()
    # So we don't read it here - it will be set later
    RAG_AI_PROVIDER = None  # Will be set from main config
    RAG_MODEL_NAME = None   # Will be set from main config
    RAG_OLLAMA_HOST = None  # Will be set from main config if needed
    RAG_OLLAMA_PORT = None  # Will be set from main config if needed
else:
    # Defaults when RAG is disabled (these won't be used)
    WEAVIATE_URL = None
    RAG_AI_PROVIDER = None
    RAG_OLLAMA_HOST = None
    RAG_OLLAMA_PORT = None
    RAG_MODEL_NAME = None

# API Keys - from environment variables (optional)
RAG_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAG_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL")
RAG_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Google/Gemini Configuration - from environment variables (optional)
RAG_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RAG_GOOGLE_MODEL = os.getenv("RAG_GOOGLE_MODEL")

# Text Splitter Configuration (required from YAML if USE_RAG is True)
if USE_RAG:
    if "text_splitter" not in rag_section:
        raise ValueError("'rag.text_splitter' section is required in rag_config.yaml")
    text_splitter = rag_section["text_splitter"]
    RAG_TEXT_SPLITTER_STRATEGY = text_splitter["strategy"]
    RAG_CHUNK_SIZE = text_splitter["chunk_size"]
    RAG_CHUNK_OVERLAP = text_splitter["chunk_overlap"]
    RAG_MIN_CHUNK_SIZE = text_splitter["min_chunk_size"]
else:
    # Defaults when RAG is disabled
    RAG_TEXT_SPLITTER_STRATEGY = "cell_based"
    RAG_CHUNK_SIZE = 1500
    RAG_CHUNK_OVERLAP = 250
    RAG_MIN_CHUNK_SIZE = 100

# Reranking Configuration (required from YAML if USE_RAG is True)
if USE_RAG:
    if "reranking" not in rag_section:
        raise ValueError("'rag.reranking' section is required in rag_config.yaml")
    reranking = rag_section["reranking"]
    RAG_ENABLE_RERANKING = reranking["enabled"]
    RAG_RERANKING_MODEL = reranking["model"]
    RAG_INITIAL_RETRIEVAL_COUNT = reranking["initial_retrieval_count"]
    RAG_FINAL_RETRIEVAL_COUNT = reranking["final_retrieval_count"]
else:
    # Defaults when RAG is disabled
    RAG_ENABLE_RERANKING = False
    RAG_RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_INITIAL_RETRIEVAL_COUNT = 15
    RAG_FINAL_RETRIEVAL_COUNT = 5

# Temperature Configuration (required from YAML if USE_RAG is True)
if USE_RAG:
    if "temperatures" not in rag_section:
        raise ValueError("'rag.temperatures' section is required in rag_config.yaml")
    temps = rag_section["temperatures"]
    RAG_TEMPERATURE_EXAMPLE_GENERATION = temps["example_generation"]
    RAG_TEMPERATURE_THEORY_CORRECTION = temps["theory_correction"]
    RAG_TEMPERATURE_FILTERING = temps["filtering"]
else:
    # Defaults when RAG is disabled
    RAG_TEMPERATURE_EXAMPLE_GENERATION = 0.7
    RAG_TEMPERATURE_THEORY_CORRECTION = 0.3
    RAG_TEMPERATURE_FILTERING = 0.1

# Dataset Configuration (required from YAML if USE_RAG is True)
if USE_RAG:
    if "datasets" not in rag_section:
        raise ValueError("'rag.datasets' section is required in rag_config.yaml")
    datasets = rag_section["datasets"]
    if "python" in datasets:
        RAG_PYTHON_NOTEBOOKS_DIR = datasets["python"]["notebooks_dir"]
    else:
        raise ValueError("'rag.datasets.python' is required in rag_config.yaml")
    if "haskell" in datasets:
        RAG_HASKELL_NOTEBOOKS_DIR = datasets["haskell"]["notebooks_dir"]
    else:
        RAG_HASKELL_NOTEBOOKS_DIR = None  # Optional
else:
    # Defaults when RAG is disabled
    RAG_PYTHON_NOTEBOOKS_DIR = "data/Clases"
    RAG_HASKELL_NOTEBOOKS_DIR = "data/learnyouahaskell"

# Debug Configuration (required from YAML if USE_RAG is True)
if USE_RAG:
    if "debug" not in rag_section:
        raise ValueError("'rag.debug' section is required in rag_config.yaml")
    debug_config = rag_section["debug"]
    RAG_DEBUG_MODE = debug_config["enabled"]
    RAG_ENABLE_FILTERING = debug_config["filtering_enabled"]
else:
    # Defaults when RAG is disabled
    RAG_DEBUG_MODE = False
    RAG_ENABLE_FILTERING = True

# LangSmith Configuration (optional)
langsmith_config = rag_section.get("langsmith", {})
RAG_LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")  # Optional
RAG_LANGSMITH_PROJECT = langsmith_config.get("project", "hyper-jade-rag")
RAG_LANGSMITH_ENDPOINT = langsmith_config.get("endpoint", "https://api.smith.langchain.com")
RAG_LANGSMITH_TRACING = langsmith_config.get("enabled", False)
RAG_ENABLE_LANGSMITH_TRACING = RAG_LANGSMITH_TRACING

# Theory Improvement Model Configuration - from YAML (optional)
theory_improvement_config = rag_section.get("theory_improvement", {})
RAG_THEORY_IMPROVEMENT_PROVIDER = theory_improvement_config.get("provider") if theory_improvement_config else None
RAG_THEORY_IMPROVEMENT_MODEL = theory_improvement_config.get("model_name") if theory_improvement_config else None

# Inherit LLM configuration from main system
def get_main_llm_config():
    """
    Get LLM configuration from main config file.
    
    Uses JADE_AGENT_CONFIG environment variable if set, otherwise defaults to assignment_config.yaml.
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required config keys are missing
    """
    from src.config import load_config
    
    # Use JADE_AGENT_CONFIG if set (for evaluator mode), otherwise use assignment_config.yaml
    config_path = os.getenv("JADE_AGENT_CONFIG", "src/config/assignment_config.yaml")
    main_config = load_config(config_path)
    
    if not main_config:
        raise ValueError(f"Failed to load main config from {config_path}")
    
   # Required config keys
    return {
        "provider": main_config["provider"],
        "model_name": main_config["model_name"],
        "temperature": main_config["temperature"],
        "api_key": os.getenv("OPENAI_API_KEY")  # Optional
    }

# Load main system LLM configuration (required)
_main_llm_config = get_main_llm_config()

# Override RAG settings with main system LLM config
RAG_AI_PROVIDER = _main_llm_config["provider"]
model_name = _main_llm_config.get("model_name")
api_key = _main_llm_config.get("api_key")

# Set the appropriate model variable based on provider
if RAG_AI_PROVIDER in ("openai", "openai-compatible"):
    if model_name:
        RAG_OPENAI_MODEL = model_name
    if api_key:
        RAG_OPENAI_API_KEY = api_key
elif RAG_AI_PROVIDER in ("gemini", "google", "google-genai"):
    if model_name:
        RAG_GOOGLE_MODEL = model_name
elif RAG_AI_PROVIDER == "ollama":
    if model_name:
        RAG_MODEL_NAME = model_name

