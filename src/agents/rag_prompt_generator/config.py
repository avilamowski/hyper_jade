"""
RAG Prompt Generator Configuration

Independent configuration for RAG functionality that doesn't interfere
with the existing hyper_jade configuration system.
"""

import os
from typing import Optional

# RAG System Configuration
WEAVIATE_URL = os.getenv("RAG_WEAVIATE_URL", "http://localhost:8080")
RAG_AI_PROVIDER = os.getenv("RAG_AI_PROVIDER", "ollama")  # "openai" or "ollama"

# Ollama Configuration for RAG
RAG_OLLAMA_HOST = os.getenv("RAG_OLLAMA_HOST", "localhost")
RAG_OLLAMA_PORT = int(os.getenv("RAG_OLLAMA_PORT", "11434"))
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "qwen2.5:7b")

# OpenAI Configuration for RAG - uses same API key as main system
RAG_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Use same key as main system
RAG_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-3.5-turbo")
RAG_OPENAI_BASE_URL = os.getenv("RAG_OPENAI_BASE_URL")

# Google/Gemini Configuration for RAG
RAG_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RAG_GOOGLE_MODEL = os.getenv("RAG_GOOGLE_MODEL", "gemini-pro")
RAG_THEORY_IMPROVEMENT_PROVIDER = os.getenv("RAG_THEORY_IMPROVEMENT_PROVIDER", "openai")  # "openai", "gemini", or "ollama"
RAG_THEORY_IMPROVEMENT_MODEL = os.getenv("RAG_THEORY_IMPROVEMENT_MODEL")  # If None, uses RAG_GOOGLE_MODEL or RAG_OPENAI_MODEL

# Text Splitter Configuration (will be overridden by YAML config)
RAG_TEXT_SPLITTER_STRATEGY = os.getenv("RAG_TEXT_SPLITTER_STRATEGY", "cell_based")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
RAG_MIN_CHUNK_SIZE = int(os.getenv("RAG_MIN_CHUNK_SIZE", "100"))

# Reranking Configuration (will be overridden by YAML config)
RAG_ENABLE_RERANKING = os.getenv("RAG_ENABLE_RERANKING", "false").lower() == "true"
RAG_RERANKING_MODEL = os.getenv("RAG_RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RAG_INITIAL_RETRIEVAL_COUNT = int(os.getenv("RAG_INITIAL_RETRIEVAL_COUNT", "15"))
RAG_FINAL_RETRIEVAL_COUNT = int(os.getenv("RAG_FINAL_RETRIEVAL_COUNT", "5"))

# Temperature Configuration (will be overridden by YAML config)
RAG_TEMPERATURE_EXAMPLE_GENERATION = float(os.getenv("RAG_TEMPERATURE_EXAMPLE_GENERATION", "0.7"))
RAG_TEMPERATURE_THEORY_CORRECTION = float(os.getenv("RAG_TEMPERATURE_THEORY_CORRECTION", "0.3"))
RAG_TEMPERATURE_FILTERING = float(os.getenv("RAG_TEMPERATURE_FILTERING", "0.1"))

# Dataset Configuration (will be overridden by YAML config)
RAG_PYTHON_NOTEBOOKS_DIR = os.getenv("RAG_PYTHON_NOTEBOOKS_DIR", "data/Clases")
RAG_HASKELL_NOTEBOOKS_DIR = os.getenv("RAG_HASKELL_NOTEBOOKS_DIR", "data/learnyouahaskell")

# Debug Configuration (will be overridden by YAML config)
RAG_DEBUG_MODE = os.getenv("RAG_DEBUG_MODE", "false").lower() == "true"
RAG_ENABLE_FILTERING = os.getenv("RAG_ENABLE_FILTERING", "true").lower() == "true"

# LangSmith Configuration (will be overridden by YAML config)
RAG_LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
RAG_LANGSMITH_PROJECT = os.getenv("RAG_LANGSMITH_PROJECT", "hyper-jade-rag")
RAG_LANGSMITH_ENDPOINT = os.getenv("RAG_LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
RAG_LANGSMITH_TRACING = os.getenv("RAG_LANGSMITH_TRACING", "false").lower() == "true"
RAG_ENABLE_LANGSMITH_TRACING = os.getenv("RAG_ENABLE_LANGSMITH_TRACING", "false").lower() == "true"

# Load RAG configuration from YAML file
def load_rag_config():
    """Load RAG configuration from YAML file"""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent.parent / "config" / "rag_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Could not load RAG config from {config_path}: {e}")
        return {"use_rag": False}

# Load configuration
_rag_config = load_rag_config()

# Main Control Flag - from YAML config only
USE_RAG = _rag_config.get("use_rag", False)

# Override config values from YAML if available
if "rag" in _rag_config:
    rag_section = _rag_config["rag"]
    
    # Override debug settings
    if "debug" in rag_section:
        debug_config = rag_section["debug"]
        RAG_DEBUG_MODE = debug_config.get("enabled", RAG_DEBUG_MODE)
        RAG_ENABLE_FILTERING = debug_config.get("filtering_enabled", RAG_ENABLE_FILTERING)
    
    # Override dataset paths
    if "datasets" in rag_section:
        datasets = rag_section["datasets"]
        if "python" in datasets:
            RAG_PYTHON_NOTEBOOKS_DIR = datasets["python"].get("notebooks_dir", RAG_PYTHON_NOTEBOOKS_DIR)
        if "haskell" in datasets:
            RAG_HASKELL_NOTEBOOKS_DIR = datasets["haskell"].get("notebooks_dir", RAG_HASKELL_NOTEBOOKS_DIR)
    
    # Override reranking settings
    if "reranking" in rag_section:
        reranking = rag_section["reranking"]
        RAG_ENABLE_RERANKING = reranking.get("enabled", RAG_ENABLE_RERANKING)
        RAG_RERANKING_MODEL = reranking.get("model", RAG_RERANKING_MODEL)
        RAG_INITIAL_RETRIEVAL_COUNT = reranking.get("initial_retrieval_count", RAG_INITIAL_RETRIEVAL_COUNT)
        RAG_FINAL_RETRIEVAL_COUNT = reranking.get("final_retrieval_count", RAG_FINAL_RETRIEVAL_COUNT)
    
    # Override temperature settings
    if "temperatures" in rag_section:
        temps = rag_section["temperatures"]
        RAG_TEMPERATURE_EXAMPLE_GENERATION = temps.get("example_generation", RAG_TEMPERATURE_EXAMPLE_GENERATION)
        RAG_TEMPERATURE_THEORY_CORRECTION = temps.get("theory_correction", RAG_TEMPERATURE_THEORY_CORRECTION)
        RAG_TEMPERATURE_FILTERING = temps.get("filtering", RAG_TEMPERATURE_FILTERING)

# Inherit LLM configuration from main system
def get_main_llm_config():
    """Get LLM configuration from main assignment_config.yaml"""
    try:
        from src.config import load_config
        main_config = load_config("src/config/assignment_config.yaml")
        return {
            "provider": main_config.get("provider", "openai"),
            "model_name": main_config.get("model_name", "gpt-4o-mini"),
            "temperature": main_config.get("temperature", 0.1),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    except Exception as e:
        logger.warning(f"Could not load main LLM config: {e}")
        return {
            "provider": "openai",
            "model_name": "gpt-4o-mini", 
            "temperature": 0.1,
            "api_key": os.getenv("OPENAI_API_KEY")
        }

# Load main system LLM configuration
_main_llm_config = get_main_llm_config()

# Override RAG settings with main system LLM config
RAG_AI_PROVIDER = _main_llm_config["provider"]
RAG_OPENAI_MODEL = _main_llm_config["model_name"]
RAG_OPENAI_API_KEY = _main_llm_config["api_key"]

# Temperature configurations for different RAG operations
RAG_TEMPERATURE_EXAMPLE_GENERATION = _rag_config.get("rag", {}).get("temperatures", {}).get("example_generation", 0.7)
RAG_TEMPERATURE_THEORY_CORRECTION = _rag_config.get("rag", {}).get("temperatures", {}).get("theory_correction", 0.3)
RAG_TEMPERATURE_FILTERING = _rag_config.get("rag", {}).get("temperatures", {}).get("filtering", 0.1)

# Override configurations with YAML values
RAG_TEXT_SPLITTER_STRATEGY = _rag_config.get("rag", {}).get("text_splitter", {}).get("strategy", RAG_TEXT_SPLITTER_STRATEGY)
RAG_CHUNK_SIZE = _rag_config.get("rag", {}).get("text_splitter", {}).get("chunk_size", RAG_CHUNK_SIZE)
RAG_CHUNK_OVERLAP = _rag_config.get("rag", {}).get("text_splitter", {}).get("chunk_overlap", RAG_CHUNK_OVERLAP)
RAG_MIN_CHUNK_SIZE = _rag_config.get("rag", {}).get("text_splitter", {}).get("min_chunk_size", RAG_MIN_CHUNK_SIZE)

RAG_ENABLE_RERANKING = _rag_config.get("rag", {}).get("reranking", {}).get("enabled", RAG_ENABLE_RERANKING)
RAG_RERANKING_MODEL = _rag_config.get("rag", {}).get("reranking", {}).get("model", RAG_RERANKING_MODEL)
RAG_INITIAL_RETRIEVAL_COUNT = _rag_config.get("rag", {}).get("reranking", {}).get("initial_retrieval_count", RAG_INITIAL_RETRIEVAL_COUNT)
RAG_FINAL_RETRIEVAL_COUNT = _rag_config.get("rag", {}).get("reranking", {}).get("final_retrieval_count", RAG_FINAL_RETRIEVAL_COUNT)

RAG_PYTHON_NOTEBOOKS_DIR = _rag_config.get("rag", {}).get("datasets", {}).get("python", {}).get("notebooks_dir", RAG_PYTHON_NOTEBOOKS_DIR)
RAG_HASKELL_NOTEBOOKS_DIR = _rag_config.get("rag", {}).get("datasets", {}).get("haskell", {}).get("notebooks_dir", RAG_HASKELL_NOTEBOOKS_DIR)

RAG_DEBUG_MODE = _rag_config.get("rag", {}).get("debug", {}).get("enabled", RAG_DEBUG_MODE)
RAG_ENABLE_FILTERING = _rag_config.get("rag", {}).get("debug", {}).get("filtering_enabled", RAG_ENABLE_FILTERING)

RAG_LANGSMITH_PROJECT = _rag_config.get("rag", {}).get("langsmith", {}).get("project", RAG_LANGSMITH_PROJECT)
RAG_LANGSMITH_ENDPOINT = _rag_config.get("rag", {}).get("langsmith", {}).get("endpoint", RAG_LANGSMITH_ENDPOINT)
RAG_LANGSMITH_TRACING = _rag_config.get("rag", {}).get("langsmith", {}).get("enabled", RAG_LANGSMITH_TRACING)
RAG_ENABLE_LANGSMITH_TRACING = _rag_config.get("rag", {}).get("langsmith", {}).get("enabled", RAG_ENABLE_LANGSMITH_TRACING)

# Theory Improvement Model Configuration (from YAML)
theory_improvement_config = _rag_config.get("rag", {}).get("theory_improvement", {})
if theory_improvement_config:
    RAG_THEORY_IMPROVEMENT_PROVIDER = theory_improvement_config.get("provider", RAG_THEORY_IMPROVEMENT_PROVIDER)
    RAG_THEORY_IMPROVEMENT_MODEL = theory_improvement_config.get("model_name", RAG_THEORY_IMPROVEMENT_MODEL)
