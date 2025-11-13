"""
Configuration utilities for the assignment evaluation system
"""

from typing import Dict, Any, Optional
import yaml
import os
from dotenv import load_dotenv, dotenv_values

def get_agent_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    Get agent-specific configuration with fallback to global configuration
    
    Args:
        config: The complete configuration dictionary
        agent_name: Name of the agent (e.g., 'requirement_generator', 'prompt_generator', 'code_corrector')
        
    Returns:
        Dictionary with agent-specific configuration, falling back to global config
    """
    # Get agent-specific configuration
    agent_config = config.get('agents', {}).get(agent_name, {})
    
    # Create merged configuration with global fallbacks
    merged_config = {
        # Global configuration as fallback
        'model_name': config.get('model_name', 'gpt-4'),
        'provider': config.get('provider', 'openai'),
        'temperature': config.get('temperature', 0.1),
        
        # Agent-specific overrides
        'model_name': agent_config.get('model_name', config.get('model_name', 'gpt-4')),
        'provider': agent_config.get('provider', config.get('provider', 'openai')),
        'temperature': agent_config.get('temperature', config.get('temperature', 0.1)),
        
        # Include all other agent-specific settings
        **agent_config
    }
    
    return merged_config


def get_global_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get global configuration settings
    
    Args:
        config: The complete configuration dictionary
        
    Returns:
        Dictionary with global configuration
    """
    return {
        'model_name': config.get('model_name', 'gpt-4'),
        'provider': config.get('provider', 'openai'),
        'temperature': config.get('temperature', 0.1),
        'enable_rag': config.get('enable_rag', False),
        'rag_knowledge_base': config.get('rag_knowledge_base'),
        'output': config.get('output', {}),
        'logging': config.get('logging', {})
    }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}
    
    
def load_langsmith_config():
    """Load LangSmith config from config/langsmith_config.yaml and .env"""
    config_path = os.path.join(os.path.dirname(__file__), "langsmith_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading langsmith config: {e}")
        config = {}
    load_dotenv(override=True)
    DOTENV = dotenv_values()
    langsmith = config.get("langsmith", {})
    if langsmith.get("enable"):
        print("🔧 LangSmith integration enabled")
        os.environ["LANGSMITH_TRACING"] = langsmith.get("tracing", "")
        if langsmith.get("endpoint"):
            os.environ["LANGSMITH_ENDPOINT"] = langsmith["endpoint"]
        if DOTENV.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = DOTENV["LANGSMITH_API_KEY"]
        if DOTENV.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = DOTENV["LANGSMITH_PROJECT"]