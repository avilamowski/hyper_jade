"""
Configuration utilities for the assignment evaluation system
"""

from typing import Dict, Any, Optional


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
        'logging': config.get('logging', {}),
        'languages': config.get('languages', {})
    }
