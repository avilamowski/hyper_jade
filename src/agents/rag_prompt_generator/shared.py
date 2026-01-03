"""
Shared dependencies and utilities for the code_generator module.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import (
    RAG_AI_PROVIDER,
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_MODEL,
    RAG_OPENAI_BASE_URL,
    RAG_OLLAMA_HOST,
    RAG_OLLAMA_PORT,
    RAG_MODEL_NAME,
    RAG_TEMPERATURE_EXAMPLE_GENERATION,
    RAG_TEMPERATURE_THEORY_CORRECTION,
    RAG_TEMPERATURE_FILTERING,
    RAG_GOOGLE_API_KEY,
    RAG_GOOGLE_MODEL,
    RAG_THEORY_IMPROVEMENT_PROVIDER,
    RAG_THEORY_IMPROVEMENT_MODEL,
)

# Configure logging
logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for creating LLM instances with different configurations."""
    
    def __init__(self):
        self.ai_provider = RAG_AI_PROVIDER
        self.openai_api_key = RAG_OPENAI_API_KEY
        self.openai_model = RAG_OPENAI_MODEL
        self.openai_base_url = RAG_OPENAI_BASE_URL
        self.ollama_host = RAG_OLLAMA_HOST
        self.ollama_port = RAG_OLLAMA_PORT
        self.model_name = RAG_MODEL_NAME
        self.google_api_key = RAG_GOOGLE_API_KEY
        self.google_model = RAG_GOOGLE_MODEL
        self.theory_improvement_provider = RAG_THEORY_IMPROVEMENT_PROVIDER
        self.theory_improvement_model = RAG_THEORY_IMPROVEMENT_MODEL
    
    def create_llm(self, temperature: float = RAG_TEMPERATURE_EXAMPLE_GENERATION, 
                   provider: Optional[str] = None, model_name: Optional[str] = None):
        """Create an LLM instance with a specific temperature.
        
        Args:
            temperature: Temperature for the LLM
            provider: Optional provider override ("openai", "ollama", "gemini")
            model_name: Optional model name override
        """
        # Use provided provider or default to self.ai_provider
        effective_provider = provider or self.ai_provider
        
        if effective_provider in ("openai", "openai-compatible"):
            if not self.openai_api_key:
                raise Exception("OpenAI API key is required when using OpenAI provider")
            
            effective_model = model_name or self.openai_model
            return ChatOpenAI(
                model=effective_model,
                api_key=SecretStr(self.openai_api_key) if self.openai_api_key else None,
                base_url=self.openai_base_url,
                temperature=temperature
            )
        elif effective_provider in ("gemini", "google", "google-genai"):
            if not self.google_api_key:
                raise Exception("Google API key is required when using Gemini provider")
            
            effective_model = model_name or self.google_model
            return ChatGoogleGenerativeAI(
                model=effective_model,
                temperature=temperature,
                google_api_key=self.google_api_key
            )
        else:
            # Initialize Ollama LLM
            effective_model = model_name or self.model_name
            return OllamaLLM(
                model=effective_model,
                temperature=temperature,
                base_url=f"http://{self.ollama_host}:{self.ollama_port}"
            )
    
    def create_theory_improvement_llm(self, temperature: float = RAG_TEMPERATURE_THEORY_CORRECTION):
        """Create an LLM instance specifically for theory improvement.
        
        Uses the configured theory improvement provider and model, or falls back
        to the default provider/model if not configured.
        """
        # If no specific model is configured, use None to let create_llm use defaults
        effective_model = self.theory_improvement_model if self.theory_improvement_model else None
        return self.create_llm(
            temperature=temperature,
            provider=self.theory_improvement_provider,
            model_name=effective_model
        )
    
    def create_ragas_llm(self):
        """Create an LLM instance for Ragas metrics."""
        if self.ai_provider == "openai":
            from ragas.llms.base import llm_factory
            return llm_factory('gpt-4o-mini')
        else:
            from ragas.llms.base import llm_factory
            return llm_factory(self.model_name)


class XMLParser:
    """Utility class for parsing XML responses from LLMs."""
    
    @staticmethod
    def parse_xml_response(response_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Parse XML response to extract code examples."""
        import re
        
        try:
            examples = []
            
            # Clean the response text - remove any "json" prefix or other text
            cleaned_text = response_text.strip()
            if cleaned_text.lower().startswith('json'):
                # Find the first '<' or '[' after "json"
                for i, char in enumerate(cleaned_text):
                    if char in '<[':
                        cleaned_text = cleaned_text[i:]
                        break
            
            # Try to find XML-like structure
            # Look for patterns like <example>, <code>, <approach>, etc.
            example_pattern = r'<example[^>]*>(.*?)</example>'
            code_pattern = r'<code[^>]*>(.*?)</code>'
            approach_pattern = r'<approach[^>]*>(.*?)</approach>'
            description_pattern = r'<description[^>]*>(.*?)</description>'
            improvements_pattern = r'<improvements[^>]*>(.*?)</improvements>'
            theory_alignment_pattern = r'<theory_alignment[^>]*>(.*?)</theory_alignment>'
            
            # Find all example blocks
            example_matches = re.findall(example_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            
            if example_matches:
                for i, example_content in enumerate(example_matches[:num_examples]):
                    # Extract code
                    code_match = re.search(code_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    code = code_match.group(1).strip() if code_match else "# No code found"
                    
                    # Extract approach
                    approach_match = re.search(approach_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    approach = approach_match.group(1).strip() if approach_match else "No approach provided"
                    
                    # Extract description
                    desc_match = re.search(description_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    description = desc_match.group(1).strip() if desc_match else f"Example {i+1}"
                    
                    # Extract improvements (for improved examples)
                    improvements_match = re.search(improvements_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    improvements = improvements_match.group(1).strip() if improvements_match else None
                    
                    # Extract theory alignment (for improved examples)
                    theory_match = re.search(theory_alignment_pattern, example_content, re.DOTALL | re.IGNORECASE)
                    theory_alignment = theory_match.group(1).strip() if theory_match else None
                    
                    example_dict = {
                        "example_id": i + 1,
                        "description": description,
                        "code": code,
                        "approach": approach
                    }
                    
                    # Add optional fields if they exist
                    if improvements:
                        example_dict["improvements"] = improvements.split('\n') if '\n' in improvements else [improvements]
                    if theory_alignment:
                        example_dict["theory_alignment"] = theory_alignment
                    
                    examples.append(example_dict)
            else:
                # If no XML structure found, try to extract code blocks from the text
                code_blocks = re.findall(r'```python\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                if not code_blocks:
                    code_blocks = re.findall(r'```\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                
                for i, code in enumerate(code_blocks[:num_examples]):
                    examples.append({
                        "example_id": i + 1,
                        "description": f"Example {i+1}",
                        "code": code.strip(),
                        "approach": "Extracted from response"
                    })
            
            # If still no examples, create a basic one
            if not examples:
                examples.append({
                    "example_id": 1,
                    "description": "Generated example based on requirement",
                    "code": "# Example code\nprint('Hello, World!')",
                    "approach": "Basic implementation"
                })
            
            logger.info(f"Parsed {len(examples)} examples using XML fallback")
            return examples
            
        except Exception as e:
            logger.error(f"Error in XML fallback parsing: {e}")
            # Ultimate fallback
            return [{
                "example_id": 1,
                "description": "Generated example based on requirement",
                "code": "# Example code\nprint('Hello, World!')",
                "approach": "Basic implementation"
            }]
    
    @staticmethod
    def parse_filtered_xml_response(response_text: str, original_example: Dict[str, Any]) -> Dict[str, Any]:
        """Parse XML response to extract filtered example."""
        import re
        
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Look for analysis and filtered_example patterns
            analysis_pattern = r'<analysis[^>]*>(.*?)</analysis>'
            filtered_example_pattern = r'<filtered_example[^>]*>(.*?)</filtered_example>'
            
            # Extract analysis section
            analysis_match = re.search(analysis_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            analysis_content = analysis_match.group(1) if analysis_match else ""
            
            # Extract filtered example section
            filtered_match = re.search(filtered_example_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
            filtered_content = filtered_match.group(1) if filtered_match else ""
            
            # Parse analysis elements
            has_irrelevant_pattern = r'<has_irrelevant_elements[^>]*>(.*?)</has_irrelevant_elements>'
            irrelevant_elements_pattern = r'<irrelevant_elements[^>]*>(.*?)</irrelevant_elements>'
            justification_pattern = r'<filtering_justification[^>]*>(.*?)</filtering_justification>'
            
            has_irrelevant = re.search(has_irrelevant_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            irrelevant_elements = re.search(irrelevant_elements_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            justification = re.search(justification_pattern, analysis_content, re.DOTALL | re.IGNORECASE)
            
            # Parse filtered example elements
            code_pattern = r'<code[^>]*>(.*?)</code>'
            approach_pattern = r'<approach[^>]*>(.*?)</approach>'
            summary_pattern = r'<filtering_summary[^>]*>(.*?)</filtering_summary>'
            
            filtered_code = re.search(code_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            filtered_approach = re.search(approach_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            filtering_summary = re.search(summary_pattern, filtered_content, re.DOTALL | re.IGNORECASE)
            
            # Create filtered example - preserve ALL original fields first
            filtered_example = original_example.copy()
            
            # Preserve important fields from improved examples
            # These should not be lost during filtering
            preserved_fields = ['class_name', 'improvements', 'theory_alignment', 'original_code', 'original_approach']
            for field in preserved_fields:
                if field in original_example:
                    filtered_example[field] = original_example[field]
            
            # Update code if filtered version was provided
            if filtered_code and filtered_code.group(1).strip():
                filtered_example["code"] = filtered_code.group(1).strip()
                filtered_example["was_filtered"] = True
            else:
                filtered_example["was_filtered"] = False
                # If no filtered code, keep original code
                if "code" not in filtered_example:
                    filtered_example["code"] = original_example.get("code", "")
            
            # Update approach if provided
            if filtered_approach and filtered_approach.group(1).strip():
                filtered_example["approach"] = filtered_approach.group(1).strip()
            # If no filtered approach, preserve original
            elif "approach" not in filtered_example:
                filtered_example["approach"] = original_example.get("approach", "")
            
            # Ensure description is preserved
            if "description" not in filtered_example:
                filtered_example["description"] = original_example.get("description", "Example")
            
            # Add filtering metadata
            filtered_example["has_irrelevant_elements"] = has_irrelevant.group(1).strip().lower() == "true" if has_irrelevant else False
            filtered_example["irrelevant_elements"] = irrelevant_elements.group(1).strip() if irrelevant_elements else ""
            filtered_example["filtering_justification"] = justification.group(1).strip() if justification else ""
            filtered_example["filtering_summary"] = filtering_summary.group(1).strip() if filtering_summary else "No filtering applied"
            
            return filtered_example
            
        except Exception as e:
            logger.error(f"Error parsing filtered XML response: {e}")
            # Return original example with error metadata
            original_example["filtering_error"] = f"XML parsing error: {e}"
            original_example["was_filtered"] = False
            original_example["filtering_summary"] = "No filtering applied due to parsing error"
            return original_example
