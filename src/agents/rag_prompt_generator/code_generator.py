"""
Code Generator for RAG-Enhanced Prompt Generation

Generates code examples using course theory retrieved from notebooks.
Uses the same implementation as JADE_RAG with Jinja2 templates.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from .shared import LLMFactory, XMLParser
from .prompt_templates import PromptTemplates, Language
from .config import (
    RAG_TEMPERATURE_EXAMPLE_GENERATION,
    RAG_TEMPERATURE_THEORY_CORRECTION,
    RAG_TEMPERATURE_FILTERING,
    RAG_ENABLE_FILTERING,
)

# Import LangSmith for tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # Create a no-op decorator if LangSmith is not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Handles code generation using LLM with Jinja2 templates."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
        self.llm = None
    
    async def initialize(self):
        """Initialize the code generator."""
        self.llm = self.llm_factory.create_llm(temperature=RAG_TEMPERATURE_EXAMPLE_GENERATION)
    
    @traceable(name="generate_initial_examples")
    async def generate_examples(self, requirement: str, num_examples: int = 3, language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
        """Generate code examples for a given requirement."""
        try:
            # Get language-specific prompt
            prompt = self.prompt_templates.format_template(
                language, 
                "code_generation", 
                requirement=requirement, 
                num_examples=num_examples
            )
            
            # Get language-specific system message
            system_message = self.prompt_templates.get_system_message(language, "code_generation")
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types (OpenAI has .content, Ollama returns string directly)
            if hasattr(response, 'content'):
                response_text = getattr(response, 'content', '').strip()
            else:
                response_text = str(response).strip()
            
            # Parse XML response
            examples = self.xml_parser.parse_xml_response(response_text, num_examples)
            
            logger.info(f"Generated {len(examples)} examples for requirement: {requirement[:50]}...")
            return examples
            
        except Exception as e:
            logger.error(f"Error generating examples: {e}")
            return []


class TheoryImprover:
    """Handles improvement of code examples using theoretical content."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
    
    @traceable(name="retrieve_course_theory")
    async def get_relevant_theory(self, rag_system, requirement: str, examples: Optional[List[Dict[str, Any]]] = None, max_results: int = 5, max_class_number: Optional[int] = None, dataset: str = "python") -> List[Dict[str, Any]]:
        """Get relevant theoretical content from notebooks using RAG."""
        try:
            # Create a comprehensive query that includes both requirement and examples
            query_parts = [requirement]
            
            if examples:
                # Add code snippets from examples to the query
                for example in examples:
                    if "code" in example and example["code"]:
                        # Extract key concepts from code (remove comments and common keywords)
                        code_snippet = example["code"][:500]  # Limit length
                        query_parts.append(f"código ejemplo: {code_snippet}")
            
            # Combine all query parts
            combined_query = " ".join(query_parts)
            
            # Retrieve relevant documents from vector database (no LLM processing)
            theory_sources = await rag_system.retrieve_documents(combined_query, max_results=max_results, max_class_number=max_class_number, dataset=dataset)
            
            return theory_sources
                
        except Exception as e:
            logger.error(f"Error getting relevant theory: {e}")
            return []
    
    @traceable(name="improve_examples_with_theory")
    async def improve_examples_with_theory(self, examples: List[Dict[str, Any]], 
                                         requirement: str, 
                                         theory_sources: List[Dict[str, Any]],
                                         language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
        """Improve code examples using theoretical content."""
        try:
            if not theory_sources:
                logger.warning("No theory sources provided, returning original examples")
                return examples
            
            # Combine theory sources into context
            theory_context = "\n\n".join([
                f"**{source.get('class_name', 'Unknown')}:**\n{source.get('content', '')}"
                for source in theory_sources
            ])
            
            improved_examples = []
            
            for example in examples:
                # Get language-specific prompt
                prompt = self.prompt_templates.format_template(
                    language,
                    "theory_improvement",
                    requirement=requirement,
                    theory_context=theory_context,
                    description=example['description'],
                    code=example['code'],
                    approach=example['approach']
                )
                
                # Get language-specific system message
                system_message = self.prompt_templates.get_system_message(language, "theory_improvement")

                # Create LLM instance with lower temperature for theory-based correction
                theory_llm = self.llm_factory.create_llm(temperature=RAG_TEMPERATURE_THEORY_CORRECTION)
                    
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=prompt)
                ]
                
                response = theory_llm.invoke(messages)
                
                # Handle different response types (OpenAI has .content, Ollama returns string directly)
                if hasattr(response, 'content'):
                    response_text = getattr(response, 'content', '').strip()
                else:
                    response_text = str(response).strip()
                
                try:
                    # Parse XML response for improved example
                    xml_parsed = self.xml_parser.parse_xml_response(response_text, 1)
                    if xml_parsed:
                        improved_example = xml_parsed[0]
                        improved_example['original_code'] = example['code']
                        improved_example['original_approach'] = example['approach']
                        # Add class name from the most relevant theory source
                        if theory_sources:
                            improved_example['class_name'] = theory_sources[0].get('class_name', 'Unknown')
                        improved_examples.append(improved_example)
                    else:
                        # Fallback to original example
                        example['class_name'] = theory_sources[0].get('class_name', 'Unknown') if theory_sources else 'Unknown'
                        improved_examples.append(example)
                except Exception as parse_error:
                    logger.warning(f"Error parsing improved example: {parse_error}")
                    example['class_name'] = theory_sources[0].get('class_name', 'Unknown') if theory_sources else 'Unknown'
                    improved_examples.append(example)
            
            logger.info(f"Improved {len(improved_examples)} examples with theory")
            return improved_examples
            
        except Exception as e:
            logger.error(f"Error improving examples with theory: {e}")
            return examples


class TheoryFilter:
    """Handles filtering of theory-specific elements from improved examples."""
    
    def __init__(self, llm_factory: LLMFactory):
        self.llm_factory = llm_factory
        self.xml_parser = XMLParser()
        self.prompt_templates = PromptTemplates()
    
    @traceable(name="filter_theory_specific_elements")
    async def filter_theory_specific_elements(self, examples: List[Dict[str, Any]], 
                                            requirement: str, 
                                            theory_context: str,
                                            language: Language = Language.PYTHON) -> List[Dict[str, Any]]:
        """Filter theory-specific elements that may not be relevant to the requirement."""
        try:
            if not RAG_ENABLE_FILTERING:
                logger.info("Filtering disabled, returning original examples")
                return examples
            
            filtered_examples = []
            
            for example in examples:
                # Get language-specific prompt
                prompt = self.prompt_templates.format_template(
                    language,
                    "filtering",
                    requirement=requirement,
                    theory_context=theory_context,
                    original_code=example.get('original_code', example['code']),
                    improved_code=example['code'],
                    description=example['description'],
                    approach=example.get('original_approach', example['approach']),
                    improvements=example.get('improvements', []),
                    theory_alignment=example.get('theory_alignment', '')
                )
                
                # Get language-specific system message
                system_message = self.prompt_templates.get_system_message(language, "filtering")

                # Create LLM instance with low temperature for filtering
                filter_llm = self.llm_factory.create_llm(temperature=RAG_TEMPERATURE_FILTERING)
                    
                messages = [
                    SystemMessage(content=system_message),
                    HumanMessage(content=prompt)
                ]
                
                response = filter_llm.invoke(messages)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    response_text = getattr(response, 'content', '').strip()
                else:
                    response_text = str(response).strip()
                
                try:
                    # Parse filtered XML response
                    filtered_example = self.xml_parser.parse_filtered_xml_response(response_text, example)
                    filtered_examples.append(filtered_example)
                except Exception as parse_error:
                    logger.warning(f"Error parsing filtered example: {parse_error}")
                    filtered_examples.append(example)
            
            logger.info(f"Filtered {len(filtered_examples)} examples")
            return filtered_examples
            
        except Exception as e:
            logger.error(f"Error filtering examples: {e}")
            return examples


class CodeExampleGenerator:
    """Main class that orchestrates code example generation with RAG enhancement."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.llm_factory = LLMFactory()
        self.code_generator = CodeGenerator(self.llm_factory)
        self.theory_improver = TheoryImprover(self.llm_factory)
        self.theory_filter = TheoryFilter(self.llm_factory)
    
    async def initialize(self):
        """Initialize all components."""
        await self.code_generator.initialize()
    
    @traceable(name="generate_enhanced_examples")
    async def generate_enhanced_examples(self, requirement: str, num_examples: int = 3, 
                                       max_theory_results: int = 5, 
                                       max_class_number: Optional[int] = None,
                                       dataset: str = "python") -> List[Dict[str, Any]]:
        """Generate code examples enhanced with course theory."""
        try:
            logger.info(f"Generating enhanced examples for: {requirement[:50]}...")
            
            # Step 1: Generate initial examples
            examples = await self.code_generator.generate_examples(requirement, num_examples)
            
            if not examples:
                logger.warning("No examples generated, returning empty list")
                return []
            
            # Step 2: Get relevant theory
            theory_sources = await self.theory_improver.get_relevant_theory(
                self.rag_system, requirement, examples, max_theory_results, max_class_number, dataset
            )
            
            if not theory_sources:
                logger.warning("No theory sources found, returning original examples")
                return examples
            
            # Step 3: Improve examples with theory
            improved_examples = await self.theory_improver.improve_examples_with_theory(
                examples, requirement, theory_sources
            )
            
            # Step 4: Filter theory-specific elements
            theory_context = "\n\n".join([
                f"**{source.get('class_name', 'Unknown')}:**\n{source.get('content', '')}"
                for source in theory_sources
            ])
            
            filtered_examples = await self.theory_filter.filter_theory_specific_elements(
                improved_examples, requirement, theory_context
            )
            
            logger.info(f"Generated {len(filtered_examples)} enhanced examples")
            return filtered_examples
            
        except Exception as e:
            logger.error(f"Error generating enhanced examples: {e}")
            # Fallback to basic examples
            return await self.code_generator.generate_examples(requirement, num_examples)