#!/usr/bin/env python3
"""
Prompt Generator Agent

This agent generates specialized correction prompts for each rubric item.
It uses RAG (Retrieval Augmented Generation) to enhance prompts with relevant
knowledge and examples.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
import logging
import json
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

from ..requirement_generator.requirement_generator import Rubric, RubricItem

logger = logging.getLogger(__name__)

@dataclass
class CorrectionPrompt:
    """Represents a correction prompt for a specific rubric item"""
    rubric_item_id: str
    rubric_item_title: str
    prompt: str
    criteria: List[str]
    examples: Optional[List[str]] = None
    resources: Optional[List[str]] = None

@dataclass
class PromptSet:
    """Complete set of correction prompts for an assignment"""
    assignment_description: str
    programming_language: str
    prompts: List[CorrectionPrompt]
    general_prompt: str

class PromptGeneratorState(TypedDict):
    """State for prompt generation process"""
    assignment_description: str
    rubric: Optional[Any]
    programming_language: str
    generated_prompts: List[CorrectionPrompt]
    general_prompt: str
    rag_knowledge: List[str]

class PromptGeneratorAgent:
    """Agent that generates correction prompts for each rubric item"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._setup_llm()
        self.graph = self._create_graph()
        self.rag_enabled = config.get("enable_rag", False)
    
    def _setup_llm(self):
        """Setup LLM based on configuration"""
        if self.config.get("provider") == "openai":
            return ChatOpenAI(
                model=self.config.get("model_name", "gpt-4"),
                temperature=self.config.get("temperature", 0.1)
            )
        else:
            return OllamaLLM(
                model=self.config.get("model_name", "qwen2.5:7b"),
                temperature=self.config.get("temperature", 0.1)
            )
    
    def _create_graph(self) -> Any:
        """Create the LangGraph workflow for prompt generation"""
        graph = StateGraph(PromptGeneratorState)
        
        # Add nodes
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("generate_prompts", self._generate_prompts)
        graph.add_node("generate_general_prompt", self._generate_general_prompt)
        
        # Set entry point
        graph.set_entry_point("retrieve_knowledge")
        
        # Add edges
        graph.add_edge("retrieve_knowledge", "generate_prompts")
        graph.add_edge("generate_prompts", "generate_general_prompt")
        graph.add_edge("generate_general_prompt", END)
        
        return graph.compile()
    
    def _retrieve_knowledge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge for prompt generation"""
        logger.info("Retrieving knowledge for prompt generation")
        
        if not self.rag_enabled:
            state["rag_knowledge"] = []
            return state
        
        # This would integrate with a knowledge base
        # For now, we'll use some basic knowledge
        knowledge_base = {
            "python": [
                "Python best practices include clear variable names, proper indentation, and docstrings",
                "Common beginner mistakes: forgetting colons, incorrect indentation, undefined variables",
                "Error handling should include try-except blocks and input validation",
                "Functions should have clear docstrings explaining parameters and return values"
            ],
            "javascript": [
                "JavaScript best practices include using const/let, arrow functions, and proper scoping",
                "Common beginner mistakes: undefined variables, hoisting issues, async/await misuse",
                "Error handling should include try-catch blocks and null checks",
                "Functions should have JSDoc comments explaining parameters and return values"
            ]
        }
        
        state["rag_knowledge"] = knowledge_base.get(state["programming_language"], [])
        
        return state
    
    def _generate_prompts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate correction prompts for each rubric item"""
        logger.info("Generating correction prompts for rubric items")
        
        if not state["rubric"]:
            logger.error("No rubric provided for prompt generation")
            return state
        
        prompts = []
        
        for item in state["rubric"].items:
            prompt = self._generate_item_prompt(item, state)
            prompts.append(prompt)
        
        state["generated_prompts"] = prompts
        
        return state
    
    def _generate_item_prompt(self, item: RubricItem, state: Dict[str, Any]) -> CorrectionPrompt:
        """Generate a correction prompt for a specific rubric item"""
        
        knowledge_context = ""
        if state["rag_knowledge"]:
            knowledge_context = f"\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {k}" for k in state["rag_knowledge"])
        
        prompt_text = f"""You are a specialized code correction agent for {state["programming_language"]} programming.

ASSIGNMENT DESCRIPTION:
{state["assignment_description"]}

RUBRIC ITEM: {item.title}
DESCRIPTION: {item.description}

CRITERIA TO EVALUATE:
{chr(10).join(f"- {criterion}" for criterion in item.criteria)}

{knowledge_context}

TASK: Create a detailed prompt for evaluating student code against the criteria above.

INSTRUCTIONS:
1. Create a prompt that will guide the evaluation of student code
2. Focus on identifying errors and issues related to each criterion
3. Request specific feedback for each error found
4. Ask for specific improvements where issues are identified
5. Focus on error identification rather than scoring

OUTPUT FORMAT:
<EVALUATION>
  <ERRORS>
    <ERROR type="error_type" line="line_number">
      <LOCATION>Where the error occurs</LOCATION>
      <DESCRIPTION>Detailed description of the error</DESCRIPTION>
      <SUGGESTION>How to fix this error</SUGGESTION>
    </ERROR>
  </ERRORS>
  
  <ASSESSMENT>
    <IS_PASSING>Yes/No</IS_PASSING>
    <OVERALL_FEEDBACK>Overall assessment of this aspect</OVERALL_FEEDBACK>
  </ASSESSMENT>
</EVALUATION>

This prompt will be used by the code correction agent to evaluate student code.
"""
        
        return CorrectionPrompt(
            rubric_item_id=item.id,
            rubric_item_title=item.title,
            prompt=prompt_text,
            criteria=item.criteria,
            examples=self._get_examples_for_item(item),
            resources=self._get_resources_for_item(item)
        )
    
    def _generate_general_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a general prompt for overall evaluation"""
        logger.info("Generating general evaluation prompt")
        
        knowledge_context = ""
        if state["rag_knowledge"]:
            knowledge_context = f"\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {k}" for k in state["rag_knowledge"])
        
        general_prompt = f"""You are a comprehensive code evaluation agent for {state["programming_language"]} programming.

ASSIGNMENT DESCRIPTION:
{state["assignment_description"]}

{knowledge_context}

TASK: Create a comprehensive prompt for overall evaluation of student code, focusing on error identification and improvement suggestions.

INSTRUCTIONS:
1. Create a prompt that will guide the analysis of code for overall correctness and functionality
2. Focus on identifying critical errors that prevent the code from working
3. Request assessment of code quality and structure
4. Ask for evaluation of error handling and edge case management
5. Request constructive feedback for improvement

OUTPUT FORMAT:
<COMPREHENSIVE_EVALUATION>
  <OVERALL_ASSESSMENT>
    <CORRECTNESS>Assessment of functional correctness and errors</CORRECTNESS>
    <QUALITY>Assessment of code quality issues</QUALITY>
    <ERROR_HANDLING>Assessment of error handling and edge cases</ERROR_HANDLING>
  </OVERALL_ASSESSMENT>
  
  <DETAILED_FEEDBACK>
    <STRENGTHS>List of what the student did well</STRENGTHS>
    <AREAS_FOR_IMPROVEMENT>Specific areas that need work</AREAS_FOR_IMPROVEMENT>
    <SUGGESTIONS>Concrete suggestions for improvement</SUGGESTIONS>
  </DETAILED_FEEDBACK>
  
  <LEARNING_RESOURCES>
    <RESOURCE>Links or references for learning</RESOURCE>
  </LEARNING_RESOURCES>
</COMPREHENSIVE_EVALUATION>

Focus on identifying errors and providing constructive feedback for improvement.
"""
        
        state["general_prompt"] = general_prompt
        return state
    

    
    def _get_examples_for_item(self, item: RubricItem) -> List[str]:
        """Get examples relevant to the rubric item"""
        examples = {
            "correctness": [
                "Function returns expected output for given inputs",
                "Handles edge cases like empty lists or null values",
                "No infinite loops or crashes"
            ],
            "code_quality": [
                "Variable names are descriptive and meaningful",
                "Code is properly indented and formatted",
                "Functions are reasonably sized and focused"
            ],
            "error_handling": [
                "Validates input parameters",
                "Uses try-except blocks where appropriate",
                "Provides meaningful error messages"
            ]
        }
        
        return examples.get(item.id, [])
    
    def _get_resources_for_item(self, item: RubricItem) -> List[str]:
        """Get learning resources for the rubric item"""
        resources = {
            "correctness": [
                "Python documentation on functions and control flow",
                "Common programming patterns and algorithms"
            ],
            "code_quality": [
                "PEP 8 style guide for Python",
                "Clean code principles"
            ],
            "error_handling": [
                "Python exception handling",
                "Input validation techniques"
            ]
        }
        
        return resources.get(item.id, [])
    
    def _prompt_to_dict(self, prompt: CorrectionPrompt) -> Dict[str, Any]:
        """Convert CorrectionPrompt to dictionary"""
        return {
            "rubric_item_id": prompt.rubric_item_id,
            "rubric_item_title": prompt.rubric_item_title,
            "prompt": prompt.prompt,
            "criteria": prompt.criteria,
            "examples": prompt.examples,
            "resources": prompt.resources
        }
    
    def generate_prompts(
        self,
        assignment_description: str,
        rubric: Rubric
    ) -> PromptSet:
        """Generate correction prompts for the given assignment and rubric"""
        
        state = PromptGeneratorState()
        state["assignment_description"] = assignment_description
        state["rubric"] = rubric
        state["programming_language"] = rubric.programming_language
        
        result = self.graph.invoke(state)
        
        return PromptSet(
            assignment_description=assignment_description,
            programming_language=rubric.programming_language,
            prompts=result["generated_prompts"],
            general_prompt=result["general_prompt"]
        )
