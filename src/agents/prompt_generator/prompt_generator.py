#!/usr/bin/env python3
"""
Prompt Generator Agent

This agent generates specialized correction prompts for each rubric item.
It uses RAG (Retrieval Augmented Generation) to enhance prompts with relevant
knowledge and examples.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
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
    max_score: int
    examples: List[str] = None
    resources: List[str] = None

@dataclass
class PromptSet:
    """Complete set of correction prompts for an assignment"""
    assignment_description: str
    programming_language: str
    prompts: List[CorrectionPrompt]
    general_prompt: str

class PromptGeneratorState:
    """State for prompt generation process"""
    def __init__(self):
        self.assignment_description: str = ""
        self.rubric: Optional[Rubric] = None
        self.programming_language: str = "python"
        self.generated_prompts: List[CorrectionPrompt] = []
        self.general_prompt: str = ""
        self.rag_knowledge: List[str] = []
        self.evaluation_score: float = 0.0

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
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for prompt generation"""
        graph = StateGraph(PromptGeneratorState)
        
        # Add nodes
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("generate_prompts", self._generate_prompts)
        graph.add_node("generate_general_prompt", self._generate_general_prompt)
        graph.add_node("evaluate_prompts", self._evaluate_prompts)
        graph.add_node("refine_prompts", self._refine_prompts)
        
        # Set entry point
        graph.set_entry_point("retrieve_knowledge")
        
        # Add edges
        graph.add_edge("retrieve_knowledge", "generate_prompts")
        graph.add_edge("generate_prompts", "generate_general_prompt")
        graph.add_edge("generate_general_prompt", "evaluate_prompts")
        graph.add_conditional_edges(
            "evaluate_prompts",
            self._should_refine,
            {True: "refine_prompts", False: END}
        )
        graph.add_edge("refine_prompts", "evaluate_prompts")
        
        return graph.compile()
    
    def _retrieve_knowledge(self, state: PromptGeneratorState) -> PromptGeneratorState:
        """Retrieve relevant knowledge for prompt generation"""
        logger.info("Retrieving knowledge for prompt generation")
        
        if not self.rag_enabled:
            state.rag_knowledge = []
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
        
        state.rag_knowledge = knowledge_base.get(state.programming_language, [])
        
        return state
    
    def _generate_prompts(self, state: PromptGeneratorState) -> PromptGeneratorState:
        """Generate correction prompts for each rubric item"""
        logger.info("Generating correction prompts for rubric items")
        
        if not state.rubric:
            logger.error("No rubric provided for prompt generation")
            return state
        
        prompts = []
        
        for item in state.rubric.items:
            prompt = self._generate_item_prompt(item, state)
            prompts.append(prompt)
        
        state.generated_prompts = prompts
        
        return state
    
    def _generate_item_prompt(self, item: RubricItem, state: PromptGeneratorState) -> CorrectionPrompt:
        """Generate a correction prompt for a specific rubric item"""
        
        knowledge_context = ""
        if state.rag_knowledge:
            knowledge_context = f"\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {k}" for k in state.rag_knowledge)
        
        prompt_text = f"""You are a specialized code correction agent for {state.programming_language} programming.

ASSIGNMENT DESCRIPTION:
{state.assignment_description}

RUBRIC ITEM: {item.title}
DESCRIPTION: {item.description}
MAX SCORE: {item.max_score}

CRITERIA TO EVALUATE:
{chr(10).join(f"- {criterion}" for criterion in item.criteria)}

{knowledge_context}

TASK: Analyze the student's code and evaluate it against the criteria above.

INSTRUCTIONS:
1. Examine the code carefully for the specific criteria listed
2. Check if the code meets each criterion
3. Provide specific feedback for each criterion
4. Award points based on how well each criterion is met
5. Suggest specific improvements where criteria are not met

OUTPUT FORMAT:
<EVALUATION>
  <CRITERION name="">
    <MET>Yes/No</MET>
    <SCORE>Points awarded (0-{item.max_score})</SCORE>
    <FEEDBACK>Specific feedback about this criterion</FEEDBACK>
    <SUGGESTION>How to improve this aspect</SUGGESTION>
  </CRITERION>
  
  <SUMMARY>
    <TOTAL_SCORE>Total points for this item</TOTAL_SCORE>
    <OVERALL_FEEDBACK>Overall assessment of this aspect</OVERALL_FEEDBACK>
  </SUMMARY>
</EVALUATION>

When you receive the student's code, analyze it according to these instructions and provide the evaluation in the specified format.
"""
        
        return CorrectionPrompt(
            rubric_item_id=item.id,
            rubric_item_title=item.title,
            prompt=prompt_text,
            criteria=item.criteria,
            max_score=item.max_score,
            examples=self._get_examples_for_item(item),
            resources=self._get_resources_for_item(item)
        )
    
    def _generate_general_prompt(self, state: PromptGeneratorState) -> PromptGeneratorState:
        """Generate a general correction prompt for overall assessment"""
        logger.info("Generating general correction prompt")
        
        knowledge_context = ""
        if state.rag_knowledge:
            knowledge_context = f"\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {k}" for k in state.rag_knowledge)
        
        general_prompt = f"""You are a comprehensive code correction agent for {state.programming_language} programming.

ASSIGNMENT DESCRIPTION:
{state.assignment_description}

PROGRAMMING LANGUAGE: {state.programming_language}

{knowledge_context}

TASK: Provide a comprehensive evaluation of the student's code.

INSTRUCTIONS:
1. Analyze the code for overall correctness and functionality
2. Check code quality, readability, and structure
3. Evaluate documentation and comments
4. Assess error handling and edge cases
5. Provide constructive feedback for improvement

OUTPUT FORMAT:
<COMPREHENSIVE_EVALUATION>
  <OVERALL_ASSESSMENT>
    <CORRECTNESS>Assessment of functional correctness</CORRECTNESS>
    <QUALITY>Assessment of code quality</QUALITY>
    <DOCUMENTATION>Assessment of documentation</DOCUMENTATION>
    <ERROR_HANDLING>Assessment of error handling</ERROR_HANDLING>
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

When you receive the student's code, provide a comprehensive evaluation following these instructions.
"""
        
        state.general_prompt = general_prompt
        
        return state
    
    def _evaluate_prompts(self, state: PromptGeneratorState) -> PromptGeneratorState:
        """Evaluate the generated prompts"""
        logger.info("Evaluating generated prompts")
        
        if not state.generated_prompts:
            state.evaluation_score = 0.0
            return state
        
        prompt = f"""Evaluate the following generated correction prompts for a programming assignment.

ASSIGNMENT:
{state.assignment_description}

GENERATED PROMPTS:
{json.dumps([self._prompt_to_dict(p) for p in state.generated_prompts], indent=2)}

Please evaluate these prompts on a scale of 0-10 based on:
1. Clarity: Are the instructions clear and specific?
2. Completeness: Do they cover all necessary aspects?
3. Appropriateness: Are they suitable for beginner students?
4. Actionability: Do they provide actionable feedback?

Provide a score and brief feedback.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        evaluation_text = str(response.content)
        
        # Extract score from response (simple parsing)
        try:
            score_text = evaluation_text.split("score")[1].split()[0]
            state.evaluation_score = float(score_text)
        except:
            state.evaluation_score = 7.0  # Default score
        
        return state
    
    def _should_refine(self, state: PromptGeneratorState) -> bool:
        """Determine if the prompts need refinement"""
        return state.evaluation_score < 8.0
    
    def _refine_prompts(self, state: PromptGeneratorState) -> PromptGeneratorState:
        """Refine the prompts based on evaluation"""
        logger.info("Refining prompts based on evaluation")
        
        # Regenerate prompts with improved instructions
        refined_prompts = []
        
        for item in state.rubric.items:
            prompt = self._generate_item_prompt(item, state)
            # Add refinement instructions
            prompt.prompt += "\n\nREFINEMENT: Make sure to provide very specific, actionable feedback that a beginner student can understand and implement."
            refined_prompts.append(prompt)
        
        state.generated_prompts = refined_prompts
        
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
            "documentation": [
                "Functions have clear docstrings",
                "Comments explain complex logic",
                "Code is self-explanatory"
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
            "documentation": [
                "Python docstring conventions",
                "Writing clear comments and documentation"
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
            "max_score": prompt.max_score,
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
        state.assignment_description = assignment_description
        state.rubric = rubric
        state.programming_language = rubric.programming_language
        
        result = self.graph.invoke(state)
        
        return PromptSet(
            assignment_description=assignment_description,
            programming_language=rubric.programming_language,
            prompts=result.generated_prompts,
            general_prompt=result.general_prompt
        )
