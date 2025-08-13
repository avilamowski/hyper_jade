#!/usr/bin/env python3
"""
Test script for the individual agent functionality

This script tests the ability to run each agent independently and store/load outputs.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.output_storage import OutputStorage
from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent, Rubric, RubricItem
from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent, PromptSet, CorrectionPrompt
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent, CorrectionResult, ItemEvaluation, ErrorIdentification, ComprehensiveEvaluation

def create_test_config():
    """Create a test configuration"""
    return {
        "provider": "ollama",
        "model_name": "qwen2.5:7b",
        "temperature": 0.1
    }

def create_test_assignment():
    """Create a test assignment description"""
    return """
    Write a Python function that calculates the factorial of a given number.
    
    Requirements:
    - The function should be named 'factorial'
    - It should take one parameter: n (integer)
    - It should return the factorial of n
    - Handle edge cases (negative numbers, zero)
    - Include proper error handling
    """

def create_test_code():
    """Create test student code"""
    return '''
def factorial(n):
    """Calculate the factorial of a number."""
    if n < 0:
        return None
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
'''

def test_output_storage():
    """Test the output storage functionality"""
    print("🧪 Testing Output Storage...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = OutputStorage(temp_dir)
        
        # Test rubric storage
        rubric = Rubric(
            title="Test Rubric",
            description="Test rubric for factorial function",
            programming_language="python",
            items=[
                RubricItem(
                    id="functionality",
                    title="Functionality",
                    description="Function works correctly",
                    criteria=["Returns correct factorial", "Handles edge cases"]
                )
            ]
        )
        
        # Save and load rubric
        saved_path = storage.save_rubric(rubric, "test_assignment", {"test": "metadata"})
        loaded_rubric = storage.load_rubric(saved_path)
        
        assert loaded_rubric.title == rubric.title
        assert len(loaded_rubric.items) == len(rubric.items)
        print("✅ Rubric storage test passed")
        
        # Test prompt storage
        prompt_set = PromptSet(
            assignment_description="Test assignment",
            programming_language="python",
            prompts=[
                CorrectionPrompt(
                    rubric_item_id="functionality",
                    rubric_item_title="Functionality",
                    prompt="Test prompt",
                    criteria=["Test criteria"]
                )
            ],
            general_prompt="General test prompt"
        )
        
        # Save and load prompts
        saved_prompts_path = storage.save_prompts(prompt_set, "test_assignment", {"test": "metadata"})
        loaded_prompts = storage.load_prompts(saved_prompts_path)
        
        assert loaded_prompts.assignment_description == prompt_set.assignment_description
        assert len(loaded_prompts.prompts) == len(prompt_set.prompts)
        print("✅ Prompt storage test passed")
        
        # Test correction result storage
        correction_result = CorrectionResult(
            student_code="test code",
            assignment_description="test assignment",
            programming_language="python",
            item_evaluations=[
                ItemEvaluation(
                    rubric_item_id="functionality",
                    rubric_item_title="Functionality",
                    is_passing=True,
                    overall_feedback="Good work",
                    errors_found=[]
                )
            ],
            comprehensive_evaluation=ComprehensiveEvaluation(
                correctness="Good",
                quality="Good",
                error_handling="Good",
                strengths=["Well structured"],
                areas_for_improvement=[],
                suggestions=[],
                learning_resources=[]
            ),
            total_errors=0,
            critical_errors=0,
            summary="Good implementation"
        )
        
        # Save and load correction result
        saved_result_path = storage.save_correction_result(correction_result, "test_assignment", {"test": "metadata"})
        loaded_result = storage.load_correction_result(saved_result_path)
        
        assert loaded_result.student_code == correction_result.student_code
        assert loaded_result.total_errors == correction_result.total_errors
        print("✅ Correction result storage test passed")
        
        # Test listing outputs
        outputs = storage.list_outputs()
        assert "requirement_generator" in outputs
        assert "prompt_generator" in outputs
        assert "code_corrector" in outputs
        print("✅ Output listing test passed")
        
        # Test getting latest output
        latest_rubric = storage.get_latest_output("requirement_generator", "test_assignment")
        assert latest_rubric is not None
        print("✅ Latest output retrieval test passed")

def test_agent_initialization():
    """Test that agents can be initialized"""
    print("🧪 Testing Agent Initialization...")
    
    config = create_test_config()
    
    try:
        requirement_agent = RequirementGeneratorAgent(config)
        print("✅ Requirement Generator Agent initialized")
    except Exception as e:
        print(f"❌ Requirement Generator Agent failed: {e}")
        return False
    
    try:
        prompt_agent = PromptGeneratorAgent(config)
        print("✅ Prompt Generator Agent initialized")
    except Exception as e:
        print(f"❌ Prompt Generator Agent failed: {e}")
        return False
    
    try:
        code_agent = CodeCorrectorAgent(config)
        print("✅ Code Corrector Agent initialized")
    except Exception as e:
        print(f"❌ Code Corrector Agent failed: {e}")
        return False
    
    return True

def test_agent_workflow():
    """Test the complete agent workflow"""
    print("🧪 Testing Agent Workflow...")
    
    config = create_test_config()
    assignment = create_test_assignment()
    code = create_test_code()
    
    try:
        # Step 1: Generate rubric
        requirement_agent = RequirementGeneratorAgent(config)
        rubric = requirement_agent.generate_rubric(assignment, "python")
        print(f"✅ Generated rubric with {len(rubric.items)} items")
        
        # Step 2: Generate prompts
        prompt_agent = PromptGeneratorAgent(config)
        prompt_set = prompt_agent.generate_prompts(assignment, rubric)
        print(f"✅ Generated {len(prompt_set.prompts)} prompts")
        
        # Step 3: Evaluate code
        code_agent = CodeCorrectorAgent(config)
        result = code_agent.correct_code(code, prompt_set)
        print(f"✅ Evaluated code with {result.total_errors} errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent workflow failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Agent Tests")
    print("=" * 50)
    
    # Test 1: Output Storage
    try:
        test_output_storage()
    except Exception as e:
        print(f"❌ Output storage test failed: {e}")
        return False
    
    # Test 2: Agent Initialization
    if not test_agent_initialization():
        return False
    
    # Test 3: Agent Workflow (optional - requires LLM)
    print("\n💡 Note: Agent workflow test requires LLM (Ollama/OpenAI)")
    print("   Run with --test-workflow to enable this test")
    
    if "--test-workflow" in sys.argv:
        if not test_agent_workflow():
            return False
    
    print("\n✅ All tests passed!")
    print("🎉 Individual agent functionality is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
