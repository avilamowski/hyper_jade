#!/usr/bin/env python3
"""
Test script to test the full RAG flow: generate_enhanced_examples -> format -> rag_generate_prompts
Run with: uv run python test_rag_full_flow.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.rag_prompt_generator.code_generator import CodeExampleGenerator
from src.agents.rag_prompt_generator.rag_system import RAGSystem
from src.agents.rag_prompt_generator.rag_prompt_generator import rag_example_generation_node
from src.models import Requirement

# Sample data for testing
SAMPLE_REQUIREMENT = "El código no debe permitir que se intente vender una cantidad negativa o cero, y debe manejar estos casos adecuadamente."

SAMPLE_ASSIGNMENT_DESCRIPTION = """
Tarea: Sistema de Ventas
Desarrolla un programa que permita registrar ventas de productos.
El programa debe validar que las cantidades sean números enteros positivos.
"""


async def test_full_flow():
    """Test the full RAG flow from example generation to prompt formatting"""
    print("=" * 80)
    print("Testing Full RAG Flow")
    print("=" * 80)
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not set!")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        return None
    
    # Initialize RAG System
    print("\n1. Initializing RAG System...")
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        print("   ✅ RAG System initialized")
    except Exception as e:
        print(f"   ⚠️  RAG System initialization failed: {e}")
        print("   Continuing without RAG (will use mock data)")
        rag_system = None
    
    # Initialize Code Example Generator
    print("\n2. Initializing Code Example Generator...")
    code_generator = CodeExampleGenerator(rag_system)
    await code_generator.initialize()
    print("   ✅ Code Example Generator initialized")
    
    # Test generate_enhanced_examples
    print("\n3. Testing generate_enhanced_examples...")
    print(f"   Requirement: {SAMPLE_REQUIREMENT[:60]}...")
    
    try:
        enhanced_examples = await code_generator.generate_enhanced_examples(
            requirement=SAMPLE_REQUIREMENT,
            num_examples=3,
            max_theory_results=5,
            dataset="python"
        )
        
        print(f"   ✅ Generated {len(enhanced_examples)} enhanced examples")
        
        # Log details of each example
        for i, ex in enumerate(enhanced_examples, 1):
            print(f"\n   Example {i}:")
            print(f"     Description: {ex.get('description', 'N/A')}")
            print(f"     Class Name: {ex.get('class_name', 'Unknown')}")
            print(f"     Code length: {len(ex.get('code', ''))} chars")
            print(f"     Has Improvements: {bool(ex.get('improvements'))}")
            print(f"     Has Theory Alignment: {bool(ex.get('theory_alignment'))}")
            
            if ex.get('improvements'):
                improvements = ex.get('improvements', [])
                if isinstance(improvements, list):
                    print(f"     Improvements ({len(improvements)} items)")
                else:
                    print(f"     Improvements: {str(improvements)[:100]}...")
            
            if ex.get('theory_alignment'):
                ta = ex.get('theory_alignment', '')
                if isinstance(ta, list):
                    print(f"     Theory Alignment (list): {len(ta)} items")
                else:
                    print(f"     Theory Alignment: {str(ta)[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Error generating enhanced examples: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Test formatting (simulate what rag_example_generation_node does)
    print("\n4. Testing example formatting (rag_example_generation_node logic)...")
    
    try:
        formatted_examples = []
        for i, example in enumerate(enhanced_examples, 1):
            code = example.get("code", "")
            description = example.get("description", f"Example {i}")
            improvements = example.get("improvements", [])
            theory_alignment = example.get("theory_alignment", "")
            class_name = example.get("class_name", "Unknown")
            
            print(f"\n   Formatting example {i}:")
            print(f"     class_name={class_name}")
            print(f"     improvements_type={type(improvements).__name__}")
            print(f"     theory_alignment_type={type(theory_alignment).__name__}")
            
            formatted_example = f"Example {i}: {description}\n"
            formatted_example += f"**{class_name}**\n"
            formatted_example += f"```python\n{code}\n```\n"
            
            if improvements:
                # Handle both list and string formats
                if isinstance(improvements, list):
                    improvements_str = '; '.join(str(imp) for imp in improvements if imp)
                else:
                    improvements_str = str(improvements)
                if improvements_str:
                    formatted_example += f"Improvements: {improvements_str}\n"
                    print(f"     ✓ Added improvements ({len(improvements_str)} chars)")
            
            if theory_alignment:
                # Handle both list and string formats
                if isinstance(theory_alignment, list):
                    theory_alignment_str = ' '.join(str(ta) for ta in theory_alignment if ta)
                else:
                    theory_alignment_str = str(theory_alignment).strip()
                if theory_alignment_str:
                    formatted_example += f"Theory alignment: {theory_alignment_str}\n"
                    print(f"     ✓ Added theory alignment ({len(theory_alignment_str)} chars)")
            
            formatted_examples.append(formatted_example)
        
        formatted_output = "\n\n".join(formatted_examples)
        print(f"\n   ✅ Formatted {len(formatted_examples)} examples")
        print(f"   Total formatted output length: {len(formatted_output)} chars")
        
        # Show a preview of the formatted output
        print("\n5. Formatted Output Preview:")
        print("=" * 80)
        print(formatted_output[:1000] + "..." if len(formatted_output) > 1000 else formatted_output)
        print("=" * 80)
        
        return {
            'enhanced_examples': enhanced_examples,
            'formatted_output': formatted_output
        }
        
    except Exception as e:
        print(f"   ❌ Error formatting examples: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_rag_example_generation_node():
    """Test the actual rag_example_generation_node function"""
    print("\n" + "=" * 80)
    print("Testing rag_example_generation_node directly")
    print("=" * 80)
    
    # Create a mock requirement
    requirement = Requirement(
        requirement=SAMPLE_REQUIREMENT,
        requirement_id="test-001"
    )
    
    # Initialize RAG System
    print("\n1. Initializing RAG System...")
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        print("   ✅ RAG System initialized")
    except Exception as e:
        print(f"   ⚠️  RAG System initialization failed: {e}")
        return None
    
    # Initialize Code Example Generator
    print("\n2. Initializing Code Example Generator...")
    code_generator = CodeExampleGenerator(rag_system)
    await code_generator.initialize()
    print("   ✅ Code Example Generator initialized")
    
    # Test rag_example_generation_node
    print("\n3. Testing rag_example_generation_node...")
    try:
        result = await rag_example_generation_node(
            requirement=requirement,
            agent_config={},
            llm=None,  # Not used in this node
            rag_system=rag_system,
            code_generator=code_generator
        )
        
        print(f"   ✅ Got result from rag_example_generation_node")
        print(f"   Result length: {len(result)} chars")
        print("\n4. Result Preview:")
        print("=" * 80)
        print(result[:1500] + "..." if len(result) > 1500 else result)
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error in rag_example_generation_node: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔧 RAG Full Flow Test")
    print("=" * 80)
    
    # Check for Google API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  WARNING: GOOGLE_API_KEY not set in environment")
        print("   The test will fail if the API key is required")
        print()
    
    # Run the full flow test
    print("\n📋 Test 1: Full Flow (generate_enhanced_examples -> format)")
    result1 = asyncio.run(test_full_flow())
    
    # Run the node test
    print("\n📋 Test 2: rag_example_generation_node")
    result2 = asyncio.run(test_rag_example_generation_node())
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if result1:
        print("✅ Test 1 (Full Flow): PASSED")
        print(f"   Generated {len(result1['enhanced_examples'])} enhanced examples")
    else:
        print("❌ Test 1 (Full Flow): FAILED")
    
    if result2:
        print("✅ Test 2 (rag_example_generation_node): PASSED")
        print(f"   Generated formatted output ({len(result2)} chars)")
    else:
        print("❌ Test 2 (rag_example_generation_node): FAILED")
    
    print("=" * 80)


