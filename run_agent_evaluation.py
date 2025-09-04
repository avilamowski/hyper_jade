#!/usr/bin/env python3
"""
Agent Evaluation Runner - Standalone

This script allows running the agent evaluator independently
to evaluate agent outputs without waiting for the generation process.
"""

import sys
import os
import json
import yaml
import argparse
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.utils.agent_evaluator import AgentEvaluator

from src.config import get_agent_config, load_config, load_langsmith_config



def read_requirement_files(requirements_dir: str) -> list:
    """Read all requirement files from the specified directory"""
    requirements = []
    req_dir = Path(requirements_dir)
    
    if not req_dir.exists():
        print(f"Error: Requirements directory not found: {requirements_dir}")
        return requirements
    
    # Read requirement files in order
    for i in range(1, 100):  # Support up to 99 requirements
        req_file = req_dir / f"requirement_{i:02d}.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        requirements.append(content)
            except Exception as e:
                print(f"Error reading {req_file}: {e}")
        else:
            break
    
    return requirements

def main():
    """Main entry point for agent evaluation"""
    parser = argparse.ArgumentParser(description="Agent Evaluation Runner")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--requirements-dir", "-r", required=True, help="Directory containing requirement files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for evaluation results")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--assignment-path", help="Assignment path for teacher comparison (e.g., '2p', '3p')")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()
    
    # Print model information
    agent_config = get_agent_config(config, 'agent_evaluator')
    print(f"🤖 Using model: {agent_config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {agent_config.get('provider', 'Unknown')}")
    
    # Print evaluation mode
    if args.assignment_path:
        print(f"📚 Mode: Evaluation with teacher comparison for {args.assignment_path}")
    else:
        print("📚 Mode: Standard evaluation (no teacher comparison)")
    
    print("-" * 50)
    
    # Check if assignment file exists
    if not os.path.exists(args.assignment):
        print(f"Error: Assignment file not found: {args.assignment}")
        sys.exit(1)
    
    # Check if requirements directory exists
    if not os.path.exists(args.requirements_dir):
        print(f"Error: Requirements directory not found: {args.requirements_dir}")
        sys.exit(1)
    
    # Initialize evaluator
    try:
        evaluator = AgentEvaluator(config)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        sys.exit(1)
    
    # Read assignment description
    try:
        with open(args.assignment, 'r', encoding='utf-8') as f:
            assignment_description = f.read().strip()
    except Exception as e:
        print(f"Error reading assignment file: {e}")
        sys.exit(1)
    
    # Read requirement files
    print("📖 Reading requirement files...")
    requirements = read_requirement_files(args.requirements_dir)
    
    if not requirements:
        print("Error: No requirement files found")
        sys.exit(1)
    
    print(f"✅ Found {len(requirements)} requirement files")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run the evaluation
    print("🚀 Starting agent evaluation...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"📁 Requirements directory: {args.requirements_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Evaluate requirement generator
        evaluation_result = evaluator.evaluate_requirement_generator(
            assignment_description,
            requirements,
            args.requirements_dir,
            assignment_path=args.assignment_path
        )
        
        end_time = time.time()
        
        # Save evaluation result
        evaluation_file = output_path / "requirement_generator_evaluation.json"
        
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        
        # Output results
        print(f"✅ Evaluation completed")
        print(f"⏱️  Evaluation time: {end_time - start_time:.2f} seconds")
        
        # Print summary
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Assignment file: {args.assignment}")
        print(f"Requirements directory: {args.requirements_dir}")
        print(f"Number of requirements evaluated: {len(requirements)}")
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        
        if "error" in evaluation_result:
            print(f"❌ Evaluation error: {evaluation_result['error']}")
        else:
            print(f"📊 Overall score: {evaluation_result.get('overall_score', 'N/A')}")
            print(f"📊 Average score: {evaluation_result.get('average_score', 'N/A')}")
            
            # Display teacher comparison results if available
            if 'coverage_ratio' in evaluation_result:
                print(f"\n📚 TEACHER COMPARISON RESULTS")
                print("=" * 50)
                print(f"📊 Coverage ratio: {evaluation_result.get('coverage_ratio', 0.0):.2f}")
                print(f"📊 Teacher requirements: {evaluation_result.get('teacher_requirements_count', 0)}")
                print(f"📊 Generated requirements: {evaluation_result.get('generated_requirements_count', 0)}")
            
            if args.verbose:
                print("\n📄 DETAILED EVALUATION")
                print("=" * 50)
                for key, value in evaluation_result.items():
                    if key not in ['overall_score', 'average_score', 'teacher_comparison', 'summary', 'suggestions']:
                        if isinstance(value, dict) and 'score' in value:
                            print(f"{key.replace('_', ' ').title()}: {value['score']}/5")
                            if 'justification' in value:
                                print(f"  Justification: {value['justification']}")
                        else:
                            print(f"{key}: {value}")
                
                if 'summary' in evaluation_result:
                    print(f"\nSummary: {evaluation_result['summary']}")
                
                if 'suggestions' in evaluation_result:
                    print(f"\nSuggestions:")
                    for i, suggestion in enumerate(evaluation_result['suggestions'], 1):
                        print(f"  {i}. {suggestion}")
        
        print(f"\n💾 Evaluation saved to: {evaluation_file}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
