#!/usr/bin/env python3
"""
Main entry point for the Assignment Evaluation System

This script provides a command-line interface for evaluating student assignments
using the three-agent pipeline: Requirement Generator → Prompt Generator → Code Corrector.
"""

import sys
import os
import json
import yaml
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.assignment_evaluator import AssignmentEvaluator

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assignment Evaluation System")
    parser.add_argument("--code", "-c", required=True, help="Path to student code file")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file")
    parser.add_argument("--language", "-l", default="python", help="Programming language")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--use-stored", action="store_true", help="Use stored outputs from previous agent runs")
    parser.add_argument("--assignment-id", help="Assignment ID for stored outputs (defaults to assignment filename)")
    parser.add_argument("--storage-dir", default="outputs", help="Output storage directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    
    # Print model information
    print(f"🤖 Using model: {config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {config.get('provider', 'Unknown')}")
    print("-" * 50)
    
    # Read input files
    try:
        with open(args.code, "r", encoding="utf-8") as f:
            student_code = f.read()
        
        with open(args.assignment, "r", encoding="utf-8") as f:
            assignment_description = f.read()
    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)
    
    # Generate assignment ID if not provided
    assignment_id = args.assignment_id or Path(args.assignment).stem
    
    # Initialize the assignment evaluator
    try:
        evaluator = AssignmentEvaluator(config)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        sys.exit(1)
    
    # Check if we should use stored outputs
    if args.use_stored:
        from src.core.output_storage import OutputStorage
        storage = OutputStorage(args.storage_dir)
        
        print("🔄 Using stored outputs from previous agent runs...")
        print(f"🆔 Assignment ID: {assignment_id}")
        
        # Check for stored rubric
        rubric_path = storage.get_latest_output("requirement_generator", assignment_id)
        if rubric_path:
            print(f"📋 Using stored rubric: {Path(rubric_path).name}")
            rubric = storage.load_rubric(rubric_path)
        else:
            print("⚠️  No stored rubric found, will generate new one")
            rubric = None
        
        # Check for stored prompts
        prompts_path = storage.get_latest_output("prompt_generator", assignment_id)
        if prompts_path:
            print(f"📝 Using stored prompts: {Path(prompts_path).name}")
            prompt_set = storage.load_prompts(prompts_path)
        else:
            print("⚠️  No stored prompts found, will generate new ones")
            prompt_set = None
        
        # Run evaluation with stored outputs
        if rubric and prompt_set:
            print("✅ Using stored rubric and prompts for evaluation")
            # We need to modify the evaluator to accept pre-generated outputs
            # For now, we'll run the full pipeline but it will skip generation
            result = evaluator.evaluate_assignment(
                assignment_description=assignment_description,
                student_code=student_code,
                programming_language=args.language
            )
        else:
            print("🔄 Running full pipeline (some outputs not found)")
            result = evaluator.evaluate_assignment(
                assignment_description=assignment_description,
                student_code=student_code,
                programming_language=args.language
            )
    else:
        # Run the evaluation pipeline
        print("🚀 Starting assignment evaluation pipeline...")
        print(f"📝 Assignment: {args.assignment}")
        print(f"💻 Code: {args.code}")
        print(f"🐍 Language: {args.language}")
        print("-" * 50)
        
        result = evaluator.evaluate_assignment(
            assignment_description=assignment_description,
            student_code=student_code,
            programming_language=args.language
        )
        
        # Prepare output
        output = {
            "metadata": {
                "assignment_file": args.assignment,
                "code_file": args.code,
                "programming_language": args.language,
                "pipeline_version": result.metadata.get("pipeline_version", "1.0"),
                "agents_used": result.metadata.get("agents_used", [])
            },
            "evaluation": {
                "total_errors": result.correction_result.total_errors,
                "critical_errors": result.correction_result.critical_errors,
                "summary": result.correction_result.summary
            },
            "detailed_results": {
                "rubric": {
                    "title": result.rubric.title,
                    "description": result.rubric.description,
                    "items": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "description": item.description,
                            "criteria": item.criteria
                        }
                        for item in result.rubric.items
                    ]
                },
                "item_evaluations": [
                    {
                        "rubric_item_id": eval.rubric_item_id,
                        "rubric_item_title": eval.rubric_item_title,
                        "is_passing": eval.is_passing,
                        "overall_feedback": eval.overall_feedback,
                        "errors_found": [
                            {
                                "error_type": error.error_type,
                                "location": error.location,
                                "description": error.description,
                                "severity": error.severity,
                                "suggestion": error.suggestion,
                                "line_number": error.line_number
                            }
                            for error in eval.errors_found
                        ]
                    }
                    for eval in result.correction_result.item_evaluations
                ],
                "comprehensive_evaluation": {
                    "correctness": result.correction_result.comprehensive_evaluation.correctness,
                    "quality": result.correction_result.comprehensive_evaluation.quality,
                    "error_handling": result.correction_result.comprehensive_evaluation.error_handling,
                    "strengths": result.correction_result.comprehensive_evaluation.strengths,
                    "areas_for_improvement": result.correction_result.comprehensive_evaluation.areas_for_improvement,
                    "suggestions": result.correction_result.comprehensive_evaluation.suggestions,
                    "learning_resources": result.correction_result.comprehensive_evaluation.learning_resources
                }
            }
        }
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            print(f"✅ Results saved to {args.output}")
        else:
            # Print summary
            print("\n📊 ERROR ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Total Errors Found: {result.correction_result.total_errors}")
            print(f"Critical Errors: {result.correction_result.critical_errors}")
            print(f"\n{result.correction_result.summary}")
            
            if args.verbose:
                print("\n📋 DETAILED RESULTS")
                print("=" * 50)
                print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
