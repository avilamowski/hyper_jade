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
    
    # Initialize the assignment evaluator
    try:
        evaluator = AssignmentEvaluator(config)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        sys.exit(1)
    
    # Run the evaluation pipeline
    print("🚀 Starting assignment evaluation pipeline...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"💻 Code: {args.code}")
    print(f"🐍 Language: {args.language}")
    print("-" * 50)
    
    try:
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
                "grade_percentage": result.correction_result.grade_percentage,
                "total_score": result.correction_result.total_score,
                "max_possible_score": result.correction_result.max_possible_score,
                "summary": result.correction_result.summary
            },
            "detailed_results": {
                "rubric": {
                    "title": result.rubric.title,
                    "description": result.rubric.description,
                    "total_score": result.rubric.total_score,
                    "items": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "description": item.description,
                            "max_score": item.max_score,
                            "criteria": item.criteria
                        }
                        for item in result.rubric.items
                    ]
                },
                "item_evaluations": [
                    {
                        "rubric_item_id": eval.rubric_item_id,
                        "rubric_item_title": eval.rubric_item_title,
                        "total_score": eval.total_score,
                        "max_score": eval.max_score,
                        "overall_feedback": eval.overall_feedback,
                        "criteria_evaluations": [
                            {
                                "name": criterion.name,
                                "met": criterion.met,
                                "score": criterion.score,
                                "feedback": criterion.feedback,
                                "suggestion": criterion.suggestion
                            }
                            for criterion in eval.criteria_evaluations
                        ]
                    }
                    for eval in result.correction_result.item_evaluations
                ],
                "comprehensive_evaluation": {
                    "correctness": result.correction_result.comprehensive_evaluation.correctness,
                    "quality": result.correction_result.comprehensive_evaluation.quality,
                    "documentation": result.correction_result.comprehensive_evaluation.documentation,
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
            print("\n📊 EVALUATION SUMMARY")
            print("=" * 50)
            print(f"Overall Grade: {result.correction_result.grade_percentage:.1f}%")
            print(f"Total Score: {result.correction_result.total_score}/{result.correction_result.max_possible_score}")
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
