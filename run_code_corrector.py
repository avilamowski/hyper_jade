#!/usr/bin/env python3
"""
Code Corrector Agent - Standalone Runner

This script allows running the code corrector agent independently
to evaluate student code using generated prompts.
"""

import sys
import os
import json
import yaml
import argparse
import time
import mlflow
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.core.output_storage import OutputStorage

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def main():
    """Main entry point for code corrector"""
    parser = argparse.ArgumentParser(description="Code Corrector Agent")
    parser.add_argument("--code", "-c", required=True, help="Path to student code file")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file")
    parser.add_argument("--prompts", "-p", required=True, help="Path to prompts file (from prompt generator)")
    parser.add_argument("--language", "-l", default="python", help="Programming language")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--output", "-o", help="Output file path (optional, will auto-generate if not provided)")
    parser.add_argument("--assignment-id", help="Assignment ID for output naming (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
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
        
        # Load prompts from storage or file
        storage = OutputStorage(args.storage_dir)
        if args.prompts.endswith('.json'):
            prompt_set = storage.load_prompts(args.prompts)
        else:
            # Assume it's an assignment ID and try to find the latest prompts
            prompts_path = storage.get_latest_output("prompt_generator", args.prompts)
            if not prompts_path:
                print(f"Error: Could not find prompts for assignment ID '{args.prompts}'")
                sys.exit(1)
            prompt_set = storage.load_prompts(prompts_path)
            
    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)
    
    # Generate assignment ID if not provided
    assignment_id = args.assignment_id or Path(args.assignment).stem
    
    # Initialize agent
    try:
        agent = CodeCorrectorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run the code corrector
    print("🚀 Starting code evaluation...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"💻 Code: {args.code}")
    print(f"📋 Prompts: {args.prompts}")
    print(f"🐍 Language: {args.language}")
    print(f"🆔 Assignment ID: {assignment_id}")
    print("-" * 50)
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"code_corrector_{assignment_id}") as run:
            mlflow.log_param("assignment_file", args.assignment)
            mlflow.log_param("code_file", args.code)
            mlflow.log_param("prompts_file", args.prompts)
            mlflow.log_param("assignment_id", assignment_id)
            mlflow.log_param("programming_language", args.language)
            mlflow.log_param("model_name", config.get("model_name"))
            mlflow.log_param("provider", config.get("provider"))
            mlflow.log_param("assignment_length", len(assignment_description))
            mlflow.log_param("code_length", len(student_code))
            mlflow.log_param("prompts_count", len(prompt_set.prompts))
            
            start_time = time.time()
            correction_result = agent.correct_code(
                student_code=student_code,
                prompt_set=prompt_set
            )
            end_time = time.time()
            
            # Log metrics
            mlflow.log_metric("code_evaluation_time", end_time - start_time)
            mlflow.log_metric("total_errors", correction_result.total_errors)
            mlflow.log_metric("critical_errors", correction_result.critical_errors)
            
            # Save evaluation results as artifact
            evaluation_data = {
                "student_code": correction_result.student_code,
                "assignment_description": correction_result.assignment_description,
                "programming_language": correction_result.programming_language,
                "total_errors": correction_result.total_errors,
                "critical_errors": correction_result.critical_errors,
                "summary": correction_result.summary,
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
                    for eval in correction_result.item_evaluations
                ],
                "comprehensive_evaluation": {
                    "correctness": correction_result.comprehensive_evaluation.correctness,
                    "quality": correction_result.comprehensive_evaluation.quality,
                    "error_handling": correction_result.comprehensive_evaluation.error_handling,
                    "strengths": correction_result.comprehensive_evaluation.strengths,
                    "areas_for_improvement": correction_result.comprehensive_evaluation.areas_for_improvement,
                    "suggestions": correction_result.comprehensive_evaluation.suggestions,
                    "learning_resources": correction_result.comprehensive_evaluation.learning_resources
                }
            }
            mlflow.log_dict(evaluation_data, "evaluation_results.json")
            
            # Save to storage
            metadata = {
                "assignment_file": args.assignment,
                "code_file": args.code,
                "prompts_file": args.prompts,
                "programming_language": args.language,
                "model_name": config.get("model_name"),
                "provider": config.get("provider"),
                "mlflow_run_id": run.info.run_id
            }
            
            storage_path = storage.save_correction_result(correction_result, assignment_id, metadata)
            
            # Log storage path
            mlflow.log_param("storage_path", storage_path)
            mlflow.log_param("mlflow_run_id", run.info.run_id)
        
        # Prepare output
        output_data = {
            "metadata": {
                "assignment_file": args.assignment,
                "code_file": args.code,
                "prompts_file": args.prompts,
                "programming_language": args.language,
                "assignment_id": assignment_id,
                "storage_path": storage_path
            },
            "evaluation": {
                "total_errors": correction_result.total_errors,
                "critical_errors": correction_result.critical_errors,
                "summary": correction_result.summary
            },
            "detailed_results": {
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
                    for eval in correction_result.item_evaluations
                ],
                "comprehensive_evaluation": {
                    "correctness": correction_result.comprehensive_evaluation.correctness,
                    "quality": correction_result.comprehensive_evaluation.quality,
                    "error_handling": correction_result.comprehensive_evaluation.error_handling,
                    "strengths": correction_result.comprehensive_evaluation.strengths,
                    "areas_for_improvement": correction_result.comprehensive_evaluation.areas_for_improvement,
                    "suggestions": correction_result.comprehensive_evaluation.suggestions,
                    "learning_resources": correction_result.comprehensive_evaluation.learning_resources
                }
            }
        }
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)
            print(f"✅ Results saved to {args.output}")
        else:
            print(f"✅ Results saved to storage: {storage_path}")
        
        # Print summary
        print("\n📊 ERROR ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Errors Found: {correction_result.total_errors}")
        print(f"Critical Errors: {correction_result.critical_errors}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"\n{correction_result.summary}")
        
        print(f"\n📋 EVALUATION BY RUBRIC ITEM:")
        for i, eval in enumerate(correction_result.item_evaluations, 1):
            print(f"  {i}. {eval.rubric_item_title} ({eval.rubric_item_id})")
            print(f"     Status: {'✅ PASS' if eval.is_passing else '❌ FAIL'}")
            print(f"     Errors: {len(eval.errors_found)}")
            if eval.errors_found:
                for error in eval.errors_found[:3]:  # Show first 3 errors
                    print(f"       - {error.severity}: {error.description}")
                if len(eval.errors_found) > 3:
                    print(f"       ... and {len(eval.errors_found) - 3} more errors")
        
        print(f"\n💡 COMPREHENSIVE EVALUATION:")
        print(f"  Correctness: {correction_result.comprehensive_evaluation.correctness}")
        print(f"  Quality: {correction_result.comprehensive_evaluation.quality}")
        print(f"  Error Handling: {correction_result.comprehensive_evaluation.error_handling}")
        
        if correction_result.comprehensive_evaluation.strengths:
            print(f"\n✅ STRENGTHS:")
            for strength in correction_result.comprehensive_evaluation.strengths:
                print(f"  - {strength}")
        
        if correction_result.comprehensive_evaluation.areas_for_improvement:
            print(f"\n🔧 AREAS FOR IMPROVEMENT:")
            for area in correction_result.comprehensive_evaluation.areas_for_improvement:
                print(f"  - {area}")
        
        if args.verbose:
            print("\n📋 DETAILED RESULTS")
            print("=" * 50)
            print(json.dumps(output_data, indent=2))
        
    except Exception as e:
        print(f"❌ Error during code evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

