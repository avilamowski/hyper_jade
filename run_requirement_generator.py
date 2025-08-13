#!/usr/bin/env python3
"""
Requirement Generator Agent - Standalone Runner

This script allows running the requirement generator agent independently
to generate rubrics from assignment descriptions.
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

from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent
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
    """Main entry point for requirement generator"""
    parser = argparse.ArgumentParser(description="Requirement Generator Agent")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file")
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
    
    # Read assignment file
    try:
        with open(args.assignment, "r", encoding="utf-8") as f:
            assignment_description = f.read()
    except Exception as e:
        print(f"Error reading assignment file: {e}")
        sys.exit(1)
    
    # Generate assignment ID if not provided
    assignment_id = args.assignment_id or Path(args.assignment).stem
    
    # Initialize storage and agent
    try:
        storage = OutputStorage(args.storage_dir)
        agent = RequirementGeneratorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run the requirement generator
    print("🚀 Starting requirement generation...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"🐍 Language: {args.language}")
    print(f"🆔 Assignment ID: {assignment_id}")
    print("-" * 50)
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"requirement_generator_{assignment_id}") as run:
            mlflow.log_param("assignment_file", args.assignment)
            mlflow.log_param("programming_language", args.language)
            mlflow.log_param("assignment_id", assignment_id)
            mlflow.log_param("model_name", config.get("model_name"))
            mlflow.log_param("provider", config.get("provider"))
            mlflow.log_param("assignment_length", len(assignment_description))
            
            start_time = time.time()
            rubric = agent.generate_rubric(
                assignment_description=assignment_description,
                programming_language=args.language
            )
            end_time = time.time()
            
            # Log metrics
            mlflow.log_metric("rubric_generation_time", end_time - start_time)
            mlflow.log_metric("rubric_items_count", len(rubric.items))
            mlflow.log_param("rubric_title", rubric.title)
            
            # Save rubric as artifact
            rubric_data = {
                "title": rubric.title,
                "description": rubric.description,
                "programming_language": rubric.programming_language,
                "items": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "criteria": item.criteria
                    }
                    for item in rubric.items
                ]
            }
            mlflow.log_dict(rubric_data, "generated_rubric.json")
            
            # Save to storage
            metadata = {
                "assignment_file": args.assignment,
                "programming_language": args.language,
                "model_name": config.get("model_name"),
                "provider": config.get("provider"),
                "mlflow_run_id": run.info.run_id
            }
            
            storage_path = storage.save_rubric(rubric, assignment_id, metadata)
            
            # Log storage path
            mlflow.log_param("storage_path", storage_path)
            mlflow.log_param("mlflow_run_id", run.info.run_id)
        
        # Prepare output
        output_data = {
            "metadata": {
                "assignment_file": args.assignment,
                "programming_language": args.language,
                "assignment_id": assignment_id,
                "storage_path": storage_path
            },
            "rubric": {
                "title": rubric.title,
                "description": rubric.description,
                "programming_language": rubric.programming_language,
                "items": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "description": item.description,
                        "criteria": item.criteria
                    }
                    for item in rubric.items
                ]
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
        print("\n📊 RUBRIC GENERATION SUMMARY")
        print("=" * 50)
        print(f"Rubric Title: {rubric.title}")
        print(f"Number of Items: {len(rubric.items)}")
        print(f"Programming Language: {rubric.programming_language}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"\nRubric Items:")
        for i, item in enumerate(rubric.items, 1):
            print(f"  {i}. {item.title} ({item.id})")
            print(f"     {item.description}")
            print(f"     Criteria: {len(item.criteria)} items")
        
        if args.verbose:
            print("\n📋 DETAILED RUBRIC")
            print("=" * 50)
            print(json.dumps(output_data, indent=2))
        
    except Exception as e:
        print(f"❌ Error during requirement generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

