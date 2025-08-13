#!/usr/bin/env python3
"""
Prompt Generator Agent - Standalone Runner

This script allows running the prompt generator agent independently
to generate correction prompts from rubrics.
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

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
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
    """Main entry point for prompt generator"""
    parser = argparse.ArgumentParser(description="Prompt Generator Agent")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file")
    parser.add_argument("--rubric", "-r", required=True, help="Path to rubric file (from requirement generator)")
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
        with open(args.assignment, "r", encoding="utf-8") as f:
            assignment_description = f.read()
        
        # Load rubric from storage or file
        storage = OutputStorage(args.storage_dir)
        if args.rubric.endswith('.json'):
            rubric = storage.load_rubric(args.rubric)
        else:
            # Assume it's an assignment ID and try to find the latest rubric
            rubric_path = storage.get_latest_output("requirement_generator", args.rubric)
            if not rubric_path:
                print(f"Error: Could not find rubric for assignment ID '{args.rubric}'")
                sys.exit(1)
            rubric = storage.load_rubric(rubric_path)
            
    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)
    
    # Generate assignment ID if not provided
    assignment_id = args.assignment_id or Path(args.assignment).stem
    
    # Initialize agent
    try:
        agent = PromptGeneratorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run the prompt generator
    print("🚀 Starting prompt generation...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"📋 Rubric: {args.rubric}")
    print(f"🆔 Assignment ID: {assignment_id}")
    print("-" * 50)
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"prompt_generator_{assignment_id}") as run:
            mlflow.log_param("assignment_file", args.assignment)
            mlflow.log_param("rubric_file", args.rubric)
            mlflow.log_param("assignment_id", assignment_id)
            mlflow.log_param("model_name", config.get("model_name"))
            mlflow.log_param("provider", config.get("provider"))
            mlflow.log_param("assignment_length", len(assignment_description))
            mlflow.log_param("rubric_items_count", len(rubric.items))
            
            start_time = time.time()
            prompt_set = agent.generate_prompts(
                assignment_description=assignment_description,
                rubric=rubric
            )
            end_time = time.time()
            
            # Log metrics
            mlflow.log_metric("prompt_generation_time", end_time - start_time)
            mlflow.log_metric("prompts_count", len(prompt_set.prompts))
            mlflow.log_param("general_prompt_length", len(prompt_set.general_prompt))
            
            # Save prompts as artifact
            prompts_data = {
                "assignment_description": prompt_set.assignment_description,
                "programming_language": prompt_set.programming_language,
                "general_prompt": prompt_set.general_prompt,
                "prompts": [
                    {
                        "rubric_item_id": prompt.rubric_item_id,
                        "rubric_item_title": prompt.rubric_item_title,
                        "prompt": prompt.prompt,
                        "criteria": prompt.criteria,
                        "examples": getattr(prompt, 'examples', None),
                        "resources": getattr(prompt, 'resources', None)
                    }
                    for prompt in prompt_set.prompts
                ]
            }
            mlflow.log_dict(prompts_data, "generated_prompts.json")
            
            # Save to storage
            metadata = {
                "assignment_file": args.assignment,
                "rubric_file": args.rubric,
                "model_name": config.get("model_name"),
                "provider": config.get("provider"),
                "mlflow_run_id": run.info.run_id
            }
            
            storage_path = storage.save_prompts(prompt_set, assignment_id, metadata)
            
            # Log storage path
            mlflow.log_param("storage_path", storage_path)
            mlflow.log_param("mlflow_run_id", run.info.run_id)
        
        # Prepare output
        output_data = {
            "metadata": {
                "assignment_file": args.assignment,
                "rubric_file": args.rubric,
                "assignment_id": assignment_id,
                "storage_path": storage_path
            },
            "prompt_set": {
                "assignment_description": prompt_set.assignment_description,
                "programming_language": prompt_set.programming_language,
                "general_prompt": prompt_set.general_prompt,
                "prompts": [
                    {
                        "rubric_item_id": prompt.rubric_item_id,
                        "rubric_item_title": prompt.rubric_item_title,
                        "prompt": prompt.prompt,
                        "criteria": prompt.criteria,
                        "examples": getattr(prompt, 'examples', None),
                        "resources": getattr(prompt, 'resources', None)
                    }
                    for prompt in prompt_set.prompts
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
        print("\n📊 PROMPT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Number of Prompts Generated: {len(prompt_set.prompts)}")
        print(f"Programming Language: {prompt_set.programming_language}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"\nPrompts by Rubric Item:")
        for i, prompt in enumerate(prompt_set.prompts, 1):
            print(f"  {i}. {prompt.rubric_item_title} ({prompt.rubric_item_id})")
            print(f"     Criteria: {len(prompt.criteria)} items")
            if hasattr(prompt, 'examples') and prompt.examples:
                print(f"     Examples: {len(prompt.examples)} items")
            if hasattr(prompt, 'resources') and prompt.resources:
                print(f"     Resources: {len(prompt.resources)} items")
        
        print(f"\nGeneral Prompt Length: {len(prompt_set.general_prompt)} characters")
        
        if args.verbose:
            print("\n📋 DETAILED PROMPTS")
            print("=" * 50)
            print(json.dumps(output_data, indent=2))
        
    except Exception as e:
        print(f"❌ Error during prompt generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

