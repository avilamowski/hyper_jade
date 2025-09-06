#!/usr/bin/env python3
"""
Prompt Generator Agent - Standalone Runner

This script allows running the prompt generator agent independently
to generate Jinja2 template prompts from individual requirement files.
"""

import sys
import os
import json
import yaml
import argparse
import time
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.config import get_agent_config, load_config, load_langsmith_config
from src.models import Requirement, PromptType

# Import MLflow logger lazily to avoid circular imports
def get_mlflow_logger():
    """Get MLflow logger instance, importing it only when needed"""
    try:
        from src.core.mlflow_utils import mlflow_logger
        return mlflow_logger
    except ImportError:
        logging.warning("MLflow not available - logging will be disabled")
        return None

def safe_log_call(logger_instance, method_name, *args, **kwargs):
    """Safely call a logging method, doing nothing if logger is None"""
    if logger_instance is not None and hasattr(logger_instance, method_name):
        try:
            method = getattr(logger_instance, method_name)
            method(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Failed to call {method_name}: {e}")

def load_requirement_from_json(json_file_path: str) -> Requirement:
    """Load a requirement from a JSON file and convert it to a Requirement object"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert the type string to PromptType enum
        prompt_type = PromptType(data["type"]) if isinstance(data["type"], str) else data["type"]
        
        requirement: Requirement = {
            "requirement": data["requirement"],
            "function": data["function"],
            "type": prompt_type
        }
        
        return requirement
    except Exception as e:
        raise ValueError(f"Error loading requirement from {json_file_path}: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for prompt generator"""
    parser = argparse.ArgumentParser(description="Prompt Generator Agent")
    parser.add_argument("--requirement", "-r", nargs="+", required=True, help="Path(s) to requirement file(s) (.json)")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output", "-o", help="Output path for Jinja2 template (.jinja) - required for single requirement")
    parser.add_argument("--output-dir", help="Output directory for multiple templates - required for multiple requirements")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()
    
    # Print model information
    agent_config = get_agent_config(config, 'prompt_generator')
    print(f"🤖 Using model: {agent_config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {agent_config.get('provider', 'Unknown')}")
    print("-" * 50)
    
    # Validate arguments based on number of requirements
    num_requirements = len(args.requirement)
    
    if num_requirements == 1:
        if not args.output:
            print("Error: --output is required when processing a single requirement")
            sys.exit(1)
    else:
        if not args.output_dir:
            print("Error: --output-dir is required when processing multiple requirements")
            sys.exit(1)
    
    # Check if files exist
    for req_file in args.requirement:
        if not os.path.exists(req_file):
            print(f"Error: Requirement file not found: {req_file}")
            sys.exit(1)
    
    if not os.path.exists(args.assignment):
        print(f"Error: Assignment file not found: {args.assignment}")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = PromptGeneratorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Start MLflow run for prompt generation
    mlflow_logger = get_mlflow_logger()
    requirement_files_str = ", ".join([Path(req).name for req in args.requirement])
    output_info = Path(args.output).name if args.output else f"{num_requirements}_files"
    
    safe_log_call(mlflow_logger, "start_run",
        run_name="prompt_generation",
        tags={
            "agent": "prompt_generator",
            "requirement_files": requirement_files_str,
            "assignment_file": Path(args.assignment).name,
            "output_info": output_info,
            "num_requirements": str(num_requirements)
        }
    )
    
    # Run the prompt generator
    print("🚀 Starting prompt generation...")
    print(f"📋 Requirements: {', '.join([Path(req).name for req in args.requirement])}")
    print(f"📝 Assignment: {args.assignment}")
    if args.output:
        print(f"📄 Output: {args.output}")
    else:
        print(f"📁 Output directory: {args.output_dir}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Read input files
        logger.info("Reading input files...")
        requirements = []
        for req_file in args.requirement:
            requirement = load_requirement_from_json(req_file)
            requirements.append(requirement)
        
        with open(args.assignment, 'r', encoding='utf-8') as f:
            assignment_description = f.read().strip()
        
        # Log input files as artifacts
        for i, requirement in enumerate(requirements):
            req_json = json.dumps({
                "requirement": requirement["requirement"],
                "function": requirement["function"], 
                "type": requirement["type"].value
            }, indent=2)
            safe_log_call(mlflow_logger, "log_text", req_json, f"input_requirement_{i+1}.json")
        safe_log_call(mlflow_logger, "log_text", assignment_description, "input_assignment.txt")
        safe_log_call(mlflow_logger, "log_trace_step", "read_inputs", {
            "requirement_files": args.requirement,
            "assignment_file": args.assignment,
            "num_requirements": len(requirements),
            "assignment_length": len(assignment_description)
        }, step_number=0)
        
        # Generate Jinja2 template prompts using the core logic
        if num_requirements == 1:
            logger.info("Generating single Jinja2 template using core logic...")
            result = agent.generate_prompt(requirements[0], assignment_description)
            jinja_template = result["jinja_template"]
            examples = result["examples"]
            
            # Log the examples that were generated (for debugging)
            safe_log_call(mlflow_logger, "log_text", examples, "generated_examples.txt")
            
            # Save the template
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(jinja_template)
            
            # Save the state as JSON (excluding jinja_template)
            state_output_path = output_path.with_suffix('.json')
            state_data = {
                "requirement": {
                    "requirement": result["requirement"]["requirement"],
                    "function": result["requirement"]["function"],
                    "type": result["requirement"]["type"].value
                },
                "examples": result["examples"],
                "index": result["index"]
            }
            with open(state_output_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Log the generated template and state
            prompt_name = output_path.stem
            safe_log_call(mlflow_logger, "log_text", jinja_template, f"generated_templates/{output_path.name}")
            safe_log_call(mlflow_logger, "log_text", json.dumps(state_data, indent=2), f"generated_states/{state_output_path.name}")
            safe_log_call(mlflow_logger, "log_prompt_metrics", jinja_template, prompt_name)
            
            output_files = [output_path, state_output_path]
        else:
            logger.info(f"Generating {num_requirements} Jinja2 templates in parallel using core logic...")
            results = agent.generate_prompts_batch(requirements, assignment_description)
            print(results)
            # Save all templates and states
            output_files = []
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, result in enumerate(results):
                # Save the Jinja template
                template_output_path = output_dir / f"prompt_{i+1:02d}.jinja"
                with open(template_output_path, 'w', encoding='utf-8') as f:
                    f.write(result["jinja_template"])
                output_files.append(template_output_path)
                
                # Save the state as JSON (excluding jinja_template)
                state_output_path = output_dir / f"prompt_{i+1:02d}.json"
                state_data = {
                    "requirement": {
                        "requirement": result["requirement"]["requirement"],
                        "function": result["requirement"]["function"],
                        "type": result["requirement"]["type"].value
                    },
                    "examples": result["examples"],
                    "index": result["index"]
                }
                with open(state_output_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                output_files.append(state_output_path)
                
                # Log the examples, template, and state
                safe_log_call(mlflow_logger, "log_text", result["examples"], f"generated_examples_{i+1}.txt")
                safe_log_call(mlflow_logger, "log_text", result["jinja_template"], f"generated_templates/{template_output_path.name}")
                safe_log_call(mlflow_logger, "log_text", json.dumps(state_data, indent=2), f"generated_states/{state_output_path.name}")
                safe_log_call(mlflow_logger, "log_prompt_metrics", result["jinja_template"], f"prompt_{i+1}")
                
                print(f"  Generated prompt: {template_output_path.name} & {state_output_path.name}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log final metrics
        actual_templates_generated = num_requirements  # Use the actual number of requirements processed
        safe_log_call(mlflow_logger, "log_metrics", {
            "total_generation_time_seconds": total_time,
            "template_generation_rate": actual_templates_generated / total_time if total_time > 0 else 0,
            "num_templates_generated": actual_templates_generated
        })
        
        # Output results
        if num_requirements == 1:
            print(f"✅ Generated Jinja2 template: {output_files[0]}")
            print(f"✅ Generated state JSON: {output_files[1]}")
        else:
            print(f"✅ Generated {actual_templates_generated} Jinja2 templates and state files in parallel")
        print(f"⏱️  Generation time: {total_time:.2f} seconds")
        
        # Print summary
        print("\n📊 PROMPT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Requirement files: {', '.join([Path(req).name for req in args.requirement])}")
        print(f"Assignment file: {args.assignment}")
        if num_requirements == 1:
            print(f"Output template: {output_files[0]}")
            print(f"Output state: {output_files[1]}")
        else:
            print(f"Output directory: {args.output_dir}")
            print(f"Generated template pairs: {num_requirements}")
        print(f"Generation time: {total_time:.2f} seconds")
        
        if args.verbose and num_requirements == 1:
            print("\n📄 TEMPLATE PREVIEW")
            print("=" * 50)
            print("Template content:")
            print("-" * 30)
            print(jinja_template)
            print("-" * 30)
            print(f"Template size: {len(jinja_template)} characters")
        
    except Exception as e:
        # Log error metrics
        safe_log_call(mlflow_logger, "log_metric", "error_occurred", 1.0)
        safe_log_call(mlflow_logger, "log_text", str(e), "error_log.txt")
        print(f"❌ Error during prompt generation: {e}")
        raise e
    finally:
        safe_log_call(mlflow_logger, "end_run")

if __name__ == "__main__":
    main()

