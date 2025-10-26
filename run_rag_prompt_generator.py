#!/usr/bin/env python3
"""
RAG-Enhanced Prompt Generator Agent - Standalone Runner

This script allows running the RAG-enhanced prompt generator agent independently
to generate Jinja2 template prompts from requirement files using course theory.
It handles both single and multiple requirements using RAG enhancement.
"""

import sys
import os
import json
import yaml
import argparse
import time
import logging
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.rag_prompt_generator.rag_prompt_generator import RAGPromptGeneratorAgent
from src.agents.rag_prompt_generator.config import USE_RAG
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

async def main():
    """Main entry point for RAG-enhanced prompt generator"""
    parser = argparse.ArgumentParser(description="RAG-Enhanced Prompt Generator Agent")
    parser.add_argument("--requirement", "-r", nargs="+", required=True, help="Path(s) to requirement file(s) (.json)")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated templates")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--ingest-notebooks", action="store_true", help="Ingest notebooks before generating prompts")
    parser.add_argument("--dataset", default="python", choices=["python", "haskell"], help="Dataset to use for RAG")
    
    args = parser.parse_args()
    
    # Check if RAG is enabled
    if not USE_RAG:
        print("❌ RAG is not enabled. Set USE_RAG=true environment variable to use RAG functionality.")
        sys.exit(1)
    
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
    print(f"🧠 RAG Enhanced: {USE_RAG}")
    print(f"📚 Dataset: {args.dataset}")
    print("🔧 RAG Mode: ENABLED")
    print("📚 Course theory integration: ACTIVE")
    print("-" * 50)
    
    # Get number of requirements
    num_requirements = len(args.requirement)
    
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
        agent = RAGPromptGeneratorAgent(config)
        await agent.initialize()
    except Exception as e:
        print(f"Error initializing RAG agent: {e}")
        sys.exit(1)
    
    # Ingest notebooks if requested
    if args.ingest_notebooks:
        print(f"📚 Ingesting {args.dataset} notebooks...")
        try:
            result = await agent.rag_system.ingest_notebooks(args.dataset)
            print(f"✅ Ingested {result['count']} chunks from {args.dataset} dataset")
        except Exception as e:
            print(f"❌ Error ingesting notebooks: {e}")
            sys.exit(1)
    
    # Start MLflow run for RAG prompt generation
    mlflow_logger = get_mlflow_logger()
    requirement_files_str = ", ".join([Path(req).name for req in args.requirement])
    output_info = f"{num_requirements}_files_rag"
    
    safe_log_call(mlflow_logger, "start_run",
        run_name="rag_prompt_generation",
        tags={
            "agent": "rag_prompt_generator",
            "requirement_files": requirement_files_str,
            "assignment_file": Path(args.assignment).name,
            "output_info": output_info,
            "num_requirements": str(num_requirements),
            "rag_enhanced": "true",
            "dataset": args.dataset
        }
    )
    
    # Run the RAG prompt generator
    print("🚀 Starting RAG-enhanced prompt generation...")
    print(f"📋 Requirements: {', '.join([Path(req).name for req in args.requirement])}")
    print(f"📝 Assignment: {args.assignment}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🧠 RAG Enhanced: True")
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
            "assignment_length": len(assignment_description),
            "rag_enhanced": True,
            "dataset": args.dataset
        }, step_number=0)
        
        # Generate RAG-enhanced Jinja2 template prompts
        logger.info(f"Generating {num_requirements} RAG-enhanced Jinja2 template(s)...")
        results = agent.generate_prompts_batch(requirements, assignment_description)
        
        # Save all templates and states
        output_files = []
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            # Save the Jinja template
            template_output_path = output_dir / f"rag_prompt_{i+1:02d}.jinja"
            with open(template_output_path, 'w', encoding='utf-8') as f:
                f.write(result["jinja_template"])
            output_files.append(template_output_path)
            
            # Save the state as JSON (excluding jinja_template)
            state_output_path = output_dir / f"rag_prompt_{i+1:02d}.json"
            state_data = {
                "requirement": {
                    "requirement": result["requirement"]["requirement"],
                    "function": result["requirement"]["function"],
                    "type": result["requirement"]["type"].value
                },
                "examples": result["examples"],
                "index": result["index"],
                "extra": result.get("extra", {}),
                "rag_enhanced": True,
                "dataset": args.dataset
            }
            with open(state_output_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            output_files.append(state_output_path)
            
            # Log the examples, template, and state
            safe_log_call(mlflow_logger, "log_text", result["examples"], f"rag_generated_examples_{i+1}.txt")
            safe_log_call(mlflow_logger, "log_text", result["jinja_template"], f"rag_generated_templates/{template_output_path.name}")
            safe_log_call(mlflow_logger, "log_text", json.dumps(state_data, indent=2), f"rag_generated_states/{state_output_path.name}")
            safe_log_call(mlflow_logger, "log_prompt_metrics", result["jinja_template"], f"rag_prompt_{i+1}")
            
            print(f"  Generated RAG-enhanced prompt: {template_output_path.name} & {state_output_path.name}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log final metrics
        actual_templates_generated = num_requirements
        safe_log_call(mlflow_logger, "log_metrics", {
            "total_generation_time_seconds": total_time,
            "template_generation_rate": actual_templates_generated / total_time if total_time > 0 else 0,
            "num_templates_generated": actual_templates_generated,
            "rag_enhanced": True,
            "dataset": args.dataset
        })
        
        # Output results
        print(f"✅ Generated {actual_templates_generated} RAG-enhanced Jinja2 template(s) and state file(s)")
        print(f"⏱️  Generation time: {total_time:.2f} seconds")
        
        # Print summary
        print("\n📊 RAG-ENHANCED PROMPT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Requirement files: {', '.join([Path(req).name for req in args.requirement])}")
        print(f"Assignment file: {args.assignment}")
        print(f"Output directory: {args.output_dir}")
        print(f"Generated template pairs: {num_requirements}")
        print(f"Generation time: {total_time:.2f} seconds")
        print(f"RAG Enhanced: True")
        print(f"Dataset: {args.dataset}")
        
        if args.verbose and num_requirements == 1:
            print("\n📄 RAG-ENHANCED TEMPLATE PREVIEW")
            print("=" * 50)
            print("Template content:")
            print("-" * 30)
            print(results[0]["jinja_template"])
            print("-" * 30)
            print(f"Template size: {len(results[0]['jinja_template'])} characters")
            print(f"RAG enhanced: {results[0].get('extra', {}).get('rag_enhanced', False)}")
        
    except Exception as e:
        # Log error metrics
        safe_log_call(mlflow_logger, "log_metric", "error_occurred", 1.0)
        safe_log_call(mlflow_logger, "log_text", str(e), "error_log.txt")
        print(f"❌ Error during RAG prompt generation: {e}")
        raise e
    finally:
        safe_log_call(mlflow_logger, "end_run")

if __name__ == "__main__":
    asyncio.run(main())
