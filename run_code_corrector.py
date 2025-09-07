#!/usr/bin/env python3
"""
Code Corrector Agent - Standalone Runner

This script allows running the code corrector agent independently
to analyze code against generated prompts (JSON files with Jinja templates).
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

from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.config import get_agent_config, load_config, load_langsmith_config
from src.models import GeneratedPrompt, Submission, Correction, PromptType

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

def load_generated_prompt(prompt_json_path: str) -> GeneratedPrompt:
    """Load a GeneratedPrompt from a JSON file"""
    try:
        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert the type string to PromptType enum if needed
        requirement = data["requirement"]
        if isinstance(requirement.get("type"), str):
            requirement["type"] = PromptType(requirement["type"])
        
        # Convert the loaded data to GeneratedPrompt format
        generated_prompt: GeneratedPrompt = {
            "requirement": requirement,
            "examples": data.get("examples", ""),
            "jinja_template": "",  # Will be loaded from .jinja file
            "index": data.get("index", 0),
        }
        
        return generated_prompt
    except Exception as e:
        raise RuntimeError(f"Failed to load generated prompt from {prompt_json_path}: {e}")

def load_jinja_template(jinja_path: str) -> str:
    """Load Jinja template content from file"""
    try:
        with open(jinja_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load Jinja template from {jinja_path}: {e}")

def save_corrections_state(corrections: list, output_path: str, metadata: dict = None):
    """Save corrections and metadata to JSON file"""
    output_data = {
        "corrections": [
            {
                "requirement": {
                    "requirement": correction["requirement"]["requirement"],
                    "function": correction["requirement"]["function"], 
                    "type": correction["requirement"]["type"].value
                },
                "result": correction["result"]
            }
            for correction in corrections
        ],
        "metadata": metadata or {},
        "timestamp": time.time()
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save corrections to {output_path}: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for code corrector"""
    parser = argparse.ArgumentParser(description="Code Corrector Agent")
    parser.add_argument("--prompt", "-p", nargs="+", required=True, help="Path(s) to generated prompt JSON file(s)")
    parser.add_argument("--code", "-c", required=True, help="Path to Python code file (.py or .txt)")
    parser.add_argument("--output", "-o", help="Output JSON file path for single correction")
    parser.add_argument("--output-dir", help="Output directory for multiple corrections")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--assignment", "-a", help="Assignment description (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()
    
    # Print model information
    agent_config = get_agent_config(config, 'code_corrector')
    print(f"🤖 Using model: {agent_config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {agent_config.get('provider', 'Unknown')}")
    print("-" * 50)
    
    # Validate arguments based on number of prompts
    num_prompts = len(args.prompt)
    
    if num_prompts == 1:
        if not args.output:
            print("Error: --output is required when processing a single prompt")
            sys.exit(1)
    else:
        if not args.output_dir:
            print("Error: --output-dir is required when processing multiple prompts")
            sys.exit(1)
    
    # Check if files exist
    for prompt_file in args.prompt:
        if not os.path.exists(prompt_file):
            print(f"Error: Prompt JSON file not found: {prompt_file}")
            sys.exit(1)
        
        # Check if corresponding Jinja template exists
        jinja_file = Path(prompt_file).with_suffix('.jinja')
        if not jinja_file.exists():
            print(f"Error: Jinja template file not found: {jinja_file}")
            sys.exit(1)
    
    if not os.path.exists(args.code):
        print(f"Error: Code file not found: {args.code}")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = CodeCorrectorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Start MLflow run for code correction
    mlflow_logger = get_mlflow_logger()
    prompt_files_str = ", ".join([Path(prompt).name for prompt in args.prompt])
    output_info = Path(args.output).name if args.output else f"{num_prompts}_corrections"
    
    safe_log_call(mlflow_logger, "start_run",
        run_name="code_correction",
        tags={
            "agent": "code_corrector",
            "prompt_files": prompt_files_str,
            "code_file": Path(args.code).name,
            "output_info": output_info,
            "num_prompts": str(num_prompts)
        }
    )
    
    # Run the code corrector
    print("🚀 Starting code correction...")
    print(f"📋 Prompts: {', '.join([Path(prompt).name for prompt in args.prompt])}")
    print(f"💻 Code file: {args.code}")
    if args.output:
        print(f"📄 Output: {args.output}")
    else:
        print(f"📁 Output directory: {args.output_dir}")
    if args.assignment:
        print(f"📝 Assignment description: {args.assignment}")
    print("-" * 50)
    

    try:
        start_time = time.time()
        
        # Read input files
        logger.info("Reading input files...")
        generated_prompts = []
        for prompt_file in args.prompt:
            generated_prompt = load_generated_prompt(prompt_file)
            
            # Load corresponding Jinja template
            jinja_file = Path(prompt_file).with_suffix('.jinja')
            generated_prompt["jinja_template"] = load_jinja_template(str(jinja_file))
            generated_prompts.append(generated_prompt)
        
        # Read student code
        with open(args.code, 'r', encoding='utf-8') as f:
            student_code = f.read().strip()
        
        submission: Submission = {"code": student_code}
        
        # Log input files as artifacts
        for i, generated_prompt in enumerate(generated_prompts):
            prompt_json = json.dumps({
                "requirement": {
                    "requirement": generated_prompt["requirement"]["requirement"],
                    "function": generated_prompt["requirement"]["function"], 
                    "type": generated_prompt["requirement"]["type"].value
                },
                "examples": generated_prompt["examples"],
                "index": generated_prompt["index"]
            }, indent=2)
            safe_log_call(mlflow_logger, "log_text", prompt_json, f"input_prompt_{i+1}.json")
            safe_log_call(mlflow_logger, "log_text", generated_prompt["jinja_template"], f"input_template_{i+1}.jinja")
        safe_log_call(mlflow_logger, "log_text", student_code, "input_code.py")
        safe_log_call(mlflow_logger, "log_trace_step", "read_inputs", {
            "prompt_files": args.prompt,
            "code_file": args.code,
            "num_prompts": len(generated_prompts),
            "code_length": len(student_code)
        }, step_number=0)
        
        # Generate corrections using the core logic
        if num_prompts == 1:
            logger.info("Generating single correction using core logic...")
            correction = agent.correct_code(generated_prompts[0], submission, args.assignment or "")
            
            # Save the correction
            metadata = {
                "prompt_file": args.prompt[0],
                "code_file": args.code,
                "assignment_description": args.assignment or ""
            }
            save_corrections_state([correction], args.output, metadata)
            
            # Log the correction
            safe_log_call(mlflow_logger, "log_text", correction["result"], "generated_correction.txt")
            safe_log_call(mlflow_logger, "log_text", json.dumps({
                "requirement": {
                    "requirement": correction["requirement"]["requirement"],
                    "function": correction["requirement"]["function"], 
                    "type": correction["requirement"]["type"].value
                },
                "result": correction["result"]
            }, indent=2), "generated_correction.json")
            
            output_files = [args.output]
        else:
            logger.info(f"Generating {num_prompts} corrections in parallel using core logic...")
            corrections = agent.correct_code_batch(generated_prompts, submission, args.assignment or "")
            
            # Save all corrections
            output_files = []
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual corrections and combined
            for i, correction in enumerate(corrections):
                # Save individual correction
                individual_output_path = output_dir / f"correction_{i+1:02d}.json"
                metadata = {
                    "prompt_file": args.prompt[i],
                    "code_file": args.code,
                    "assignment_description": args.assignment or ""
                }
                save_corrections_state([correction], str(individual_output_path), metadata)
                output_files.append(individual_output_path)
                
                # Log the correction
                safe_log_call(mlflow_logger, "log_text", correction["result"], f"generated_correction_{i+1}.txt")
                safe_log_call(mlflow_logger, "log_text", json.dumps({
                    "requirement": {
                        "requirement": correction["requirement"]["requirement"],
                        "function": correction["requirement"]["function"], 
                        "type": correction["requirement"]["type"].value
                    },
                    "result": correction["result"]
                }, indent=2), f"generated_corrections/{individual_output_path.name}")
                
                print(f"  Generated correction: {individual_output_path.name}")
            
            # Save combined corrections
            combined_output_path = output_dir / "all_corrections.json"
            combined_metadata = {
                "prompt_files": args.prompt,
                "code_file": args.code,
                "num_corrections": len(corrections),
                "assignment_description": args.assignment or ""
            }
            save_corrections_state(corrections, str(combined_output_path), combined_metadata)
            output_files.append(combined_output_path)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log final metrics
        actual_corrections_generated = len(corrections) if num_prompts > 1 else 1
        safe_log_call(mlflow_logger, "log_metrics", {
            "total_correction_time_seconds": total_time,
            "correction_generation_rate": actual_corrections_generated / total_time if total_time > 0 else 0,
            "num_corrections_generated": actual_corrections_generated
        })
        
        # Output results
        if num_prompts == 1:
            print(f"✅ Generated correction: {output_files[0]}")
        else:
            print(f"✅ Generated {actual_corrections_generated} corrections in parallel")
            print(f"✅ Combined corrections saved to: {output_files[-1]}")
        print(f"⏱️  Correction time: {total_time:.2f} seconds")
        
        # Print summary
        print("\n📊 CODE CORRECTION SUMMARY")
        print("=" * 50)
        print(f"Prompt files: {', '.join([Path(prompt).name for prompt in args.prompt])}")
        print(f"Code file: {args.code}")
        if num_prompts == 1:
            print(f"Output file: {output_files[0]}")
        else:
            print(f"Output directory: {args.output_dir}")
            print(f"Individual corrections: {num_prompts}")
        print(f"Correction time: {total_time:.2f} seconds")
        
        if args.verbose and num_prompts == 1:
            print("\n📄 CORRECTION PREVIEW")
            print("=" * 50)
            print("Correction result:")
            print("-" * 30)
            print(correction["result"])
            print("-" * 30)
            print(f"Result size: {len(correction['result'])} characters")
        
    except Exception as e:
        # Log error metrics
        safe_log_call(mlflow_logger, "log_metric", "error_occurred", 1.0)
        safe_log_call(mlflow_logger, "log_text", str(e), "error_log.txt")
        print(f"❌ Error during code correction: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise e
    finally:
        safe_log_call(mlflow_logger, "end_run")

if __name__ == "__main__":
    main()
