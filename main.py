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
import argparse
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent
from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.core.mlflow_utils import mlflow_logger
from src.config import get_agent_config, load_config, load_langsmith_config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Assignment Evaluation System")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--code", "-c", required=True, help="Path to student code file (.py or .txt)")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory for all generated files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip requirement generation (use existing)")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip prompt generation (use existing)")
    parser.add_argument("--requirements-dir", help="Directory with existing requirement files")
    parser.add_argument("--prompts-dir", help="Directory with existing prompt files")
    parser.add_argument("--context", help="Additional context for code analysis")
    
    # New arguments for selective execution
    parser.add_argument("--only-requirements", action="store_true", help="Only generate requirements and exit")
    parser.add_argument("--only-prompts", action="store_true", help="Only generate prompts and exit")
    parser.add_argument("--only-analysis", action="store_true", help="Only run code analysis and exit")
    parser.add_argument("--requirement-index", type=int, help="Work only with the requirement at this index (0-based)")
    parser.add_argument("--requirement-name", help="Work only with the requirement with this specific name")
    parser.add_argument("--generate-examples", action="store_true", help="Generate examples for requirements")
    parser.add_argument("--single-requirement-pipeline", type=int, help="Execute complete pipeline for single requirement: generate requirements, take requirement at index, generate prompt, and run analysis")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Error: Could not load configuration")
        sys.exit(1)
    load_langsmith_config()
    
    # Print model information
    print("🤖 Model Configuration:")
    print(f"   Global: {config.get('model_name', 'Unknown')} ({config.get('provider', 'Unknown')})")
    
    # Print agent-specific model information
    agents_config = config.get('agents', {})
    for agent_name, agent_config in agents_config.items():
        if agent_config.get('enabled', True):
            model_name = agent_config.get('model_name', config.get('model_name', 'Unknown'))
            provider = agent_config.get('provider', config.get('provider', 'Unknown'))
            print(f"   {agent_name.replace('_', ' ').title()}: {model_name} ({provider})")
    
    print("-" * 50)
    
    # Check if input files exist
    if not os.path.exists(args.assignment):
        print(f"Error: Assignment file not found: {args.assignment}")
        sys.exit(1)
    
    if not os.path.exists(args.code):
        print(f"Error: Code file not found: {args.code}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize agents
    try:
        requirement_agent = RequirementGeneratorAgent(config)
        prompt_agent = PromptGeneratorAgent(config)
        code_agent = CodeCorrectorAgent(config)
    except Exception as e:
        print(f"Error initializing agents: {e}")
        sys.exit(1)
    
    # Start MLflow run for the entire pipeline
    pipeline_run_id = mlflow_logger.start_run(
        run_name="assignment_evaluation_pipeline",
        tags={
            "pipeline": "assignment_evaluation",
            "assignment_file": Path(args.assignment).name,
            "code_file": Path(args.code).name,
            "output_directory": str(output_dir),
            "skip_requirements": args.skip_requirements,
            "skip_prompts": args.skip_prompts
        }
    )
    
    # Log pipeline parameters
    mlflow_logger.log_params({
        "assignment_file": args.assignment,
        "code_file": args.code,
        "output_directory": str(output_dir),
        "config_file": args.config,
        "skip_requirements": args.skip_requirements,
        "skip_prompts": args.skip_prompts,
        "verbose": args.verbose,
        "additional_context": args.context is not None
    })
    
    # Determine execution mode and handle single requirement pipeline
    execution_mode = "full_pipeline"
    if args.only_requirements:
        execution_mode = "requirements_only"
    elif args.only_prompts:
        execution_mode = "prompts_only"
    elif args.only_analysis:
        execution_mode = "analysis_only"
    
    # Handle single requirement pipeline mode
    if args.single_requirement_pipeline is not None:
        execution_mode = "single_requirement_pipeline"
        print(f"🎯 Single requirement pipeline mode: will process only requirement at index {args.single_requirement_pipeline}")
    
    # Run the evaluation pipeline
    print("🚀 Starting assignment evaluation pipeline...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"💻 Code: {args.code}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Execution mode: {execution_mode}")
    if args.requirement_index is not None:
        print(f"🎯 Working with requirement index: {args.requirement_index}")
    if args.requirement_name:
        print(f"🎯 Working with requirement: {args.requirement_name}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Step 1: Generate requirements
        requirements_dir = output_dir / "requirements"
        if args.skip_requirements and args.requirements_dir:
            requirements_dir = Path(args.requirements_dir)
            print(f"📋 Using existing requirements from: {requirements_dir}")
        else:
            print("📋 Step 1: Generating requirements...")
            requirements_dir.mkdir(exist_ok=True)
            
            # Log pipeline trace step
            mlflow_logger.log_trace_step("pipeline_step_1", {
                "step": "requirement_generation",
                "assignment_file": args.assignment,
                "output_directory": str(requirements_dir)
            }, step_number=1)
            
            requirement_files = requirement_agent.generate_requirements(
                assignment_file_path=args.assignment,
                output_directory=str(requirements_dir)
            )
            print(f"✅ Generated {len(requirement_files)} requirement files")
            
            # Generate examples if requested
            if args.generate_examples:
                print("📚 Generating examples for requirements...")
                examples_dir = output_dir / "examples"
                examples_dir.mkdir(exist_ok=True)
                
                # Get the list of requirement files that were just generated (search recursively)
                requirement_files_list = list(requirements_dir.rglob("*.txt"))
                
                for req_file in requirement_files_list:
                    example_file = examples_dir / f"example_{req_file.stem}.txt"
                    # Here you would call a method to generate examples
                    # For now, we'll just create a placeholder
                    with open(example_file, 'w', encoding='utf-8') as f:
                        f.write(f"Examples for requirement: {req_file.stem}\n")
                        f.write("=" * 50 + "\n")
                        f.write("Examples would be generated here based on the requirement content.\n")
                    print(f"  Generated example: {example_file.name}")
                
                print(f"✅ Generated {len(requirement_files_list)} example files")
        
        # If only requirements mode, exit here
        if execution_mode == "requirements_only":
            print("✅ Requirements generation completed. Exiting as requested.")
            return
        
        # Get list of requirement files (search recursively in subdirectories)
        requirement_files = list(requirements_dir.rglob("*.txt"))
        if not requirement_files:
            print("❌ No requirement files found!")
            print(f"   Searched in: {requirements_dir}")
            print(f"   Available files: {list(requirements_dir.iterdir())}")
            return
        
        # Filter requirements if specific index or name is requested
        if args.requirement_index is not None or args.requirement_name or args.single_requirement_pipeline is not None:
            target_index = args.requirement_index if args.requirement_index is not None else args.single_requirement_pipeline
            
            if target_index is not None:
                if target_index >= len(requirement_files):
                    print(f"❌ Requirement index {target_index} out of range. Only {len(requirement_files)} requirements exist.")
                    return
                selected_requirements = [requirement_files[target_index]]
                print(f"🎯 Working with requirement {target_index}: {selected_requirements[0].name}")
            else:
                # Find by name
                selected_requirements = [f for f in requirement_files if args.requirement_name in f.name]
                if not selected_requirements:
                    print(f"❌ No requirement found with name containing '{args.requirement_name}'")
                    return
                print(f"🎯 Working with requirement: {selected_requirements[0].name}")
        else:
            selected_requirements = requirement_files
        
        # Step 2: Generate prompts for selected requirements
        prompts_dir = output_dir / "prompts"
        if args.skip_prompts and args.prompts_dir:
            prompts_dir = Path(args.prompts_dir)
            print(f"📝 Using existing prompts from: {prompts_dir}")
        else:
            print("📝 Step 2: Generating prompts...")
            prompts_dir.mkdir(exist_ok=True)
            
            # Log pipeline trace step
            mlflow_logger.log_trace_step("pipeline_step_2", {
                "step": "prompt_generation",
                "requirements_count": len(selected_requirements),
                "output_directory": str(prompts_dir)
            }, step_number=2)
            
            # Read assignment description once
            with open(args.assignment, 'r', encoding='utf-8') as f:
                assignment_description = f.read().strip()
            
            # Read all requirements at once
            requirements = []
            for req_file in selected_requirements:
                with open(req_file, 'r', encoding='utf-8') as f:
                    requirement_content = f.read().strip()
                    requirements.append(requirement_content)
            
            # Generate all prompts in parallel using batch processing
            results = prompt_agent.generate_prompts_batch(requirements, assignment_description)
            
            # Save all generated templates
            prompt_files = []
            for i, result in enumerate(results):
                req_file = selected_requirements[i]
                prompt_file = prompts_dir / f"{req_file.stem}.jinja"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(result["jinja_template"])
                prompt_files.append(prompt_file)
                print(f"  Generated prompt: {prompt_file.name}")
            
            print(f"✅ Generated {len(prompt_files)} prompt templates")
        
        # If only prompts mode, exit here
        if execution_mode == "prompts_only":
            print("✅ Prompt generation completed. Exiting as requested.")
            return
        
        # Get list of prompt files for selected requirements
        prompt_files = []
        for req_file in selected_requirements:
            prompt_file = prompts_dir / f"{req_file.stem}.jinja"
            if prompt_file.exists():
                prompt_files.append(prompt_file)
        
        if not prompt_files:
            print("❌ No prompt files found!")
            return
        
        # Step 3: Analyze code against selected requirements
        print("🔍 Step 3: Analyzing code...")
        analysis_dir = output_dir / "analyses"
        analysis_dir.mkdir(exist_ok=True)
        
        # Log pipeline trace step
        mlflow_logger.log_trace_step("pipeline_step_3", {
            "step": "code_analysis",
            "prompts_count": len(prompt_files),
            "code_file": args.code,
            "output_directory": str(analysis_dir)
        }, step_number=3)
        
        analysis_files = []
        
        for prompt_file in prompt_files:
            analysis_file = analysis_dir / f"analysis_{prompt_file.stem}.txt"
            analysis = code_agent.analyze_code(
                prompt_template_path=str(prompt_file),
                code_file_path=args.code,
                output_file_path=str(analysis_file),
                additional_context=args.context
            )
            analysis_files.append(str(analysis_file))
            print(f"  Analyzed with {prompt_file.name}")
        
        end_time = time.time()
        total_pipeline_time = end_time - start_time
        
        # Log pipeline completion metrics
        mlflow_logger.log_metrics({
            "total_pipeline_time_seconds": total_pipeline_time,
            "requirements_processed": len(selected_requirements),
            "prompts_generated": len(prompt_files),
            "analyses_completed": len(analysis_files),
            "pipeline_success_rate": 1.0
        })
        
        # Log the entire output directory as artifacts
        mlflow_logger.log_artifacts(str(output_dir), "pipeline_outputs")
        
        # Log final pipeline trace step
        mlflow_logger.log_trace_step("pipeline_complete", {
            "total_time": total_pipeline_time,
            "total_files_generated": len(analysis_files),
            "success": True
        }, step_number=4)
        
        # Output results
        print(f"✅ Completed analysis of {len(analysis_files)} requirements")
        print(f"⏱️  Total pipeline time: {total_pipeline_time:.2f} seconds")
        
        # Print summary
        print("\n📊 EVALUATION PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Assignment file: {args.assignment}")
        print(f"Code file: {args.code}")
        print(f"Output directory: {output_dir}")
        print(f"Requirements processed: {len(selected_requirements)}")
        print(f"Prompts generated: {len(prompt_files)}")
        print(f"Analyses completed: {len(analysis_files)}")
        print(f"Total pipeline time: {total_pipeline_time:.2f} seconds")
        
        print(f"\n📋 Generated files:")
        print(f"  Requirements: {requirements_dir}")
        for req_file in selected_requirements:
            print(f"    - {req_file.name}")
        
        print(f"  Prompts: {prompts_dir}")
        for prompt_file in prompt_files:
            print(f"    - {prompt_file.name}")
        
        print(f"  Analyses: {analysis_dir}")
        for analysis_file in analysis_dir.glob("*.txt"):
            print(f"    - {analysis_file.name}")
        
        if args.verbose:
            print("\n📄 ANALYSIS PREVIEWS")
            print("=" * 50)
            for analysis_file in analysis_files[:3]:  # Show first 3 analyses
                print(f"\nAnalysis: {Path(analysis_file).name}")
                print("-" * 30)
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(content[:500] + "..." if len(content) > 500 else content)
                except Exception as e:
                    print(f"Error reading analysis: {e}")
        
    except Exception as e:
        # Log error metrics
        mlflow_logger.log_metric("pipeline_error_occurred", 1.0)
        mlflow_logger.log_text(str(e), "pipeline_error_log.txt")
        print(f"❌ Error during evaluation pipeline: {e}")
        sys.exit(1)
    finally:
        # End the MLflow run
        mlflow_logger.end_run()

if __name__ == "__main__":
    main()
