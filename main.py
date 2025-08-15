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
    
    # Run the evaluation pipeline
    print("🚀 Starting assignment evaluation pipeline...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"💻 Code: {args.code}")
    print(f"📁 Output directory: {output_dir}")
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
            requirement_files = requirement_agent.generate_requirements(
                assignment_file_path=args.assignment,
                output_directory=str(requirements_dir)
            )
            print(f"✅ Generated {len(requirement_files)} requirement files")
        
        # Step 2: Generate prompts for each requirement
        prompts_dir = output_dir / "prompts"
        if args.skip_prompts and args.prompts_dir:
            prompts_dir = Path(args.prompts_dir)
            print(f"📝 Using existing prompts from: {prompts_dir}")
        else:
            print("📝 Step 2: Generating prompts...")
            prompts_dir.mkdir(exist_ok=True)
            
            # Find requirement files
            requirement_files = list(requirements_dir.glob("*.txt"))
            prompt_files = []
            
            for req_file in requirement_files:
                prompt_file = prompts_dir / f"{req_file.stem}.jinja"
                prompt_path = prompt_agent.generate_prompt(
                    requirement_file_path=str(req_file),
                    assignment_file_path=args.assignment,
                    output_file_path=str(prompt_file)
                )
                prompt_files.append(prompt_path)
                print(f"  Generated prompt: {prompt_file.name}")
            
            print(f"✅ Generated {len(prompt_files)} prompt templates")
        
        # Step 3: Analyze code against each requirement
        print("🔍 Step 3: Analyzing code...")
        analysis_dir = output_dir / "analyses"
        analysis_dir.mkdir(exist_ok=True)
        
        # Find prompt files
        prompt_files = list(prompts_dir.glob("*.jinja"))
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
        
        # Output results
        print(f"✅ Completed analysis of {len(analysis_files)} requirements")
        print(f"⏱️  Total pipeline time: {end_time - start_time:.2f} seconds")
        
        # Print summary
        print("\n📊 EVALUATION PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Assignment file: {args.assignment}")
        print(f"Code file: {args.code}")
        print(f"Output directory: {output_dir}")
        print(f"Requirements generated: {len(list(requirements_dir.glob('*.txt')))}")
        print(f"Prompts generated: {len(list(prompts_dir.glob('*.jinja')))}")
        print(f"Analyses completed: {len(analysis_files)}")
        print(f"Total pipeline time: {end_time - start_time:.2f} seconds")
        
        print(f"\n📋 Generated files:")
        print(f"  Requirements: {requirements_dir}")
        for req_file in requirements_dir.glob("*.txt"):
            print(f"    - {req_file.name}")
        
        print(f"  Prompts: {prompts_dir}")
        for prompt_file in prompts_dir.glob("*.jinja"):
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
        print(f"❌ Error during evaluation pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
