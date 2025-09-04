#!/usr/bin/env python3
"""
Code Corrector Agent - Standalone Runner

This script allows running the code corrector agent independently
to analyze code against a requirement using Jinja2 template prompts.
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

from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.config import get_agent_config, load_config, load_langsmith_config

def main():
    """Main entry point for code corrector"""
    parser = argparse.ArgumentParser(description="Code Corrector Agent")
    parser.add_argument("--prompt", "-p", required=True, help="Path to Jinja2 template prompt (.jinja)")
    parser.add_argument("--code", "-c", required=True, help="Path to Python code file (.py or .txt)")
    parser.add_argument("--output", "-o", help="Output file path for analysis (optional)")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
    parser.add_argument("--context", help="Additional context for analysis (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--batch", action="store_true", help="Batch mode: analyze all .py and .txt files in code directory")
    parser.add_argument("--code-dir", help="Code directory for batch mode")
    parser.add_argument("--output-dir", help="Output directory for batch mode")
    
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
    print(f"🤖 Provider: {agent_config.get('provider', 'Unknown')}")
    print("-" * 50)
    
    # Check if files exist
    if not os.path.exists(args.prompt):
        print(f"Error: Prompt template file not found: {args.prompt}")
        sys.exit(1)
    
    if not args.batch and not os.path.exists(args.code):
        print(f"Error: Code file not found: {args.code}")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = CodeCorrectorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run the code corrector
    print("🚀 Starting code analysis...")
    print(f"📋 Prompt template: {args.prompt}")
    
    if args.batch:
        print(f"📁 Code directory: {args.code_dir}")
        print(f"📁 Output directory: {args.output_dir}")
    else:
        print(f"💻 Code file: {args.code}")
        if args.output:
            print(f"📄 Output file: {args.output}")
    
    if args.context:
        print(f"📝 Additional context: {args.context}")
    
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        if args.batch:
            # Batch mode: analyze all files in directory
            if not args.code_dir or not args.output_dir:
                print("Error: --code-dir and --output-dir are required for batch mode")
                sys.exit(1)
            
            analysis_files = agent.batch_analyze(
                prompt_template_path=args.prompt,
                code_directory=args.code_dir,
                output_directory=args.output_dir,
                additional_context=args.context
            )
            
            end_time = time.time()
            
            # Output results
            print(f"✅ Analyzed {len(analysis_files)} code files")
            print(f"⏱️  Total analysis time: {end_time - start_time:.2f} seconds")
            
            # Print summary
            print("\n📊 BATCH ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Prompt template: {args.prompt}")
            print(f"Code directory: {args.code_dir}")
            print(f"Output directory: {args.output_dir}")
            print(f"Files analyzed: {len(analysis_files)}")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            
            print(f"\n📋 Analysis files generated:")
            for i, file_path in enumerate(analysis_files, 1):
                file_name = Path(file_path).name
                print(f"  {i}. {file_name}")
            
        else:
            # Single file mode
            analysis = agent.analyze_code(
                prompt_template_path=args.prompt,
                code_file_path=args.code,
                output_file_path=args.output,
                additional_context=args.context
            )
            
            end_time = time.time()
            
            # Output results
            if args.output:
                print(f"✅ Analysis saved to: {args.output}")
            else:
                print("✅ Analysis completed")
            print(f"⏱️  Analysis time: {end_time - start_time:.2f} seconds")
            
            # Print summary
            print("\n📊 CODE ANALYSIS SUMMARY")
            print("=" * 50)
            print(f"Prompt template: {args.prompt}")
            print(f"Code file: {args.code}")
            if args.output:
                print(f"Output file: {args.output}")
            print(f"Analysis time: {end_time - start_time:.2f} seconds")
            
            if args.verbose:
                print("\n📄 ANALYSIS PREVIEW")
                print("=" * 50)
                print("Analysis content:")
                print("-" * 30)
                print(analysis)
                print("-" * 30)
                print(f"Analysis size: {len(analysis)} characters")
        
    except Exception as e:
        print(f"❌ Error during code analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

