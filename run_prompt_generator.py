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
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent

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
    parser.add_argument("--requirement", "-r", required=True, help="Path to requirement file (.txt)")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output", "-o", required=True, help="Output path for Jinja2 template (.jinja)")
    parser.add_argument("--config", default="src/config/assignment_config.yaml", help="Configuration file path")
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
    
    # Check if files exist
    if not os.path.exists(args.requirement):
        print(f"Error: Requirement file not found: {args.requirement}")
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
    
    # Run the prompt generator
    print("🚀 Starting prompt generation...")
    print(f"📋 Requirement: {args.requirement}")
    print(f"📝 Assignment: {args.assignment}")
    print(f"📄 Output: {args.output}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Generate Jinja2 template prompt
        output_path = agent.generate_prompt(
            requirement_file_path=args.requirement,
            assignment_file_path=args.assignment,
            output_file_path=args.output
        )
        
        end_time = time.time()
        
        # Output results
        print(f"✅ Generated Jinja2 template: {output_path}")
        print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
        
        # Print summary
        print("\n📊 PROMPT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Requirement file: {args.requirement}")
        print(f"Assignment file: {args.assignment}")
        print(f"Output template: {output_path}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        
        if args.verbose:
            print("\n📄 TEMPLATE PREVIEW")
            print("=" * 50)
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                    print("Template content:")
                    print("-" * 30)
                    print(template_content)
                    print("-" * 30)
                    print(f"Template size: {len(template_content)} characters")
            except Exception as e:
                print(f"Error reading template: {e}")
        
    except Exception as e:
        print(f"❌ Error during prompt generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

