#!/usr/bin/env python3
"""
Requirement Generator Agent - Standalone Runner

This script allows running the requirement generator agent independently
to generate individual requirement files from assignment descriptions.
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
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for requirement files")
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
    
    # Check if assignment file exists
    if not os.path.exists(args.assignment):
        print(f"Error: Assignment file not found: {args.assignment}")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = RequirementGeneratorAgent(config)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run the requirement generator
    print("🚀 Starting requirement generation...")
    print(f"📝 Assignment: {args.assignment}")
    print(f"📁 Output directory: {args.output_dir}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Generate requirements
        requirement_files = agent.generate_requirements(
            assignment_file_path=args.assignment,
            output_directory=args.output_dir
        )
        
        end_time = time.time()
        
        # Output results
        print(f"✅ Generated {len(requirement_files)} requirement files")
        print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
        
        # Print summary
        print("\n📊 REQUIREMENT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Assignment file: {args.assignment}")
        print(f"Output directory: {args.output_dir}")
        print(f"Number of requirements: {len(requirement_files)}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        
        print(f"\n📋 Generated requirement files:")
        for i, file_path in enumerate(requirement_files, 1):
            file_name = Path(file_path).name
            print(f"  {i}. {file_name}")
        
        if args.verbose:
            print("\n📄 DETAILED FILE LIST")
            print("=" * 50)
            for file_path in requirement_files:
                print(f"File: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"Content preview: {content[:100]}...")
                except Exception as e:
                    print(f"Error reading file: {e}")
                print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error during requirement generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

