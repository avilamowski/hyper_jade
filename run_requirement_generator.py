#!/usr/bin/env python3
"""
Requirement Generator Agent - Standalone Runner

This script allows running the requirement generator agent independently
to generate individual requirement files from assignment descriptions.
Each requirement is saved as a separate JSON file containing structured data.
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
from src.config import get_agent_config, load_config, load_langsmith_config

def main():
    """Main entry point for requirement generator"""
    parser = argparse.ArgumentParser(description="Requirement Generator Agent")
    parser.add_argument("--assignment", "-a", required=True, help="Path to assignment description file (.txt)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for requirement JSON files")
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
    agent_config = get_agent_config(config, 'requirement_generator')
    print(f"🤖 Using model: {agent_config.get('model_name', 'Unknown')}")
    print(f"🔧 Provider: {agent_config.get('provider', 'Unknown')}")
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
    print(f"📁 Base output directory: {args.output_dir}")
    
    # Show the actual output directory that will be created
    model_name = agent_config.get("model_name", "unknown")
    safe_model_name = model_name.replace(":", "_")
    actual_output_dir = Path(args.output_dir) / safe_model_name
    print(f"📁 Actual output directory: {actual_output_dir}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Read the assignment description
        with open(args.assignment, 'r', encoding='utf-8') as f:
            assignment_description = f.read().strip()
        
        # Generate requirements using the LangGraph agent
        requirements = agent.generate_requirements(assignment_description)
        
        # Create output directory
        actual_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each requirement as a separate JSON file
        requirement_files = []
        for i, requirement in enumerate(requirements, 1):
            filename = f"requirement_{i:02d}.json"
            file_path = actual_output_dir / filename
            
            # Convert requirement to a serializable dictionary
            requirement_data = {
                "requirement": requirement["requirement"],
                "function": requirement["function"],
                "type": requirement["type"].value if hasattr(requirement["type"], 'value') else str(requirement["type"])
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(requirement_data, f, indent=2, ensure_ascii=False)
            requirement_files.append(str(file_path))
        
        end_time = time.time()
        
        # Output results
        print(f"✅ Generated {len(requirement_files)} requirement JSON files")
        print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
        
        # Print summary
        print("\n📊 REQUIREMENT GENERATION SUMMARY")
        print("=" * 50)
        print(f"Assignment file: {args.assignment}")
        print(f"Base output directory: {args.output_dir}")
        print(f"Actual output directory: {actual_output_dir}")
        print(f"Number of requirements: {len(requirement_files)}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")
        
        print(f"\n📋 Generated requirement JSON files:")
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
                        content = json.load(f)
                        print(f"Requirement: {content.get('requirement', 'N/A')[:100]}...")
                        print(f"Function: {content.get('function', 'N/A')}")
                        print(f"Type: {content.get('type', 'N/A')}")
                except Exception as e:
                    print(f"Error reading file: {e}")
                print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error during requirement generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

