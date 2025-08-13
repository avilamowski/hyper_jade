#!/usr/bin/env python3
"""
Output Management Utility

This script provides utilities to list, view, and manage stored outputs
from the different agents in the pipeline.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.output_storage import OutputStorage

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename"""
    try:
        # Extract timestamp from filename like "rubric_assignment_2024-01-15T10:30:45.json"
        parts = Path(filename).stem.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[2:])  # Join remaining parts as timestamp
        return "Unknown"
    except:
        return "Unknown"

def main():
    """Main entry point for output management"""
    parser = argparse.ArgumentParser(description="Output Management Utility")
    parser.add_argument("--storage-dir", default="outputs", help="Output storage directory")
    parser.add_argument("--agent", choices=["requirement_generator", "prompt_generator", "code_corrector"], 
                       help="Filter by specific agent")
    parser.add_argument("--assignment-id", help="Filter by assignment ID")
    parser.add_argument("--view", help="View contents of specific output file")
    parser.add_argument("--latest", help="Show latest output for specific assignment ID")
    parser.add_argument("--clean", action="store_true", help="Clean old outputs (keep only latest per assignment)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize storage
    try:
        storage = OutputStorage(args.storage_dir)
    except Exception as e:
        print(f"Error initializing storage: {e}")
        sys.exit(1)
    
    # View specific file
    if args.view:
        try:
            filepath = Path(args.view)
            if not filepath.exists():
                print(f"Error: File {args.view} does not exist")
                sys.exit(1)
            
            print(f"📄 Viewing: {args.view}")
            print("=" * 50)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(json.dumps(data, indent=2))
            return
        except Exception as e:
            print(f"Error viewing file: {e}")
            sys.exit(1)
    
    # Show latest output for assignment
    if args.latest:
        try:
            print(f"🔍 Latest outputs for assignment ID: {args.latest}")
            print("=" * 50)
            
            for agent in ["requirement_generator", "prompt_generator", "code_corrector"]:
                latest = storage.get_latest_output(agent, args.latest)
                if latest:
                    timestamp = extract_timestamp_from_filename(Path(latest).name)
                    print(f"📋 {agent.replace('_', ' ').title()}: {Path(latest).name}")
                    print(f"    📅 {format_timestamp(timestamp)}")
                    print(f"    📁 {latest}")
                else:
                    print(f"📋 {agent.replace('_', ' ').title()}: No output found")
                print()
            return
        except Exception as e:
            print(f"Error finding latest outputs: {e}")
            sys.exit(1)
    
    # Clean old outputs
    if args.clean:
        try:
            print("🧹 Cleaning old outputs (keeping latest per assignment)...")
            cleaned_count = 0
            
            for agent in ["requirement_generator", "prompt_generator", "code_corrector"]:
                agent_dir = storage.output_dir / agent
                if not agent_dir.exists():
                    continue
                
                # Group files by assignment ID
                assignment_files = {}
                for file in agent_dir.glob("*.json"):
                    filename = file.name
                    # Extract assignment ID from filename
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        assignment_id = parts[1]  # Second part is assignment ID
                        if assignment_id not in assignment_files:
                            assignment_files[assignment_id] = []
                        assignment_files[assignment_id].append(file)
                
                # Keep only the latest file per assignment
                for assignment_id, files in assignment_files.items():
                    if len(files) > 1:
                        # Sort by modification time and keep the latest
                        files.sort(key=lambda f: f.stat().st_mtime)
                        for file in files[:-1]:  # Remove all but the latest
                            file.unlink()
                            cleaned_count += 1
                            if args.verbose:
                                print(f"  🗑️  Removed: {file.name}")
            
            print(f"✅ Cleaned {cleaned_count} old output files")
            return
        except Exception as e:
            print(f"Error cleaning outputs: {e}")
            sys.exit(1)
    
    # List outputs
    try:
        outputs = storage.list_outputs(args.agent)
        
        if not outputs:
            print("📭 No outputs found")
            return
        
        print("📁 STORED OUTPUTS")
        print("=" * 50)
        
        for agent_name, files in outputs.items():
            if not files:
                continue
                
            print(f"\n🤖 {agent_name.replace('_', ' ').title()}:")
            print("-" * 30)
            
            # Group files by assignment ID for better organization
            assignment_files = {}
            for filepath in files:
                filename = Path(filepath).name
                parts = filename.split('_')
                if len(parts) >= 2:
                    assignment_id = parts[1]
                    if assignment_id not in assignment_files:
                        assignment_files[assignment_id] = []
                    assignment_files[assignment_id].append((filepath, filename))
            
            for assignment_id, file_list in assignment_files.items():
                print(f"  📋 Assignment: {assignment_id}")
                
                # Sort by timestamp (newest first)
                file_list.sort(key=lambda x: x[1], reverse=True)
                
                for filepath, filename in file_list:
                    timestamp = extract_timestamp_from_filename(filename)
                    size = Path(filepath).stat().st_size
                    
                    print(f"    📄 {filename}")
                    print(f"       📅 {format_timestamp(timestamp)}")
                    print(f"       📏 {size:,} bytes")
                    print(f"       📁 {filepath}")
                
                print()
        
        if args.verbose:
            print("\n💡 Usage Examples:")
            print("  python list_outputs.py --latest assignment1")
            print("  python list_outputs.py --view outputs/requirement_generator/rubric_assignment1_2024-01-15T10:30:45.json")
            print("  python list_outputs.py --agent requirement_generator")
            print("  python list_outputs.py --clean")
        
    except Exception as e:
        print(f"Error listing outputs: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

