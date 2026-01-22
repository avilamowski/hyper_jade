#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
import re

def is_timestamp_folder(folder_name: str) -> bool:
    """Check if a folder name matches timestamp format: YYYYMMDDTHHMMSS"""
    timestamp_pattern = r'^\d{8}T\d{6}$'
    return bool(re.match(timestamp_pattern, folder_name))

def clean_experiments(base_path: str, dry_run: bool = True):
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"❌ Base path '{base_path}' does not exist.")
        return

    print(f"Scanning {base_dir} for empty experiment folders...")
    
    # Structure: base_dir / config / timestamp
    # We iterate over configs first
    for config_dir in base_dir.iterdir():
        if not config_dir.is_dir():
            continue
            
        for run_dir in config_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            if is_timestamp_folder(run_dir.name):
                # Check if empty (no files inside) or only empty subdirs
                # We can check if it has any file recursively? 
                # Or just check simple emptiness.
                # User said "carpetas vacias".
                
                # Check for explicit emptiness
                has_content = any(run_dir.iterdir())
                
                if not has_content:
                    print(f"🗑️  Found EMPTY folder: {run_dir}")
                    if not dry_run:
                        run_dir.rmdir() # rmdir is safe, only removes if empty
                        print(f"   Deleted.")
                else:
                    # Optional: Check for failed runs (no aggregate_metrics.json)
                    # But user specifically asked for "empty folders".
                    # Let's add a check for 'almost empty' or failed runs if they are just logs?
                    # For now, strict emptiness.
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up empty experiment folders.")
    parser.add_argument("--path", default="outputs/evaluation/theory_rag_experiment", help="Base path to scan")
    parser.add_argument("--force", action="store_true", help="Actually delete files (disable dry-run)")
    
    args = parser.parse_args()
    
    dry_run = not args.force
    if dry_run:
        print("⚠️  DRY RUN MODE: No folders will be deleted. Use --force to delete.")
    
    clean_experiments(args.path, dry_run)
