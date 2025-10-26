#!/usr/bin/env python3
"""
Worker Runner Script

This script runs the worker from the correct directory to avoid import issues.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Change to worker directory
worker_dir = current_dir / "worker"
os.chdir(worker_dir)

# Import and run the worker
from worker import main

if __name__ == "__main__":
    main()


