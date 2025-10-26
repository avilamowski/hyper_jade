#!/usr/bin/env python3
"""
Worker Launcher Script

This script launches the worker using the module approach to avoid import issues.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import and run the worker
if __name__ == "__main__":
    from worker.main import main
    main()


