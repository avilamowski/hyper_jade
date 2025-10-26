#!/usr/bin/env python3
"""
Unified Prompt Generator Runner with RAG Support

This script automatically selects between standard prompt generation and RAG-enhanced
prompt generation based on the USE_RAG environment variable.

- If USE_RAG=true: Uses RAG-enhanced prompt generation with course theory
- If USE_RAG=false or not set: Uses standard prompt generation
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.rag_prompt_generator.config import USE_RAG

def main():
    """Main entry point that selects the appropriate prompt generator"""
    
    # Check if RAG is enabled
    if USE_RAG:
        print("🧠 RAG Enhanced Mode: Using RAG-enhanced prompt generation")
        print("=" * 60)
        print("🔧 RAG Mode: ENABLED")
        print("📚 Course theory integration: ACTIVE")
        print("=" * 60)
        
        # Import and run RAG prompt generator
        from run_rag_prompt_generator import main as rag_main
        asyncio.run(rag_main())
    else:
        print("📝 Standard Mode: Using standard prompt generation")
        print("=" * 60)
        print("🔧 RAG Mode: DISABLED")
        print("📚 Course theory integration: INACTIVE")
        print("=" * 60)
        
        # Import and run standard prompt generator
        from run_prompt_generator import main as standard_main
        standard_main()

if __name__ == "__main__":
    main()
