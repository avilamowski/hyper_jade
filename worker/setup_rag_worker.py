#!/usr/bin/env python3
"""
RAG Worker Setup Script

This script helps set up the RAG system for the worker by:
1. Checking if RAG is enabled
2. Starting Weaviate if needed
3. Ingesting notebooks for RAG functionality
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
current_file = Path(__file__)
parent_dir = current_file.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.agents.rag_prompt_generator.config import USE_RAG
from src.agents.rag_prompt_generator.rag_system import RAGSystem

def check_rag_status():
    """Check if RAG is enabled"""
    print("🔍 Checking RAG configuration...")
    
    if USE_RAG:
        print("✅ RAG Mode: ENABLED")
        print("📚 Course theory integration: ACTIVE")
        return True
    else:
        print("❌ RAG Mode: DISABLED")
        print("📚 Course theory integration: INACTIVE")
        print("💡 To enable RAG, set use_rag: true in src/config/rag_config.yaml")
        return False

async def setup_rag_system():
    """Setup RAG system for worker"""
    if not USE_RAG:
        print("⚠️  RAG is not enabled. Skipping RAG setup.")
        return
    
    print("\n🧠 Setting up RAG system for worker...")
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        await rag_system.initialize()
        print("✅ RAG system initialized successfully")
        
        # Check if notebooks need to be ingested
        print("\n📚 Do you want to ingest notebooks for RAG? (y/n): ", end="")
        if input().lower().startswith('y'):
            for dataset in ["python", "haskell"]:
                if os.path.exists(f"data/{dataset}"):
                    print(f"📖 Ingesting {dataset} notebooks...")
                    result = await rag_system.ingest_notebooks(dataset)
                    print(f"✅ Ingested {result['count']} chunks from {dataset} dataset")
                else:
                    print(f"⚠️  Dataset directory data/{dataset} not found, skipping...")
        
        print("\n🎉 RAG system ready for worker!")
        
    except Exception as e:
        print(f"❌ Error setting up RAG system: {e}")
        print("💡 Make sure Weaviate is running: docker-compose up -d")

async def main():
    """Main setup function"""
    print("🚀 Hyper JADE RAG Worker Setup")
    print("=" * 50)
    
    # Check RAG status
    if not check_rag_status():
        return
    
    # Setup RAG system
    await setup_rag_system()
    
    print("\n📋 Worker is ready!")
    print("🔧 RAG functionality will be available through the API endpoint:")
    print("   http://localhost:8000/assignment/1/requirement/generate_prompts")

if __name__ == "__main__":
    asyncio.run(main())


