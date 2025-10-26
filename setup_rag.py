#!/usr/bin/env python3
"""
RAG System Setup Script

This script helps set up the RAG system for hyper_jade by:
1. Checking dependencies
2. Setting up Weaviate (if needed)
3. Ingesting notebooks
4. Testing the RAG system
"""

import sys
import os
import asyncio
import subprocess
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.rag_prompt_generator.config import USE_RAG, WEAVIATE_URL
from src.agents.rag_prompt_generator.rag_system import RAGSystem

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "weaviate-client",
        "sentence-transformers", 
        "langchain",
        "langchain-ollama",
        "langchain-openai",
        "nbformat"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def check_weaviate():
    """Check if Weaviate is running"""
    print(f"\n🔍 Checking Weaviate at {WEAVIATE_URL}...")
    
    try:
        import weaviate
        client = weaviate.Client(url=WEAVIATE_URL)
        if client.is_ready():
            print("✅ Weaviate is running and ready")
            return True
        else:
            print("❌ Weaviate is not ready")
            return False
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {e}")
        return False

def start_weaviate_docker():
    """Start Weaviate using Docker"""
    print("\n🐳 Starting Weaviate with Docker...")
    
    try:
        # Check if docker-compose.yml exists
        if os.path.exists("docker-compose.yml"):
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("✅ Weaviate started with docker-compose")
        else:
            # Start Weaviate directly with Docker
            subprocess.run([
                "docker", "run", "-d",
                "--name", "weaviate",
                "-p", "8080:8080",
                "-p", "50051:50051",
                "-e", "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true",
                "-e", "PERSISTENCE_DATA_PATH=/var/lib/weaviate",
                "semitechnologies/weaviate:latest"
            ], check=True)
            print("✅ Weaviate started with Docker")
        
        # Wait for Weaviate to be ready
        print("⏳ Waiting for Weaviate to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            if check_weaviate():
                return True
            time.sleep(1)
        
        print("❌ Weaviate failed to start within 30 seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Weaviate: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker or start Weaviate manually.")
        return False

async def ingest_notebooks(dataset="python"):
    """Ingest notebooks into the RAG system"""
    print(f"\n📚 Ingesting {dataset} notebooks...")
    
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        
        result = await rag_system.ingest_notebooks(dataset)
        print(f"✅ Ingested {result['count']} chunks from {dataset} dataset")
        return True
        
    except Exception as e:
        print(f"❌ Error ingesting notebooks: {e}")
        return False

async def test_rag_system():
    """Test the RAG system with a sample query"""
    print("\n🧪 Testing RAG system...")
    
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        
        # Test query
        test_query = "How to use variables in Python?"
        print(f"Query: {test_query}")
        
        results = await rag_system.retrieve_documents(
            query=test_query,
            max_results=3,
            dataset="python"
        )
        
        if results:
            print(f"✅ RAG system working - retrieved {len(results)} documents")
            for i, doc in enumerate(results[:2]):
                print(f"  Document {i+1}: {doc['content'][:100]}...")
            return True
        else:
            print("❌ No documents retrieved - RAG system may not be working properly")
            return False
            
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n📖 USAGE INSTRUCTIONS")
    print("=" * 50)
    print("To use RAG-enhanced prompt generation:")
    print("1. Set environment variable: export USE_RAG=true")
    print("2. Run: python run_prompt_generator_with_rag.py [options]")
    print("")
    print("To use standard prompt generation:")
    print("1. Set environment variable: export USE_RAG=false (or leave unset)")
    print("2. Run: python run_prompt_generator.py [options]")
    print("")
    print("RAG-specific options:")
    print("  --ingest-notebooks    Ingest notebooks before generation")
    print("  --dataset python      Use Python dataset (default)")
    print("  --dataset haskell     Use Haskell dataset")

async def main():
    """Main setup function"""
    print("🚀 Hyper JADE RAG System Setup")
    print("=" * 50)
    
    # Check if RAG is enabled
    if not USE_RAG:
        print("⚠️  RAG is not enabled. Set USE_RAG=true to use RAG functionality.")
        print_usage_instructions()
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Check Weaviate
    if not check_weaviate():
        print("\n🔧 Weaviate not running. Attempting to start...")
        if not start_weaviate_docker():
            print("\n❌ Could not start Weaviate. Please start it manually:")
            print("   docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \\")
            print("   -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \\")
            print("   semitechnologies/weaviate:latest")
            return
    
    # Step 3: Ingest notebooks
    print("\n📚 Do you want to ingest notebooks? (y/n): ", end="")
    if input().lower().startswith('y'):
        for dataset in ["python", "haskell"]:
            if os.path.exists(f"data/{dataset}"):
                await ingest_notebooks(dataset)
            else:
                print(f"⚠️  Dataset directory data/{dataset} not found, skipping...")
    
    # Step 4: Test RAG system
    print("\n🧪 Do you want to test the RAG system? (y/n): ", end="")
    if input().lower().startswith('y'):
        await test_rag_system()
    
    print("\n✅ RAG system setup complete!")
    print_usage_instructions()

if __name__ == "__main__":
    asyncio.run(main())
