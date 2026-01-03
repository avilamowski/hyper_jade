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

from src.agents.rag_prompt_generator.config import USE_RAG, WEAVIATE_URL, RAG_PYTHON_NOTEBOOKS_DIR
from src.agents.rag_prompt_generator.rag_system import RAGSystem

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Map package names to their import names
    required_packages = {
        "weaviate-client": "weaviate",
        "sentence-transformers": "sentence_transformers",
        "langchain": "langchain",
        "langchain-ollama": "langchain_ollama",
        "langchain-openai": "langchain_openai",
        "nbformat": "nbformat"
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name}")
            missing_packages.append(package_name)
    
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
        
        # Parse URL to get host and port (v4 API)
        url_parts = WEAVIATE_URL.replace('http://', '').replace('https://', '').split(':')
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080
        
        client = weaviate.connect_to_local(host=host, port=port)
        
        if client.is_ready():
            print("✅ Weaviate is running and ready")
            client.close()
            return True
        else:
            print("❌ Weaviate is not ready")
            client.close()
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

async def ingest_notebooks():
    """Ingest Python notebooks into the RAG system"""
    print(f"\n📚 Ingesting Python notebooks from {RAG_PYTHON_NOTEBOOKS_DIR}...")
    
    try:
        rag_system = RAGSystem()
        await rag_system.initialize()
        
        result = await rag_system.ingest_notebooks(dataset="python")
        print(f"✅ Ingested {result['count']} chunks from Python dataset")
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

async def display_stored_documents(limit=5):
    """Display stored documents from Weaviate for debugging"""
    print(f"\n🔍 Displaying stored documents (limit: {limit})...")
    
    try:
        import weaviate
        from weaviate.classes.query import MetadataQuery
        
        # Parse URL to get host and port
        url_parts = WEAVIATE_URL.replace('http://', '').replace('https://', '').split(':')
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080
        
        client = weaviate.connect_to_local(host=host, port=port)
        
        if not client.is_ready():
            print("❌ Weaviate is not ready")
            return False
        
        collection_name = "JadeNotebooks_Python"
        
        if not client.collections.exists(collection_name):
            print(f"❌ Collection {collection_name} does not exist")
            client.close()
            return False
        
        collection = client.collections.get(collection_name)
        
        # Get documents without vector search (just iterate)
        response = collection.query.fetch_objects(limit=limit)
        
        if not response.objects:
            print("❌ No documents found in collection")
            client.close()
            return False
        
        print(f"\n✅ Found {len(response.objects)} documents:\n")
        print("="*80)
        
        for i, obj in enumerate(response.objects, 1):
            print(f"\n📄 Document {i}:")
            print(f"   Filename: {obj.properties.get('filename', 'N/A')}")
            print(f"   Class Number: {obj.properties.get('class_number', 'N/A')}")
            print(f"   Content Length: {len(obj.properties.get('content', ''))} characters")
            print(f"\n   Content (first 500 chars with repr for whitespace):")
            content = obj.properties.get('content', '')
            print(f"   {repr(content[:500])}")
            print(f"\n   Content (actual display, first 500 chars):")
            print(f"   {content[:500]}")
            print("\n" + "-"*80)
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error displaying documents: {e}")
        import traceback
        traceback.print_exc()
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
    print("  --ingest-notebooks    Ingest Python notebooks before generation")
    print("")
    print("Configuration:")
    print(f"  RAG_PYTHON_NOTEBOOKS_DIR={RAG_PYTHON_NOTEBOOKS_DIR}")

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
    print("\n📚 Do you want to ingest Python notebooks? (y/n): ", end="")
    if input().lower().startswith('y'):
        if os.path.exists(RAG_PYTHON_NOTEBOOKS_DIR):
            await ingest_notebooks()
        else:
            print(f"⚠️  Notebooks directory not found: {RAG_PYTHON_NOTEBOOKS_DIR}")
            print(f"    Please set RAG_PYTHON_NOTEBOOKS_DIR environment variable or update config")
    
    # Step 4: Test RAG system
    print("\n🧪 Do you want to test the RAG system? (y/n): ", end="")
    if input().lower().startswith('y'):
        await test_rag_system()
    
    # Step 5: Display stored documents (for debugging)
    print("\n🔍 Do you want to display stored documents? (y/n): ", end="")
    if input().lower().startswith('y'):
        print("How many documents to display? (default: 5): ", end="")
        limit_input = input().strip()
        limit = int(limit_input) if limit_input.isdigit() else 5
        await display_stored_documents(limit)
    
    print("\n✅ RAG system setup complete!")
    print_usage_instructions()

def cli_main():
    """CLI entry point for setup-rag command"""
    asyncio.run(main())

if __name__ == "__main__":
    cli_main()
