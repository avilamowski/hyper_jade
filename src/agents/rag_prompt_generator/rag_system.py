"""
RAG System for Hyper JADE

Adapted from JADE_RAG but integrated with hyper_jade's configuration system.
Provides vector database functionality and document retrieval for RAG-enhanced prompt generation.
"""

import os
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import nbformat
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import logging
from pydantic import SecretStr

from .config import (
    WEAVIATE_URL,
    RAG_AI_PROVIDER,
    RAG_OLLAMA_HOST,
    RAG_OLLAMA_PORT,
    RAG_MODEL_NAME,
    RAG_OPENAI_API_KEY,
    RAG_OPENAI_MODEL,
    RAG_OPENAI_BASE_URL,
    RAG_TEXT_SPLITTER_STRATEGY,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_MIN_CHUNK_SIZE,
    RAG_DEBUG_MODE,
    RAG_ENABLE_RERANKING,
    RAG_RERANKING_MODEL,
    RAG_INITIAL_RETRIEVAL_COUNT,
    RAG_FINAL_RETRIEVAL_COUNT,
    RAG_LANGSMITH_API_KEY,
    RAG_LANGSMITH_PROJECT,
    RAG_LANGSMITH_ENDPOINT,
    RAG_LANGSMITH_TRACING,
    RAG_ENABLE_LANGSMITH_TRACING,
    RAG_TEMPERATURE_EXAMPLE_GENERATION,
    RAG_TEMPERATURE_THEORY_CORRECTION,
    RAG_PYTHON_NOTEBOOKS_DIR,
    RAG_HASKELL_NOTEBOOKS_DIR,
)

# Configure logging
log_level = logging.DEBUG if RAG_DEBUG_MODE else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Import LangSmith for tracing if available
try:
    from langsmith import Client
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available - tracing will be disabled")


def extract_class_number(filename: str) -> int:
    """
    Extract the class number from notebook filename.
    
    Examples:
    - "00 - Variables.ipynb" -> 0
    - "01 - Operadores aritméticos.ipynb" -> 1
    - "03_Funciones_utiles_y_Errores.ipynb" -> 3
    - "04.2-ciclos-while.ipynb" -> 4
    - "09.1-for.ipynb" -> 9
    - "09.2-intro-listas.ipynb" -> 9
    
    Args:
        filename: The notebook filename
        
    Returns:
        The class number as an integer, or 999 if no number is found
    """
    # Remove .ipynb extension
    name_without_ext = filename.replace('.ipynb', '')
    
    # Try to match number at the beginning of the filename
    # This handles patterns like "00", "01", "03", "04.2", "09.1", etc.
    match = re.match(r'^(\d+)', name_without_ext)
    
    if match:
        return int(match.group(1))
    else:
        # If no number is found, assign a high number to put it at the end
        logger.warning(f"No class number found in filename: {filename}, assigning 999")
        return 999


def preprocess_cell_content(cell_text: str, preserve_leading_whitespace: bool = False) -> str:
    """
    Preprocess cell content to clean markdown links and HTML images.
    """
    # For code cells, return as-is to preserve all indentation
    if preserve_leading_whitespace:
        logger.debug(f"Preserving whitespace for code cell (first 100 chars): {repr(cell_text[:100])}")
        return cell_text
    
    # Only apply preprocessing to non-code cells (markdown)
    # Remove markdown links but keep the text
    cell_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cell_text)
    
    # Remove HTML images but keep alt text
    cell_text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', r'\1', cell_text)
    cell_text = re.sub(r'<img[^>]*>', '', cell_text)
    
    return cell_text.strip()


class TextSplitter:
    """Configurable text splitter for Jupyter notebook content using LangChain"""
    
    def __init__(self, strategy: str = "cell_based", chunk_size: int = 1000, 
                 chunk_overlap: int = 200, min_chunk_size: int = 100, debug: bool = False):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.debug = debug
        
        # Initialize LangChain text splitter for non-cell-based strategies
        if strategy != "cell_based":
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[
                    "\n## ",       
                    "\n### ",      
                    "```python",   # Python code blocks
                    "```haskell",  # Haskell code blocks
                    "```",         # Generic code block endings
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ],
                is_separator_regex=False,
            )
        
    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text based on the configured strategy"""
        if self.strategy == "cell_based":
            return self._cell_based_split(text, metadata)
        else:
            return self._langchain_split(text, metadata)
    
    def _cell_based_split(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split by cell boundaries (original behavior)"""
        return [{"content": text, "metadata": metadata}]
    
    def _langchain_split(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text using LangChain's RecursiveCharacterTextSplitter"""
        # Use LangChain to split the text
        text_chunks = self.langchain_splitter.split_text(text)
        
        # Debug output
        if self.debug:
            print(f"\n=== DEBUG: Text Splitting for {metadata.get('filename', 'unknown')} ===")
            print(f"Original text length: {len(text)}")
            print(f"Number of chunks generated: {len(text_chunks)}")
            print(f"Chunk size limit: {self.chunk_size}")
            print(f"Chunk overlap: {self.chunk_overlap}")
            print(f"Min chunk size: {self.min_chunk_size}")
            print("\n--- Chunks ---")
            for i, chunk in enumerate(text_chunks):
                print(f"Chunk {i+1} (length: {len(chunk)}):")
                print(f"'{chunk[:100]}{'...' if len(chunk) > 100 else ''}'")
                print("-" * 50)
        
        # Convert to our format with metadata and apply stricter filtering
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "splitter_strategy": self.strategy
            })
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
        
        if self.debug:
            print(f"Final chunks after filtering and merging (min size {self.min_chunk_size}): {len(chunks)}")
            print("=" * 60)
        
        return chunks


class RAGSystem:
    def __init__(self):
        self.weaviate_url = WEAVIATE_URL
        self.ai_provider = RAG_AI_PROVIDER

        # Ollama configuration
        self.ollama_host = RAG_OLLAMA_HOST
        self.ollama_port = RAG_OLLAMA_PORT
        self.model_name = RAG_MODEL_NAME

        # OpenAI configuration
        self.openai_api_key = RAG_OPENAI_API_KEY
        self.openai_model = RAG_OPENAI_MODEL
        self.openai_base_url = RAG_OPENAI_BASE_URL

        # Text splitter configuration
        self.text_splitter = TextSplitter(
            strategy=RAG_TEXT_SPLITTER_STRATEGY,
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            min_chunk_size=RAG_MIN_CHUNK_SIZE,
            debug=RAG_DEBUG_MODE
        )

        # Reranking configuration
        self.enable_reranking = RAG_ENABLE_RERANKING
        self.reranking_model_name = RAG_RERANKING_MODEL
        self.initial_retrieval_count = RAG_INITIAL_RETRIEVAL_COUNT
        self.final_retrieval_count = RAG_FINAL_RETRIEVAL_COUNT

        self.client = None
        self.embedding_model = None
        self.reranker = None
        self.llm = None
        
        # LangSmith configuration
        self.enable_langsmith_tracing = RAG_ENABLE_LANGSMITH_TRACING
        self.langsmith_client = None
        
        # Temperature configurations
        self.temperature_example_generation = RAG_TEMPERATURE_EXAMPLE_GENERATION
        self.temperature_theory_correction = RAG_TEMPERATURE_THEORY_CORRECTION

    async def initialize(self):
        """Initialize the RAG system components"""
        try:
            # Initialize Weaviate client (v4 API)
            # Parse URL to get host and port
            url_parts = self.weaviate_url.replace('http://', '').replace('https://', '').split(':')
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 8080
            
            self.client = weaviate.connect_to_local(
                host=host,
                port=port
            )

            # Check if Weaviate is running
            if not self.client.is_ready():
                raise Exception("Weaviate is not ready. Please start Weaviate first.")

            # Create or get collection
            self._create_collection()

            # Initialize embedding model
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Initialize reranker if enabled
            if self.enable_reranking:
                logger.info(f"Initializing CrossEncoder reranker: {self.reranking_model_name}")
                self.reranker = CrossEncoder(self.reranking_model_name)
                logger.info("CrossEncoder reranker initialized successfully")
            else:
                logger.info("Reranking disabled")

            # Initialize LLM based on provider
            if self.ai_provider == "openai":
                if not self.openai_api_key:
                    raise Exception(
                        "OpenAI API key is required when using OpenAI provider"
                    )

                self.llm = ChatOpenAI(
                    model=self.openai_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    temperature=0.7
                )
                logger.info(f"Initialized OpenAI LLM with model: {self.openai_model}")
            else:
                # Initialize Ollama LLM
                self.llm = OllamaLLM(
                    model=self.model_name,
                    temperature=0.7,
                    base_url=f"http://{self.ollama_host}:{self.ollama_port}"
                )
                logger.info(f"Initialized Ollama LLM with model: {self.model_name}")

            # LangSmith is configured via environment variables
            if os.environ.get("LANGSMITH_TRACING") == "true":
                logger.info(f"LangSmith tracing enabled for project: {os.environ.get('LANGSMITH_PROJECT', 'default')}")
            else:
                logger.info("LangSmith tracing disabled")

            logger.info(
                f"RAG system initialized successfully with {self.ai_provider} provider"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _create_llm_with_temperature(self, temperature: float):
        """Create an LLM instance with a specific temperature"""
        if self.ai_provider == "openai":
            if not self.openai_api_key:
                raise Exception("OpenAI API key is required when using OpenAI provider")
            
            return ChatOpenAI(
                model=self.openai_model,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                temperature=temperature
            )
        else:
            # Initialize Ollama LLM
            return OllamaLLM(
                model=self.model_name,
                temperature=temperature,
                base_url=f"http://{self.ollama_host}:{self.ollama_port}"
            )
    
    def get_llm_for_example_generation(self):
        """Get LLM instance configured for creative example generation (higher temperature)"""
        return self._create_llm_with_temperature(self.temperature_example_generation)
    
    def get_llm_for_theory_correction(self):
        """Get LLM instance configured for precise theory-based correction (lower temperature)"""
        return self._create_llm_with_temperature(self.temperature_theory_correction)
    
    def _create_collection(self, dataset: str = "python"):
        """Create or get the notebooks collection in Weaviate for a specific dataset"""
        collection_name = f"JadeNotebooks_{dataset.title()}"
        
        # Check if collection exists
        if self.client.collections.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return

        # Create collection with v4 API
        self.client.collections.create(
            name=collection_name,
            description=f"JADE {dataset} course notebook content",
            vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own embeddings
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The content of the notebook cell",
                ),
                Property(
                    name="filename",
                    data_type=DataType.TEXT,
                    description="The name of the notebook file",
                ),
                Property(
                    name="notebook_path",
                    data_type=DataType.TEXT,
                    description="The full path to the notebook file",
                ),
                Property(
                    name="class_number",
                    data_type=DataType.INT,
                    description="The class number extracted from the notebook filename",
                ),
                Property(
                    name="dataset",
                    data_type=DataType.TEXT,
                    description="The dataset type (python or haskell)",
                ),
            ],
        )
        logger.info(f"Created collection {collection_name}")

    def extract_notebook_content(self, notebook_path: str, dataset: str = "python") -> List[Dict[str, Any]]:
        """Extract content from a Jupyter notebook as a single unit and split it consistently"""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Combine all content into a single text with proper formatting
            combined_content = []
            filename = os.path.basename(notebook_path)

            for cell in notebook.cells:
                # Preserve raw cell source for code cells so we don't remove
                # leading indentation. For markdown cells we still normalize
                # by trimming leading/trailing whitespace.
                cell_text = cell.source
                # Skip empty cells (allow whitespace-only test)
                if not cell_text or not cell_text.strip():
                    continue

                if cell.cell_type == "code":
                    # For code cells preserve leading whitespace/tabs
                    preprocessed_text = preprocess_cell_content(cell_text, preserve_leading_whitespace=True)
                    formatted_cell = f"```python\n{preprocessed_text}\n```"
                    logger.debug(f"Code cell formatted (first 200 chars):\n{repr(formatted_cell[:200])}")
                    logger.debug(f"Code cell preview:\n{formatted_cell[:300]}")
                    combined_content.append(formatted_cell)
                else:
                    # For markdown cells, clean and trim
                    preprocessed_text = preprocess_cell_content(cell_text, preserve_leading_whitespace=False)
                    combined_content.append(preprocessed_text)

            if not combined_content:
                return []

            # Join all content with double newlines
            full_content = "\n\n".join(combined_content)

            # Create metadata for the combined content
            class_number = extract_class_number(filename)
            metadata = {
                "filename": filename,
                "notebook_path": notebook_path,
                "total_cells": len(notebook.cells),
                "class_number": class_number,
                "dataset": dataset
            }
            
            # Use text splitter to split the combined content
            chunks = self.text_splitter.split_text(full_content, metadata)

            logger.info(f"Extracted {len(chunks)} chunks from {filename} using {self.text_splitter.strategy} strategy")
            return chunks

        except Exception as e:
            logger.error(f"Error processing notebook {notebook_path}: {e}")
            return []

    @traceable(name="rerank_documents")
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]):
        """Rerank documents using CrossEncoder"""
        if not self.reranker or not documents:
            logger.warning("Reranking disabled or no documents to rerank")
            return {
                "before_rerank": [],
                "after_rerank": [],
                "reranked_docs": documents
            }

        try:
            # Save before reranking snapshot for trace
            before_rerank = [
                {
                    "content": doc["content"],
                    "confidence": doc.get("confidence"),
                    "certainty": doc.get("certainty"),
                    "distance": doc.get("distance"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in documents
            ]

            # Prepare query-document pairs for CrossEncoder
            pairs = [[query, doc["content"]] for doc in documents]
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            # Add rerank scores to documents and sort
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(rerank_scores[i])
            # Sort by rerank score (higher is better)
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

            # Save after reranking snapshot for trace
            after_rerank = [
                {
                    "content": doc["content"],
                    "rerank_score": doc.get("rerank_score"),
                    "confidence": doc.get("confidence"),
                    "certainty": doc.get("certainty"),
                    "distance": doc.get("distance"),
                    "metadata": doc.get("metadata", {})
                }
                for doc in reranked_docs
            ]

            return {
                "before_rerank": before_rerank,
                "after_rerank": after_rerank,
                "reranked_docs": reranked_docs
            }
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return {
                "before_rerank": before_rerank if 'before_rerank' in locals() else [],
                "after_rerank": [],
                "reranked_docs": documents
            }

    async def ingest_notebooks(self, dataset: str = "python") -> Dict[str, Any]:
        """Ingest all notebooks from the specified dataset directory"""
        try:
            # Determine the notebooks directory based on dataset
            if dataset == "python":
                notebooks_dir = RAG_PYTHON_NOTEBOOKS_DIR
            elif dataset == "haskell":
                notebooks_dir = RAG_HASKELL_NOTEBOOKS_DIR
            else:
                raise ValueError(f"Unsupported dataset: {dataset}. Must be 'python' or 'haskell'")

            if not os.path.exists(notebooks_dir):
                raise FileNotFoundError(
                    f"Notebooks directory not found: {notebooks_dir}"
                )

            all_chunks = []
            notebook_files = [
                f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")
            ]

            logger.info(f"Found {len(notebook_files)} notebook files in {dataset} dataset")

            for notebook_file in notebook_files:
                notebook_path = os.path.join(notebooks_dir, notebook_file)
                chunks = self.extract_notebook_content(notebook_path, dataset)
                all_chunks.extend(chunks)
                logger.info(f"Processed {notebook_file}: {len(chunks)} chunks")

            if not all_chunks:
                logger.warning(f"No content found in {dataset} notebooks")
                return {"count": 0, "message": f"No content found in {dataset} dataset"}

            # Clear existing collection for this dataset
            collection_name = f"JadeNotebooks_{dataset.title()}"
            try:
                self.client.collections.delete(collection_name)
                self._create_collection(dataset)
            except:
                pass

            # Generate embeddings
            logger.info("Generating embeddings...")
            documents = [chunk["content"] for chunk in all_chunks]
            embeddings = self.embedding_model.encode(documents).tolist()

            # Add to Weaviate using v4 batch API
            logger.info("Adding to Weaviate...")
            collection = self.client.collections.get(collection_name)
            
            with collection.batch.dynamic() as batch:
                for i, chunk in enumerate(all_chunks):
                    properties = {
                        "content": chunk["content"],
                        "filename": chunk["metadata"]["filename"],
                        "notebook_path": chunk["metadata"]["notebook_path"],
                        "class_number": chunk["metadata"]["class_number"],
                        "dataset": chunk["metadata"]["dataset"],
                    }

                    batch.add_object(
                        properties=properties,
                        vector=embeddings[i],
                    )

            logger.info(f"Successfully ingested {len(all_chunks)} chunks for {dataset} dataset")
            return {
                "count": len(all_chunks),
                "message": f"Successfully ingested {dataset} notebooks",
                "dataset": dataset
            }

        except Exception as e:
            logger.error(f"Error ingesting {dataset} notebooks: {e}")
            raise

    @traceable(name="retrieve_documents")
    async def retrieve_documents(self, query: str, max_results: int, dataset: str = "python", max_class_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database without LLM processing"""
        try:
            if not self.client:
                raise Exception("Weaviate client not initialized")

            collection_name = f"JadeNotebooks_{dataset.title()}"

            if self.enable_reranking:
                retrieval_count = max(max_results * 3, self.initial_retrieval_count)  # Retrieve more for reranking
                final_count = max_results
            else:
                retrieval_count = max_results
                final_count = max_results

            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Build the query with optional class number filtering
            from weaviate.classes.query import Filter
            
            # Add class number filter if specified
            if max_class_number is not None:
                filters = Filter.by_property("class_number").less_or_equal(max_class_number)
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=retrieval_count,
                    return_metadata=MetadataQuery(certainty=True, distance=True),
                    filters=filters
                )
            else:
                response = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=retrieval_count,
                    return_metadata=MetadataQuery(certainty=True, distance=True)
                )

            if not response.objects:
                return []

            # Prepare context from retrieved documents
            context_docs = []
            i = 1
            for obj in response.objects:
                # Calculate confidence score from certainty (0-1 scale)
                certainty = obj.metadata.certainty if obj.metadata.certainty is not None else 0
                distance = obj.metadata.distance if obj.metadata.distance is not None else 1

                # Convert certainty to percentage and round to 1 decimal place
                confidence = round(certainty * 100, 1) if certainty else 0

                logger.debug(f"First 20 characters of content {i}: {obj.properties['content'][:20]}")
                i += 1

                # Generate class name from class number (handle 0 correctly)
                class_number = obj.properties.get("class_number")
                class_name = f"Clase {class_number}" if class_number is not None else "Unknown"
                
                context_docs.append(
                    {
                        "content": obj.properties["content"],
                        "confidence": confidence,
                        "certainty": certainty,
                        "distance": distance,
                        "class_name": class_name,
                        "metadata": {
                            "filename": obj.properties["filename"],
                            "notebook_path": obj.properties["notebook_path"],
                            "class_number": obj.properties["class_number"],
                            "dataset": obj.properties.get("dataset", dataset),
                        },
                    }
                )

            # Apply reranking if enabled
            if self.enable_reranking and len(context_docs) > final_count:
                logger.info(f"Reranking {len(context_docs)} documents to get top {final_count}")
                rerank_result = self.rerank_documents(query, context_docs)
                context_docs = rerank_result["reranked_docs"][:final_count]  # Take top N after reranking
            elif len(context_docs) > final_count:
                context_docs = context_docs[:final_count]  # Take top N without reranking

            return context_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
