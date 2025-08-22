# src/knowledge_weaver/vector_store.py
# Module for creating text embeddings and storing them in a ChromaDB vector store.
# This version is upgraded to use a recursive text splitter for improved accuracy.

import logging
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import google.generativeai as genai

# --- Import the new professional text splitter ---
from .text_splitter import split_text

def create_text_embeddings(
    text_chunks: List[str],
    embedding_model_name: str
) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using a Google AI model.
    Handles potential API errors gracefully.
    """
    logging.info(f"--- Generating embeddings for {len(text_chunks)} text chunk(s) ---")
    try:
        # The embed_content function can handle a batch of texts at once.
        result = genai.embed_content(
            model=embedding_model_name,
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT" # Important for retrieval tasks
        )
        logging.info("Successfully generated embeddings.")
        return result['embedding']
    except Exception as e:
        logging.error(f"Failed to generate embeddings. Error: {e}")
        # Return a list of empty lists to prevent downstream errors.
        return [[] for _ in text_chunks]

def setup_chroma_collection(
    db_path: Path,
    collection_name: str,
    documents: List[Dict[str, Any]],
    embedding_model_name: str
):
    """
    Initializes ChromaDB, splits documents into chunks using the recursive splitter,
    generates embeddings for each chunk, and stores them in the collection.
    """
    logging.info("--- Setting up ChromaDB Vector Store with Text Chunking ---")
    try:
        # Initialize the ChromaDB client with persistence.
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))

        # Create or get the collection. This also clears the old collection if it exists.
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name)
        logging.info(f"ChromaDB collection '{collection_name}' is ready.")

        # --- New Logic: Process documents and create chunks ---
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            source_filename = doc.get('source_filename', 'unknown_file')
            full_text = doc.get('full_text', '')
            
            # Use the professional splitter to break the document into chunks
            chunks = split_text(full_text)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                # Each chunk's metadata points back to the original source file
                all_metadatas.append({'source': source_filename})
                # Create a unique ID for every single chunk
                all_ids.append(f"doc_{i}_chunk_{j}")

        logging.info(f"Split {len(documents)} documents into {len(all_chunks)} text chunks.")

        # Generate embeddings for all chunks at once for efficiency.
        embeddings = create_text_embeddings(all_chunks, embedding_model_name)

        if not any(embeddings):
             logging.error("No embeddings were generated. Cannot add to ChromaDB.")
             return

        # Add the chunked data to the collection.
        # Note: ChromaDB can handle adding in batches for very large datasets.
        # For our use case, adding all at once is fine.
        collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        logging.info(f"Successfully added {len(all_chunks)} chunks to the ChromaDB collection.")

    except Exception as e:
        logging.error(f"An error occurred during ChromaDB setup: {e}", exc_info=True)
