# src/knowledge_weaver/vector_store.py
# Module for creating text embeddings and storing them in a ChromaDB vector store.

import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import google.generativeai as genai

def create_text_embeddings(
    text_chunks: List[str],
    embedding_model_name: str
) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks using a Google AI model.

    Args:
        text_chunks (List[str]): The list of text strings to embed.
        embedding_model_name (str): The name of the embedding model to use.

    Returns:
        List[List[float]]: A list of embedding vectors.
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
    Initializes a ChromaDB client, creates a collection, generates embeddings for
    the documents, and stores them in the collection.

    Args:
        db_path (Path): The directory to store the ChromaDB database files.
        collection_name (str): The name for the ChromaDB collection.
        documents (List[Dict[str, Any]]): The list of documents loaded by json_loader.
        embedding_model_name (str): The name of the embedding model.
    """
    logging.info("--- Setting up ChromaDB Vector Store ---")
    try:
        # Ensure the database directory exists.
        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize the ChromaDB client with persistence.
        client = chromadb.PersistentClient(path=str(db_path))

        # Create or get the collection.
        collection = client.get_or_create_collection(name=collection_name)
        logging.info(f"ChromaDB collection '{collection_name}' is ready.")

        # Prepare data for ChromaDB.
        texts_to_embed = [doc['full_text'] for doc in documents]
        metadatas = [{'source': doc['source_filename']} for doc in documents]
        ids = [f"doc_{i+1}" for i in range(len(documents))]

        # Generate embeddings for all texts.
        embeddings = create_text_embeddings(texts_to_embed, embedding_model_name)

        # Check if embeddings were generated successfully.
        if not any(embeddings):
             logging.error("No embeddings were generated. Cannot add to ChromaDB.")
             return

        # Add the data to the collection.
        collection.add(
            embeddings=embeddings,
            documents=texts_to_embed,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(documents)} documents to the ChromaDB collection.")

    except Exception as e:
        logging.error(f"An error occurred during ChromaDB setup: {e}")

