# src/knowledge_weaver/vector_store.py
# This module is upgraded to use the new hybrid chunking pipeline on pre-cleaned text.

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
    """
    if not text_chunks:
        logging.warning("No text chunks received for embedding.")
        return []
    logging.info(f"--- Generating embeddings for {len(text_chunks)} text chunk(s) ---")
    try:
        result = genai.embed_content(
            model=embedding_model_name,
            content=text_chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        logging.info("Successfully generated embeddings.")
        return result['embedding']
    except Exception as e:
        logging.error(f"Failed to generate embeddings. Error: {e}")
        return [[] for _ in text_chunks]

def setup_chroma_collection(
    db_path: Path,
    collection_name: str,
    documents: List[Dict[str, Any]],
    embedding_model_name: str
):
    """
    Initializes ChromaDB and builds the collection using the hybrid chunking
    pipeline on the pre-cleaned text provided by the json_loader.
    """
    logging.info("--- Setting up ChromaDB with professional processing pipeline ---")
    try:
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))

        try:
            client.delete_collection(name=collection_name)
            logging.info(f"Existing collection '{collection_name}' deleted.")
        except Exception:
            logging.info(f"Collection '{collection_name}' did not exist. A new one will be created.")
            pass
        
        collection = client.create_collection(name=collection_name)
        logging.info(f"ChromaDB collection '{collection_name}' is ready.")

        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for i, doc in enumerate(documents):
            source_filename = doc.get('source_filename', 'unknown_file')
            # The text is already cleaned by the harvester phase
            cleaned_text = doc.get('full_text', '') 
            
            # --- Split the cleaned text using the hybrid splitter ---
            chunks = split_text(cleaned_text)
            
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({'source': source_filename})
                all_ids.append(f"doc_{i}_chunk_{j}")

        logging.info(f"Processed {len(documents)} documents into {len(all_chunks)} clean, semantic chunks.")

        if not all_chunks:
            logging.warning("No chunks were generated after processing. The vector store will be empty.")
            return

        embeddings = create_text_embeddings(all_chunks, embedding_model_name)

        if not any(embeddings):
             logging.error("No embeddings were generated. Cannot add to ChromaDB.")
             return

        collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        logging.info(f"Successfully added {len(all_chunks)} chunks to the ChromaDB collection.")

    except Exception as e:
        logging.error(f"An error occurred during ChromaDB setup: {e}", exc_info=True)
