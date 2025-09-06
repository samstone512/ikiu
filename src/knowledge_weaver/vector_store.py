# src/knowledge_weaver/vector_store.py
# --- PHASE 5 UPGRADE: Building the vector store from question-answer pairs ---

import logging
import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import google.generativeai as genai

import config # Assuming config.py is accessible

def create_embeddings_from_questions(
    knowledge_pairs: List[Dict[str, str]],
    embedding_model_name: str
) -> List[List[float]]:
    """
    Generates embeddings specifically from the 'question' field of the knowledge pairs.
    """
    questions = [pair['question'] for pair in knowledge_pairs]
    if not questions:
        logging.warning("No questions received for embedding.")
        return []
    
    logging.info(f"--- Generating embeddings for {len(questions)} questions ---")
    try:
        # We use RETRIEVAL_DOCUMENT because the questions we generated are rich and descriptive
        result = genai.embed_content(
            model=embedding_model_name,
            content=questions,
            task_type="RETRIEVAL_DOCUMENT" 
        )
        logging.info("Successfully generated embeddings from questions.")
        return result['embedding']
    except Exception as e:
        logging.error(f"Failed to generate embeddings. Error: {e}")
        return []

def setup_qa_vector_store(
    db_path: Path,
    collection_name: str,
    knowledge_pairs: List[Dict[str, Any]],
    embedding_model_name: str
):
    """
    Initializes ChromaDB and builds the collection using the enriched
    question-answer knowledge pairs. Embeddings are based on questions,
    while documents are the corresponding answers.
    """
    logging.info("--- Setting up QA-based ChromaDB Vector Store ---")
    try:
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))

        try:
            client.delete_collection(name=collection_name)
            logging.info(f"Existing collection '{collection_name}' deleted for a fresh start.")
        except Exception:
            logging.info(f"Collection '{collection_name}' did not exist. A new one will be created.")
            pass
        
        collection = client.create_collection(name=collection_name)
        logging.info(f"ChromaDB collection '{collection_name}' is ready.")

        if not knowledge_pairs:
            logging.warning("No knowledge pairs provided. The vector store will be empty.")
            return

        # Generate embeddings from the 'question' part
        embeddings = create_embeddings_from_questions(knowledge_pairs, embedding_model_name)
        
        # The 'answer' part becomes the document content
        documents = [pair['answer'] for pair in knowledge_pairs]
        
        # Create metadata and IDs
        metadatas = [{'source': 'enriched_knowledge_base'} for _ in knowledge_pairs]
        ids = [f"qa_pair_{i}" for i in range(len(knowledge_pairs))]

        if not embeddings or not any(embeddings):
             logging.error("No valid embeddings were generated. Cannot add to ChromaDB.")
             return

        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(documents)} QA pairs to the ChromaDB collection.")

    except Exception as e:
        logging.error(f"An error occurred during ChromaDB setup: {e}", exc_info=True)
