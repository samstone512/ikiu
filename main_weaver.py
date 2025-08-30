# main_weaver.py
# --- PHASE 5 UPGRADE: This script now builds the QA vector store from enriched knowledge ---

import os
import sys
import logging
import json
from pathlib import Path

import google.generativeai as genai

import config
# We no longer need the old text analysis and graph building here for the vector store
from src.knowledge_weaver.vector_store import setup_qa_vector_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to build the new QA-based vector store."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Knowledge Weaver (Phase 5 - QA Vector Store)    ")
    logging.info("=========================================================")

    # Configure Gemini API for embeddings
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            sys.exit("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Google Gemini API for embedding models.")
    except Exception as e:
        sys.exit(f"FATAL: Failed to configure Gemini API. Error: {e}")

    # Ensure necessary directories exist
    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the enriched knowledge pairs from the file created by the enhancer
    knowledge_base_path = config.DRIVE_BASE_PATH / "knowledge_base" / "enriched_knowledge.json"
    if not knowledge_base_path.exists():
        sys.exit(f"FATAL: Enriched knowledge file not found at {knowledge_base_path}. Please run 'main_enhancer.py' first.")

    try:
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            knowledge_pairs = json.load(f)
        logging.info(f"Successfully loaded {len(knowledge_pairs)} QA pairs from the knowledge base.")
    except Exception as e:
        sys.exit(f"Failed to load or parse the enriched knowledge file. Error: {e}")

    # 2. Setup the vector store using the new QA-based function
    setup_qa_vector_store(
        db_path=config.VECTOR_DB_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        knowledge_pairs=knowledge_pairs,
        embedding_model_name=config.GEMINI_EMBEDDING_MODEL_NAME
    )

    # Note: Graph building is now a separate concern. The primary goal of this script
    # in Phase 5 is to build the high-accuracy QA vector store.

    logging.info("=========================================================")
    logging.info("    Project Danesh: Knowledge Weaver Phase Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
