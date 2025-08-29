# main_weaver.py
# --- V5 FINAL ROBUST VERSION: Processing document chunk-by-chunk for entity extraction ---

import os
import sys
import logging
from typing import List, Dict, Any

import google.generativeai as genai

# Import project configurations and modules
import config
from src.knowledge_weaver.json_loader import load_processed_texts
# --- KEY CHANGE: Import the text splitter ---
from src.knowledge_weaver.text_splitter import split_text
from src.knowledge_weaver.text_analyzer import load_prompt_template, analyze_text_for_entities
from src.knowledge_weaver.graph_builder import build_knowledge_graph, save_graph
from src.knowledge_weaver.vector_store import setup_chroma_collection

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to orchestrate the entire knowledge weaving pipeline."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Phase 02 - Knowledge Weaver (Robust Version)    ")
    logging.info("=========================================================")

    # --- 2. CONFIGURE GEMINI API ---
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        text_model = genai.GenerativeModel(config.GEMINI_TEXT_MODEL_NAME)
        logging.info("Successfully configured Google Gemini API.")
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        sys.exit(1)

    # --- 3. CREATE DATA DIRECTORIES ---
    config.KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Knowledge Weaver directories are ready.")

    # --- 4. LOAD PROCESSED TEXT DATA ---
    documents = load_processed_texts(config.PROCESSED_TEXT_DIR)
    if not documents:
        logging.info("No documents to process. Exiting.")
        sys.exit(0)

    # --- 5. CHUNK TEXTS AND EXTRACT STRUCTURED DATA (CHUNK-BY-CHUNK) ---
    logging.info("--- Starting Text Analysis for Entity Extraction (Chunk-by-Chunk) ---")
    prompt_template = load_prompt_template(config.ENTITY_EXTRACTION_PROMPT_PATH)
    all_structured_data: List[Dict[str, Any]] = []
    all_chunks_for_vector_store: List[str] = []
    
    # We now process each document individually
    for doc in documents:
        full_text = doc.get('full_text', '')
        if not full_text.strip():
            continue

        # --- KEY CHANGE: First, split the document into manageable chunks ---
        chunks = split_text(full_text)
        all_chunks_for_vector_store.extend(chunks) # Collect chunks for vector store
        logging.info(f"Document '{doc.get('source_filename')}' split into {len(chunks)} chunks.")

        # --- KEY CHANGE: Then, analyze each chunk individually ---
        for i, chunk in enumerate(chunks):
            logging.info(f"  - Analyzing chunk {i+1}/{len(chunks)}...")
            structured_data_chunk = analyze_text_for_entities(
                chunk, text_model, prompt_template
            )
            if structured_data_chunk:
                all_structured_data.append(structured_data_chunk)

    if not all_structured_data:
        logging.error("No structured data could be extracted from any chunk. Cannot build graph.")
        # We can still proceed to build the vector store
    else:
        # --- 6. BUILD AND SAVE KNOWLEDGE GRAPH ---
        knowledge_graph = build_knowledge_graph(all_structured_data)
        save_graph(knowledge_graph, config.KNOWLEDGE_GRAPH_DIR)

    # --- 7. SETUP VECTOR STORE AND EMBEDDINGS ---
    # We now pass the original documents list which contains metadata,
    # and the vector_store function will handle the chunking internally using the same logic.
    setup_chroma_collection(
        db_path=config.VECTOR_DB_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
        documents=documents, # Pass the original documents
        embedding_model_name=config.GEMINI_EMBEDDING_MODEL_NAME
    )

    logging.info("=========================================================")
    logging.info("    Project Danesh: Knowledge Weaver Phase Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
