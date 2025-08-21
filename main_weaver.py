# main_weaver.py
# Main executable script for the Knowledge Weaver phase of Project Danesh.

import sys
import logging
from typing import List, Dict, Any

# Handle Google Colab environment for API key access
try:
    from google.colab import userdata
    import google.generativeai as genai
except ImportError:
    pass

# Import project configurations and modules
import config
from src.knowledge_weaver.json_loader import load_processed_texts
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
    logging.info("    Project Danesh: Starting Phase 02 - Knowledge Weaver    ")
    logging.info("=========================================================")

    # --- 2. CONFIGURE GEMINI API ---
    try:
        api_key = userdata.get('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' not found in Colab secrets.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        text_model = genai.GenerativeModel(config.GEMINI_TEXT_MODEL_NAME)
        logging.info("Successfully configured Google Gemini API for text and embedding models.")
    except NameError:
        logging.error("FATAL: Script not in Colab or 'userdata' unavailable.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        sys.exit(1)

    # --- 3. CREATE DATA DIRECTORIES ON GOOGLE DRIVE ---
    try:
        logging.info(f"Ensuring data directories exist in Google Drive...")
        config.KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        config.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Knowledge Weaver directories are ready.")
    except Exception as e:
        logging.error(f"FATAL: Could not create directories on Google Drive. Error: {e}")
        sys.exit(1)

    # --- 4. LOAD PROCESSED TEXT DATA ---
    documents = load_processed_texts(config.PROCESSED_TEXT_DIR)
    if not documents:
        logging.info("No documents to process. Exiting.")
        sys.exit(0)

    # --- 5. ANALYZE TEXT AND EXTRACT STRUCTURED DATA ---
    logging.info("--- Starting Text Analysis for Entity Extraction ---")
    prompt_template = load_prompt_template(config.ENTITY_EXTRACTION_PROMPT_PATH)
    all_structured_data: List[Dict[str, Any]] = []

    for doc in documents:
        # Note: For very large documents, you might need to split the text
        # into smaller chunks before sending to the analyzer.
        # For now, we process the full text of each document.
        structured_data = analyze_text_for_entities(
            doc['full_text'], text_model, prompt_template
        )
        if structured_data:
            all_structured_data.append(structured_data)

    if not all_structured_data:
        logging.error("No structured data could be extracted from any document. Cannot proceed.")
        sys.exit(1)

    # --- 6. BUILD AND SAVE KNOWLEDGE GRAPH ---
    knowledge_graph = build_knowledge_graph(all_structured_data)
    save_graph(knowledge_graph, config.KNOWLEDGE_GRAPH_DIR)

    # --- 7. SETUP VECTOR STORE AND EMBEDDINGS ---
    # We use a clear and consistent name for our collection.
    collection_name = "ikiu_regulations"
    setup_chroma_collection(
        db_path=config.VECTOR_DB_DIR,
        collection_name=collection_name,
        documents=documents,
        embedding_model_name=config.GEMINI_EMBEDDING_MODEL_NAME
    )

    logging.info("=========================================================")
    logging.info("    Project Danesh: Knowledge Weaver Phase Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
