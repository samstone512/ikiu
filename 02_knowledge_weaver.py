import json
from pathlib import Path
import logging
import re
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "docling_output"
OUTPUT_DIR = ROOT_DIR / "03_output"
INPUT_JSON_NAME = "Book.json"
INPUT_JSON_PATH = DATA_DIR / INPUT_JSON_NAME
OUTPUT_PICKLE_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"

# --- Data Structure for Knowledge Chunks ---
@dataclass
class KnowledgeChunk:
    """A structured representation of a single piece of knowledge from the document."""
    source_page: int
    element_type: str
    cleaned_text: str

# --- Core Logic ---
def clean_text(text: str) -> str:
    """Performs basic text cleaning operations."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u200c-\u200f]', '', text) # Remove zero-width non-joiner
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def create_knowledge_chunks(docling_output: Dict[str, Any]) -> List[KnowledgeChunk]:
    """
    Extracts knowledge chunks from the top-level 'texts' list in the docling JSON output.
    """
    texts_list = docling_output.get('texts')
    if not isinstance(texts_list, list):
        logging.error("Could not find a valid 'texts' list at the top level of the JSON.")
        return []

    knowledge_chunks = []
    logging.info(f"Processing {len(texts_list)} text items from the global list...")

    for item in texts_list:
        if not isinstance(item, dict):
            continue

        # Extract text
        text_content = item.get('text', '')
        cleaned_content = clean_text(text_content)

        # Extract page number from the 'prov' (provenance) field
        source_page = None
        prov_list = item.get('prov')
        if isinstance(prov_list, list) and prov_list:
            # Take provenance from the first item in the list
            page_info = prov_list[0]
            if isinstance(page_info, dict):
                source_page = page_info.get('page_no')

        # Extract element type
        element_type = item.get('label', 'unknown')

        # We only create a chunk if we have the essential information
        if cleaned_content and source_page is not None:
            chunk = KnowledgeChunk(
                source_page=source_page,
                element_type=element_type,
                cleaned_text=cleaned_content
            )
            knowledge_chunks.append(chunk)

    logging.info(f"Successfully created {len(knowledge_chunks)} knowledge chunks.")
    return knowledge_chunks

def save_chunks_to_pickle(chunks: List[KnowledgeChunk], file_path: Path):
    """Saves the list of knowledge chunks to a pickle file."""
    try:
        # Ensure the output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(chunks, f)
        logging.info(f"Successfully saved {len(chunks)} chunks to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save chunks to pickle file: {e}", exc_info=True)

def main():
    """Main function to orchestrate the full knowledge weaving process."""
    logging.info("--- Starting Phase 07: Knowledge Weaver (Final Production Run) ---")
    
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            docling_data = json.load(f)
        logging.info("Successfully loaded the docling JSON file.")
    except Exception as e:
        logging.error(f"Failed to load the primary JSON file: {e}", exc_info=True)
        return

    knowledge_chunks = create_knowledge_chunks(docling_data)

    if knowledge_chunks:
        save_chunks_to_pickle(knowledge_chunks, OUTPUT_PICKLE_PATH)
        # Log the first couple of chunks to verify
        logging.info("--- Verification: First 2 Chunks ---")
        for chunk in knowledge_chunks[:2]:
            logging.info(f"Page: {chunk.source_page}, Type: {chunk.element_type}, Text: '{chunk.cleaned_text[:100]}...'")
    
    logging.info("--- Knowledge Weaver phase complete. ---")


if __name__ == "__main__":
    main()