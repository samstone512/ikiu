import pickle
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "03_output"

# Reliable data sources generated from Colab
GRAPH_DATA_PATH = OUTPUT_DIR / "graph_data.pkl"
KNOWLEDGE_CHUNKS_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"
PROMPT_PATH = ROOT_DIR / "02_prompts" / "relation_extraction.txt"

# Final output for this phase
ENHANCED_GRAPH_DATA_PATH = OUTPUT_DIR / "graph_enhanced_data.pkl"

def setup_gemini():
    """Configures the Gemini API key."""
    load_dotenv()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Error configuring Gemini: {e}")
        return False

def load_data(file_path: Path) -> Any:
    """Loads data from a pickle file."""
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def find_context_for_edge(source: str, target: str, chunks: List[Dict]) -> str:
    """Finds the text chunk that contains both the source and target entities."""
    for chunk in chunks:
        entities_in_chunk = chunk.get("entities", [])
        if source in entities_in_chunk and target in entities_in_chunk:
            return chunk.get("text", "")
    return None

def get_relationship(context: str, source: str, target: str, model, prompt_template: str) -> str:
    """Uses the LLM to extract the relationship between two entities."""
    prompt = prompt_template.format(context_text=context, source_entity=source, target_entity=target)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, request_options={'timeout': 120})
            return response.text.strip()
        except Exception as e:
            logging.warning(f"API call failed on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # Wait 2 seconds before retrying
    return "RELATIONSHIP_EXTRACTION_FAILED"

def main():
    """Main function to enhance the graph with semantic relationships."""
    logging.info("--- Starting Phase 10: Relation Enhancer ---")
    
    if not setup_gemini():
        return

    graph_data = load_data(GRAPH_DATA_PATH)
    knowledge_chunks = load_data(KNOWLEDGE_CHUNKS_PATH)
    try:
        prompt_template = PROMPT_PATH.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Failed to load prompt: {e}")
        return

    if not all([graph_data, knowledge_chunks, prompt_template]):
        logging.error("Halting due to missing data or prompt file.")
        return

    edges = graph_data.get("edges", [])
    nodes = graph_data.get("nodes", [])
    logging.info(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges.")

    # UPDATED: Using the more powerful Pro model for higher accuracy
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Let's process a sample of edges first to verify the logic
    edges_to_process = edges[:10]
    enhanced_relations = []

    for source, target in tqdm(edges_to_process, desc="Extracting Relations"):
        context = find_context_for_edge(source, target, knowledge_chunks)
        if context:
            relationship = get_relationship(context, source, target, model, prompt_template)
            tqdm.write(f"'{source}' -> '{relationship}' -> '{target}'")
            enhanced_relations.append((source, relationship, target))
        else:
            tqdm.write(f"Context not found for edge: ({source}, {target})")

    logging.info("--- Sample processing complete ---")
    logging.info("First 10 relationships extracted:")
    for rel in enhanced_relations:
        print(rel)

    # In the final version, we will process all edges and save the result.
    # For now, this confirms our logic is working.
    logging.info("Next step: Scale up to process all edges and save to 'graph_enhanced_data.pkl'.")

if __name__ == "__main__":
    main()