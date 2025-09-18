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

GRAPH_DATA_PATH = OUTPUT_DIR / "graph_data.pkl"
KNOWLEDGE_CHUNKS_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"
PROMPT_PATH = ROOT_DIR / "02_prompts" / "relation_extraction.txt"
# This is the final output of our phase
ENHANCED_GRAPH_DATA_PATH = OUTPUT_DIR / "graph_enhanced_data.pkl"
# Checkpoint file to save progress during the long run
CHECKPOINT_PATH = OUTPUT_DIR / "relation_enhancer_checkpoint.pkl"

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
        entities_dict = chunk.get("entities", {})
        if not isinstance(entities_dict, dict):
            continue
        all_entities_in_chunk = [entity for entity_list in entities_dict.values() for entity in entity_list]
        if source in all_entities_in_chunk and target in all_entities_in_chunk:
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
                time.sleep(2)
    return "RELATIONSHIP_EXTRACTION_FAILED"

def main():
    """Main function to enhance the entire graph with semantic relationships."""
    logging.info("--- Starting Phase 10: Relation Enhancer (FULL RUN) ---")
    
    if not setup_gemini(): return
    graph_data = load_data(GRAPH_DATA_PATH)
    knowledge_chunks = load_data(KNOWLEDGE_CHUNKS_PATH)
    try:
        prompt_template = PROMPT_PATH.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Failed to load prompt: {e}"); return

    if not all([graph_data, knowledge_chunks, prompt_template]): return

    edges = graph_data.get("edges", [])
    nodes = graph_data.get("nodes", [])
    logging.info(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges.")

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Load from checkpoint if it exists
    enhanced_relations = []
    start_index = 0
    if os.path.exists(CHECKPOINT_PATH):
        enhanced_relations = load_data(CHECKPOINT_PATH)
        start_index = len(enhanced_relations)
        logging.info(f"Resuming from checkpoint. {start_index} relations already processed.")

    # Process all edges from the start_index
    with tqdm(total=len(edges), initial=start_index, desc="Enhancing Relations") as pbar:
        for i in range(start_index, len(edges)):
            source, target = edges[i]
            context = find_context_for_edge(source, target, knowledge_chunks)
            if context:
                relationship = get_relationship(context, source, target, model, prompt_template)
                enhanced_relations.append((source, relationship, target))
            
            # Save progress periodically (e.g., every 10 edges)
            if i % 10 == 0:
                with open(CHECKPOINT_PATH, 'wb') as f:
                    pickle.dump(enhanced_relations, f)
            
            pbar.update(1)

    logging.info("--- Full processing complete ---")
    
    # Final save of all relations
    final_graph_data = {
        "nodes": nodes,
        "relations": enhanced_relations
    }
    with open(ENHANCED_GRAPH_DATA_PATH, 'wb') as f:
        pickle.dump(final_graph_data, f)
    
    logging.info(f"Successfully saved enhanced graph with {len(enhanced_relations)} relations to '{ENHANCED_GRAPH_DATA_PATH}'")
    
    # Clean up checkpoint file
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logging.info("Checkpoint file removed.")
        
    logging.info("--- Phase 10: RelationEnhancer COMPLETED ---")

if __name__ == "__main__":
    main()