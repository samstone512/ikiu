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
ENHANCED_GRAPH_DATA_PATH = OUTPUT_DIR / "graph_enhanced_data.pkl"
CHECKPOINT_PATH = OUTPUT_DIR / "relation_enhancer_checkpoint.pkl"

def setup_gemini():
    """Configures the Gemini API key."""
    load_dotenv()
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logging.error(f"Error configuring Gemini: {e}")
        return False

def load_data(file_path: Path) -> Any:
    """Loads data from a pickle file."""
    if not file_path.exists(): return None
    try:
        with open(file_path, "rb") as f: return pickle.load(f)
    except Exception: return None

def find_context_for_edge(source: str, target: str, chunks: List[Dict]) -> str:
    """
    Finds the text chunk that contains both entities.
    FINAL CORRECTED VERSION: Works with a simple list of entities.
    """
    for chunk in chunks:
        # 'entities' is now a simple list of strings, e.g., ['entity1', 'entity2']
        entities_in_chunk = chunk.get("entities", [])
        if not isinstance(entities_in_chunk, list):
            continue
        
        # Check if both source and target are present in this list
        if source in entities_in_chunk and target in entities_in_chunk:
            return chunk.get("text", "")
    return None

def get_relationship(context: str, source: str, target: str, model, prompt_template: str) -> str:
    """Uses the LLM to extract the relationship between two entities."""
    prompt = prompt_template.format(context_text=context, source_entity=source, target_entity=target)
    for attempt in range(3):
        try:
            response = model.generate_content(prompt, request_options={'timeout': 120})
            return response.text.strip()
        except Exception as e:
            logging.warning(f"API call failed on attempt {attempt+1}: {e}")
            time.sleep(2)
    return "RELATIONSHIP_EXTRACTION_FAILED"

def main():
    """Main function to enhance the entire graph with semantic relationships."""
    logging.info("--- Starting Phase 10: Relation Enhancer (FINAL RUN) ---")
    
    if not setup_gemini(): return
    graph_data = load_data(GRAPH_DATA_PATH)
    knowledge_chunks = load_data(KNOWLEDGE_CHUNKS_PATH)
    try:
        prompt_template = PROMPT_PATH.read_text(encoding='utf-8')
    except Exception: return

    if not all([graph_data, knowledge_chunks, prompt_template]): return

    edges = graph_data.get("edges", [])
    nodes = graph_data.get("nodes", [])
    logging.info(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges.")

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    enhanced_relations = []
    start_index = 0
    if os.path.exists(CHECKPOINT_PATH):
        enhanced_relations = load_data(CHECKPOINT_PATH)
        start_index = len(enhanced_relations) if enhanced_relations else 0
        logging.info(f"Resuming from checkpoint. {start_index} relations already processed.")

    # We need to track which edges have been processed
    processed_edges_count = start_index
    
    with tqdm(total=len(edges), initial=processed_edges_count, desc="Enhancing Relations") as pbar:
        for i in range(processed_edges_count, len(edges)):
            source, target = edges[i]
            context = find_context_for_edge(source, target, knowledge_chunks)
            
            relation_tuple = (source, "CONTEXT_NOT_FOUND", target)
            if context:
                relationship = get_relationship(context, source, target, model, prompt_template)
                relation_tuple = (source, relationship, target)

            enhanced_relations.append(relation_tuple)
            
            if i % 10 == 0:
                with open(CHECKPOINT_PATH, 'wb') as f:
                    pickle.dump(enhanced_relations, f)
            
            pbar.update(1)

    logging.info("--- Full processing complete ---")
    
    final_graph_data = {"nodes": nodes, "relations": enhanced_relations}
    with open(ENHANCED_GRAPH_DATA_PATH, 'wb') as f:
        pickle.dump(final_graph_data, f)
    
    logging.info(f"Successfully saved enhanced graph with {len(enhanced_relations)} relations to '{ENHANCED_GRAPH_DATA_PATH}'")
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logging.info("Checkpoint file removed.")
        
    logging.info("--- Phase 10: RelationEnhancer COMPLETED ---")

if __name__ == "__main__":
    main()