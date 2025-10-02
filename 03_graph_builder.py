import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import google.generativeai as genai
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import itertools
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "03_output"
INPUT_CHUNKS_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"
ENTITY_EXTRACTION_PROMPT_PATH = ROOT_DIR / "02_prompts" / "entity_extraction.txt"
GRAPH_DATA_PATH = OUTPUT_DIR / "graph_data.pkl"
CHECKPOINT_FILE_PATH = OUTPUT_DIR / "graph_builder_checkpoint.pkl"

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

def load_pickle_data(file_path: Path) -> List[Dict[str, Any]]:
    """Loads data from a pickle file."""
    try:
        with open(file_path, 'rb') as f: return pickle.load(f)
    except Exception: return []

def load_prompt_from_file(file_path: Path) -> str:
    """Loads a prompt from a text file."""
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception: return ""

def extract_entities_from_chunk(chunk_text: str, model, prompt_template: str) -> List[str]:
    """Uses Gemini to extract entities as a simple list."""
    prompt = prompt_template.format(text_chunk=chunk_text)
    for attempt in range(3):
        try:
            response = model.generate_content(prompt, request_options={'timeout': 300})
            text = response.text.strip().replace("```json", "").replace("```", "").strip()
            entities = json.loads(text)
            if isinstance(entities, list):
                return sorted(list(set(str(e) for e in entities)))
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []

def build_graph_from_chunks(chunks_with_entities: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Builds nodes and edges from chunks."""
    all_nodes: Set[str] = set()
    all_edges: Set[Tuple[str, str]] = set()
    for chunk in chunks_with_entities:
        entities = chunk.get("entities", [])
        if len(entities) > 1:
            all_nodes.update(entities)
            for combo in itertools.combinations(sorted(entities), 2):
                all_edges.add(combo)
    return sorted(list(all_nodes)), sorted(list(all_edges))

def save_data(data: Any, file_path: Path):
    """Saves data to a pickle file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Successfully saved data to {file_path}")

def main():
    """Main robust graph building process."""
    logging.info("--- Starting 03_graph_builder (Robust List Version) ---")
    if not setup_gemini(): return

    knowledge_chunks = load_pickle_data(INPUT_CHUNKS_PATH)
    prompt = load_prompt_from_file(ENTITY_EXTRACTION_PROMPT_PATH)
    if not knowledge_chunks or not prompt: return

    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    chunks_with_entities = []
    start_index = 0
    if CHECKPOINT_FILE_PATH.exists():
        chunks_with_entities = load_pickle_data(CHECKPOINT_FILE_PATH)
        start_index = len(chunks_with_entities)
        logging.info(f"Resuming from chunk {start_index + 1}")

    if start_index < len(knowledge_chunks):
        for i in tqdm(range(start_index, len(knowledge_chunks)), desc="Extracting Entities"):
            chunk = knowledge_chunks[i]
            entities = extract_entities_from_chunk(chunk.get('text', ''), model, prompt)
            chunk['entities'] = entities
            chunks_with_entities.append(chunk)
            save_data(chunks_with_entities, CHECKPOINT_FILE_PATH)

    logging.info(f"Entity extraction complete. Found entities in {len([c for c in chunks_with_entities if c.get('entities')])} chunks.")
    save_data(chunks_with_entities, INPUT_CHUNKS_PATH)

    nodes, edges = build_graph_from_chunks(chunks_with_entities)
    graph_data = {"nodes": nodes, "edges": edges}
    save_data(graph_data, GRAPH_DATA_PATH)

    if CHECKPOINT_FILE_PATH.exists():
        os.remove(CHECKPOINT_FILE_PATH)
        logging.info("Checkpoint file removed.")
    logging.info("--- Graph Builder phase (Rebuild) complete. ---")

if __name__ == "__main__":
    main()