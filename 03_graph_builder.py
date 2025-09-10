import pickle
from pathlib import Path
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "03_output"
KNOWLEDGE_CHUNK_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"
GRAPH_IMAGE_PATH = OUTPUT_DIR / "knowledge_graph.png"

# --- Data Structure Definitions ---
@dataclass
class KnowledgeChunk:
    source_page: int
    element_type: str
    cleaned_text: str

@dataclass(frozen=True, eq=True)
class EntityNode:
    text: str
    type: str

# --- Text Cleaning Function ---
def clean_text_final(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'^[\s\-_ـ\.\d]+|[\s\-_ـ\.\d]+$', '', text).strip()
    text = re.sub(r'\s*\d+\s*ـ\s*\d+\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Core Data Extraction Logic ---
def load_knowledge_chunks(file_path: Path) -> Optional[List[KnowledgeChunk]]:
    if not file_path.exists(): return None
    with open(file_path, 'rb') as f: return pickle.load(f)

def initialize_ner_pipeline() -> pipeline:
    model_name = "HooshvareLab/bert-fa-zwnj-base-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_relations(chunks: List[KnowledgeChunk], ner_pipeline: pipeline) -> Tuple[Set[EntityNode], Set[Tuple[EntityNode, EntityNode]]]:
    all_nodes: Set[EntityNode] = set()
    all_edges: Set[Tuple[EntityNode, EntityNode]] = set()
    for chunk in tqdm(chunks, desc="Processing Relations"):
        cleaned_text = clean_text_final(chunk.cleaned_text)
        if not cleaned_text: continue
        try:
            entities_in_chunk = [EntityNode(text=e['word'], type=e['entity_group']) for e in ner_pipeline(cleaned_text)]
            if len(entities_in_chunk) > 1:
                all_nodes.update(entities_in_chunk)
                for pair in itertools.combinations(entities_in_chunk, 2):
                    all_edges.add(tuple(sorted(pair, key=lambda x: x.text)))
        except Exception: continue
    return all_nodes, all_edges

# --- Graph Visualization Logic ---
def visualize_graph(nodes: Set[EntityNode], edges: Set[Tuple[EntityNode, EntityNode]], output_path: Path):
    """Creates and saves a visual representation of the knowledge graph."""
    logging.info("Starting graph visualization...")
    
    user_font_path = "D:/Software/font/F/Tahoma.ttf" 

    G = nx.Graph()
    labels = {}
    for node in nodes:
        G.add_node(node.text)
        labels[node.text] = node.text
    for node1, node2 in edges:
        G.add_edge(node1.text, node2.text)

    # --- Font Handling for Persian Labels ---
    font_prop = None
    try:
        font_prop = fm.FontProperties(fname=user_font_path)
        logging.info(f"Using specified font: {user_font_path}")
    except Exception as e:
        logging.error(f"Could not load the specified font. Error: {e}")
        logging.warning("Persian labels may not render correctly.")
    
    # --- Drawing the Graph ---
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    
    # === THE FIX IS HERE ===
    # Using 'font_family' instead of 'font_properties'
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family=font_prop.get_name() if font_prop else None)
    
    plt.title("Knowledge Graph Visualization", size=20)
    plt.axis('off')
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="PNG", dpi=300)
        logging.info(f"Graph visualization saved successfully to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save graph image: {e}")

def main():
    logging.info("--- Starting Phase 08: Graph Builder (Final Run with Visualization) ---")
    
    knowledge_chunks = load_knowledge_chunks(KNOWLEDGE_CHUNK_PATH)
    if not knowledge_chunks: return

    ner_pipeline = initialize_ner_pipeline()
    graph_nodes, graph_edges = extract_relations(knowledge_chunks, ner_pipeline)

    logging.info(f"\nFound {len(graph_nodes)} nodes and {len(graph_edges)} edges.")

    if graph_nodes and graph_edges:
        visualize_graph(graph_nodes, graph_edges, GRAPH_IMAGE_PATH)

    logging.info("\n--- Phase 08: GraphBuilder Complete ---")

if __name__ == "__main__":
    main()