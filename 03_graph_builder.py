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
GRAPH_DATA_PATH = OUTPUT_DIR / "graph_data.pkl" # <<< NEW: Path to save graph data

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

# --- Functions (Unchanged) ---
def clean_text_final(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'^[\s\-_ـ\.\d]+|[\s\-_ـ\.\d]+$', '', text).strip()
    text = re.sub(r'\s*\d+\s*ـ\s*\d+\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

def visualize_graph(nodes: Set[EntityNode], edges: Set[Tuple[EntityNode, EntityNode]], output_path: Path):
    # This function remains the same, no changes needed here.
    user_font_path = "D:/Software/font/F/Tahoma.ttf" 
    G = nx.Graph()
    labels = {}
    for node in nodes: G.add_node(node.text); labels[node.text] = node.text
    for node1, node2 in edges: G.add_edge(node1.text, node2.text)
    font_prop = fm.FontProperties(fname=user_font_path)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family=font_prop.get_name())
    plt.title("Knowledge Graph Visualization", size=20)
    plt.axis('off')
    plt.savefig(output_path, format="PNG", dpi=300)
    logging.info(f"Graph visualization saved successfully to: {output_path}")

# --- NEW: Function to save graph data ---
def save_graph_data(nodes: Set[EntityNode], edges: Set[Tuple[EntityNode, EntityNode]], file_path: Path):
    """Saves the graph nodes and edges to a pickle file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump({'nodes': nodes, 'edges': edges}, f)
        logging.info(f"Successfully saved graph data ({len(nodes)} nodes, {len(edges)} edges) to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save graph data: {e}", exc_info=True)

def main():
    logging.info("--- Starting Phase 08: Graph Builder (Final Run with Data Persistence) ---")
    
    knowledge_chunks = load_knowledge_chunks(KNOWLEDGE_CHUNK_PATH)
    if not knowledge_chunks: return

    ner_pipeline = initialize_ner_pipeline()
    graph_nodes, graph_edges = extract_relations(knowledge_chunks, ner_pipeline)

    logging.info(f"\nFound {len(graph_nodes)} nodes and {len(graph_edges)} edges.")

    if graph_nodes and graph_edges:
        visualize_graph(graph_nodes, graph_edges, GRAPH_IMAGE_PATH)
        save_graph_data(graph_nodes, graph_edges, GRAPH_DATA_PATH) # <<< NEW: Saving the data

    logging.info("\n--- Phase 08: GraphBuilder Complete ---")


if __name__ == "__main__":
    main()