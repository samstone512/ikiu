# src/knowledge_weaver/graph_builder.py
# Module for constructing and saving a knowledge graph using NetworkX.

import logging
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx

def build_knowledge_graph(structured_data: List[Dict[str, Any]]) -> nx.Graph:
    """
    Builds a NetworkX graph from a list of structured data containing
    entities and relationships.

    Args:
        structured_data (List[Dict[str, Any]]): A list of dictionaries, where
            each dictionary is the output of the text_analyzer module for a
            text chunk.

    Returns:
        nx.Graph: A NetworkX graph object representing the knowledge graph.
    """
    logging.info("--- Building Knowledge Graph ---")
    G = nx.Graph()

    for data_chunk in structured_data:
        if not data_chunk or 'entities' not in data_chunk or 'relationships' not in data_chunk:
            continue

        # Add entities as nodes
        for entity in data_chunk.get('entities', []):
            node_id = entity.get('id')
            node_type = entity.get('type')
            if node_id and not G.has_node(node_id):
                G.add_node(node_id, type=node_type)
                logging.debug(f"Added node: {node_id} (Type: {node_type})")

        # Add relationships as edges
        for rel in data_chunk.get('relationships', []):
            source_id = rel.get('source')
            target_id = rel.get('target')
            rel_type = rel.get('type')
            if source_id and target_id and G.has_node(source_id) and G.has_node(target_id):
                G.add_edge(source_id, target_id, type=rel_type)
                logging.debug(f"Added edge: {source_id} -> {target_id} (Type: {rel_type})")

    logging.info(f"Knowledge Graph built successfully.")
    logging.info(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    return G

def save_graph(graph: nx.Graph, output_dir: Path, filename: str = "knowledge_graph.graphml"):
    """
    Saves a NetworkX graph to a file in GraphML format.

    GraphML is a standard, portable format for graphs.

    Args:
        graph (nx.Graph): The NetworkX graph to save.
        output_dir (Path): The directory where the graph file will be saved.
        filename (str): The name of the output file.
    """
    if not graph.nodes:
        logging.warning("Graph is empty. Nothing to save.")
        return

    output_path = output_dir / filename
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, output_path)
        logging.info(f"Successfully saved knowledge graph to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save graph to {output_path}. Error: {e}")

