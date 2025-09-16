#phase10
import pickle
import os

# --- Constants ---
# Path to the input data from previous phases
GRAPH_DATA_PATH = "03_output/graph_data.pkl"
KNOWLEDGE_CHUNKS_PATH = "03_output/knowledge_chunks.pkl"

# Path for the final output of this phase
ENHANCED_GRAPH_DATA_PATH = "03_output/graph_enhanced_data.pkl"

def load_pickle_data(file_path):
    """Loads data from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def find_context_for_edge(source, target, knowledge_chunks):
    """Finds the text chunk that contains both the source and target entities."""
    for chunk in knowledge_chunks:
        # The entities are stored in the 'entities' key of each chunk dictionary
        entities_in_chunk = chunk.get("entities", [])
        if source in entities_in_chunk and target in entities_in_chunk:
            # Return the text of the first chunk that contains both
            return chunk.get("text", "Text not found in chunk.")
    return None # Return None if no chunk contains both entities

def main():
    """Main function to find the context for each co-occurrence edge."""
    print("Starting Phase 10: Relation Enhancer")
    print("Loading graph data and knowledge chunks...")

    # Load the graph data (nodes and edges)
    graph_data = load_pickle_data(GRAPH_DATA_PATH)
    if graph_data is None:
        return

    # Load the knowledge chunks
    knowledge_chunks = load_pickle_data(KNOWLEDGE_CHUNKS_PATH)
    if knowledge_chunks is None:
        return

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    print(f"Successfully loaded {len(nodes)} nodes and {len(edges)} edges.")
    print("-" * 50)
    
    # Let's process a small sample of edges first to test our logic
    edges_to_process = edges[:5] # We'll just test the first 5 edges
    
    print(f"Finding context for the first {len(edges_to_process)} edges...\n")

    for i, edge in enumerate(edges_to_process):
        source_node, target_node = edge
        
        # Find the text chunk that provides the context for this edge
        context_text = find_context_for_edge(source_node, target_node, knowledge_chunks)
        
        print(f"--- Edge {i+1}/{len(edges_to_process)} ---")
        print(f"  - Source: {source_node}")
        print(f"  - Target: {target_node}")
        
        if context_text:
            print(f"  - Found Context: '{context_text.strip()}'")
        else:
            print(f"  - Context not found!")
        print()

    print("-" * 50)
    print("Context finding test completed.")
    print("Next step will be to design a prompt and use an LLM to extract the relation.")


if __name__ == "__main__":
    main()