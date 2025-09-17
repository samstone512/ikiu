# for checking our output data (graph data and knowledge chunks)

import pickle
from pathlib import Path
import sys

# --- Configuration ---
# Get the directory where the script is located (e.g., /04_test)
SCRIPT_DIR = Path(__file__).parent 
# Get the parent of the script's directory, which is the project's root
PROJECT_ROOT = SCRIPT_DIR.parent 

# Now, construct the path to the output directory relative to the project root
OUTPUT_DIR = PROJECT_ROOT / "03_output"
GRAPH_DATA_PATH = OUTPUT_DIR / "graph_data.pkl"
KNOWLEDGE_CHUNKS_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"

def inspect_pickle_file(file_path: Path):
    """Loads a pickle file and prints information about its contents."""
    print(f"--- Inspecting: {file_path.relative_to(PROJECT_ROOT)} ---")
    if not file_path.exists():
        print("🛑 ERROR: File does not exist!")
        return

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"✅ File loaded successfully.")
        
        if isinstance(data, list):
            print(f"   - Type: List")
            print(f"   - Number of items: {len(data)}")
            if data:
                print(f"   - First item's type: {type(data[0])}")
                print(f"   - First item's content (sample): {str(data[0])[:200]}...")
        
        elif isinstance(data, dict):
            print(f"   - Type: Dictionary")
            print(f"   - Keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, '__len__'):
                    print(f"   - Length of '{key}': {len(value)}")
        
        else:
            print(f"   - Type: {type(data)}")

    except Exception as e:
        print(f"🛑 ERROR: Could not read or inspect the file: {e}")
    
    print("-" * 40)
    print()


def main():
    """Main function to inspect the data files."""
    print("Running Data Inspector...")
    print("="*40)
    inspect_pickle_file(KNOWLEDGE_CHUNKS_PATH)
    inspect_pickle_file(GRAPH_DATA_PATH)
    print("Inspection complete.")
    print("="*40)

if __name__ == "__main__":
    main()