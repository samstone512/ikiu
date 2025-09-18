import pickle
from pathlib import Path
import argparse # Library for handling command-line arguments

# --- Configuration ---
# This script is in /04_test, so we go up one level to the project root
PROJECT_ROOT = Path(__file__).parent.parent 
KNOWLEDGE_CHUNKS_PATH = PROJECT_ROOT / "03_output" / "knowledge_chunks.pkl"

def view_chunks(file_path: Path, start: int, end: int):
    """Loads chunks from a pickle file and prints a specified range."""
    print(f"🔎 Loading chunks from: {file_path.relative_to(PROJECT_ROOT)}")
    
    if not file_path.exists():
        print(f"🛑 ERROR: File not found!")
        return

    try:
        with open(file_path, "rb") as f:
            chunks = pickle.load(f)
        
        total_chunks = len(chunks)
        print(f"✅ Successfully loaded {total_chunks} chunks.")
        
        # Adjusting the range to be user-friendly (1-based index)
        # and handling Python's 0-based slicing
        start_index = max(0, start - 1)
        end_index = min(total_chunks, end)

        if start_index >= end_index:
            print(f"⚠️ Warning: Invalid range. Start ({start}) must be less than End ({end}).")
            return
            
        print(f"\n--- Displaying chunks from {start_index + 1} to {end_index} ---")
        
        for i in range(start_index, end_index):
            chunk = chunks[i]
            page = chunk.get('source_page', 'N/A')
            text = chunk.get('text', 'No text found.')
            entities = chunk.get('entities', 'Not Extracted') # Check for entities
            
            print("\n" + "="*50)
            print(f"  CHUNK #{i + 1}  |  SOURCE PAGE: {page}")
            print("="*50)
            print(text)
            print("-" * 20)
            print(f"Extracted Entities: {entities}")
            print("="*50)

    except Exception as e:
        print(f"🛑 ERROR: Could not read or display the file: {e}")

def main():
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="View knowledge chunks from the project's pickle file.")
    parser.add_argument(
        'range', 
        metavar='RANGE', 
        type=str, 
        nargs='?', # The argument is optional
        default='1-10', # Default value if no argument is provided
        help='The range of chunks to display, e.g., "1-10", "50-55", or a single chunk number like "100".'
    )
    args = parser.parse_args()

    try:
        if '-' in args.range:
            start_str, end_str = args.range.split('-')
            start = int(start_str)
            end = int(end_str)
        else:
            start = int(args.range)
            end = start
        
        # To make a single number like '100' work, we view from 100 to 100
        # The view_chunks function will handle the slicing correctly
        if end < start:
            end = start

    except ValueError:
        print("🛑 Error: Invalid range format. Please use 'start-end' or a single number.")
        return

    view_chunks(KNOWLEDGE_CHUNKS_PATH, start, end)

if __name__ == "__main__":
    main()