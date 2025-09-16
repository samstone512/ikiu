import pickle
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "03_output"
# Our new, reliable source of truth
INPUT_MD_PATH = OUTPUT_DIR / "full_regulations.md" 
# The path for the newly generated, reliable knowledge chunks
OUTPUT_PICKLE_PATH = OUTPUT_DIR / "knowledge_chunks.pkl"

# --- Data Structure for Knowledge Chunks (Simplified) ---
@dataclass
class KnowledgeChunk:
    """A structured representation of a single piece of knowledge from the document."""
    source_page: int
    text: str
    # We no longer need element_type as the markdown structure is implicit in the text
    # We also rename cleaned_text to just 'text' for clarity

def clean_text(text: str) -> str:
    """Performs basic text cleaning operations."""
    if not isinstance(text, str):
        return ""
    # Remove markdown headings, list markers, etc. to get pure text.
    text = re.sub(r'^\s*#+\s*', '', text) # Remove headings (e.g., ##)
    text = re.sub(r'^\s*[\*\-]\s*', '', text) # Remove list markers (* or -)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def create_knowledge_chunks_from_md(md_content: str) -> List[KnowledgeChunk]:
    """
    Extracts knowledge chunks from the clean markdown content.
    Each paragraph is considered a potential knowledge chunk.
    """
    knowledge_chunks = []
    
    # We split the entire document by the page separator we defined in the previous phase
    pages = md_content.split("\n\n---\n\n")
    logging.info(f"Document split into {len(pages)} pages.")

    for i, page_text in enumerate(pages):
        page_number = i + 1
        # Split each page's content into paragraphs based on empty lines
        paragraphs = page_text.split('\n\n')
        
        for paragraph in paragraphs:
            cleaned_paragraph = clean_text(paragraph)
            
            # We only create a chunk if it has meaningful content
            if len(cleaned_paragraph) > 20: # Filter out very short/empty lines
                chunk = KnowledgeChunk(
                    source_page=page_number,
                    text=cleaned_paragraph
                )
                knowledge_chunks.append(chunk)

    logging.info(f"Successfully created {len(knowledge_chunks)} knowledge chunks from Markdown file.")
    return knowledge_chunks

def save_chunks_to_pickle(chunks: List[KnowledgeChunk], file_path: Path):
    """Saves the list of knowledge chunks to a pickle file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # We are converting the list of dataclass objects to a list of dicts
        # for better compatibility with downstream processes.
        chunks_as_dicts = [asdict(chunk) for chunk in chunks]
        with open(file_path, 'wb') as f:
            pickle.dump(chunks_as_dicts, f)
        logging.info(f"Successfully saved {len(chunks_as_dicts)} chunks to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save chunks to pickle file: {e}")

def main():
    """Main function to orchestrate the new knowledge weaving process."""
    logging.info("--- Starting 02_knowledge_weaver: Rebuilding from clean Markdown ---")
    
    try:
        with open(INPUT_MD_PATH, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        logging.info("Successfully loaded the full_regulations.md file.")
    except FileNotFoundError:
        logging.error(f"FATAL: The input file was not found at '{INPUT_MD_PATH}'")
        return
    except Exception as e:
        logging.error(f"Failed to load the Markdown file: {e}")
        return

    knowledge_chunks = create_knowledge_chunks_from_md(markdown_content)

    if knowledge_chunks:
        save_chunks_to_pickle(knowledge_chunks, OUTPUT_PICKLE_PATH)
        logging.info("--- Verification: First 2 Chunks ---")
        for chunk in knowledge_chunks[:2]:
            logging.info(f"Page: {chunk.source_page}, Text: '{chunk.text[:150]}...'")
    
    logging.info("--- Knowledge Weaver phase (Rebuild) complete. ---")


if __name__ == "__main__":
    main()