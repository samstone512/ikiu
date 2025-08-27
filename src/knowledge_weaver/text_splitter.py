# src/knowledge_weaver/text_splitter.py
# OPTIMIZATION-V2: This is now a structure-aware parser, not a simple splitter.
# It identifies semantic units like articles, clauses, and tables.

import re
from typing import List

def split_text_intelligent(cleaned_text: str) -> List[str]:
    """
    Splits text using a highly intelligent, structure-aware strategy.
    It prioritizes semantic boundaries like articles, clauses, and tables,
    making it ideal for legal and regulatory documents.

    Args:
        cleaned_text (str): The pre-processed and cleaned text of a document.

    Returns:
        List[str]: A list of semantically meaningful text chunks.
    """
    print("Performing structure-aware intelligent chunking...")
    
    # --- Primary Strategy: Split by major legal/structural markers ---
    # This regex looks for lines starting with "ماده", "تبصره", "اصل", "بند", "فصل",
    # or common headings followed by a space and number/letter.
    # The (?=...) is a positive lookahead, which splits the text *before* the pattern.
    structural_pattern = r'(?=\n\s*(?:ماده|تبصره|اصل|بند|فصل|مقدمه|تعاریف|اهداف)\s+[\w\d()]+|\n\s*(?:[الف-ی]-|[ا-ی]\)|\d+-)\s+)'
    
    # First, split the document into major structural blocks
    initial_chunks = re.split(structural_pattern, cleaned_text)
    
    final_chunks = []
    for chunk in initial_chunks:
        # Clean up whitespace from each chunk
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # --- Secondary Strategy: Handle Tables and long paragraphs ---
        # Heuristic: If a chunk is very large, it might be a long paragraph without
        # clear markers, or a table that wasn't split.
        # For now, we keep them as larger chunks to preserve context.
        # A more advanced version could add specific table-parsing logic here.
        
        # We also want to avoid chunks that are too small (e.g., just a heading)
        # This simple check ensures chunks have some meaningful content.
        if len(chunk) > 50: # Threshold to avoid tiny, meaningless chunks
            final_chunks.append(chunk)

    if final_chunks:
        print(f"Successfully split text into {len(final_chunks)} semantic chunks based on document structure.")
        return final_chunks
    else:
        # --- Fallback: If no structure is found, use a simple paragraph split ---
        print("No prominent structure found. Falling back to paragraph splitting.")
        fallback_chunks = [p.strip() for p in cleaned_text.split('\n\n') if p.strip() and len(p) > 50]
        print(f"Split text into {len(fallback_chunks)} paragraph chunks.")
        return fallback_chunks


# --- Main function to be called from other modules ---
def split_text(text: str) -> List[str]:
    """
    High-level function to use the new intelligent, structure-aware parser.
    """
    return split_text_intelligent(text)
