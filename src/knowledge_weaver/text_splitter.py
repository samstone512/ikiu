# src/knowledge_weaver/text_splitter.py
# This module is upgraded to perform hybrid semantic chunking.

import re
from typing import List

# --- Configuration for the fallback recursive splitter ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

def _recursive_split(text: str, separators: List[str]) -> List[str]:
    """
    The core logic for the recursive splitting strategy (our fallback).
    """
    final_chunks = []
    if not text:
        return []

    # If we've reached the end of separators, split by character
    if not separators:
        for i in range(0, len(text), CHUNK_SIZE):
            final_chunks.append(text[i:i + CHUNK_SIZE])
        return final_chunks

    separator = separators[0]
    splits = text.split(separator)
    
    current_chunk = ""
    for s in splits:
        # If adding the next split exceeds chunk size, finalize the current chunk
        if len(current_chunk) + len(s) + len(separator) > CHUNK_SIZE:
            if current_chunk:
                final_chunks.append(current_chunk)
            current_chunk = s
        else:
            # Otherwise, append the split to the current chunk
            if current_chunk:
                current_chunk += separator + s
            else:
                current_chunk = s
    
    # Add the last remaining chunk
    if current_chunk:
        final_chunks.append(current_chunk)

    # Check if any of the generated chunks are still too large
    final_final_chunks = []
    for chunk in final_chunks:
        if len(chunk) > CHUNK_SIZE:
            # Recursively split the oversized chunk with the next separator
            deeper_chunks = _recursive_split(chunk, separators[1:])
            final_final_chunks.extend(deeper_chunks)
        else:
            final_final_chunks.append(chunk)
            
    return final_final_chunks

def split_text_hybrid(cleaned_text: str) -> List[str]:
    """
    Splits text using a hybrid strategy: first by legal articles, 
    and if that fails, falls back to a recursive character-based split.

    Args:
        cleaned_text (str): The pre-processed and cleaned text of a document.

    Returns:
        List[str]: A list of semantically meaningful text chunks.
    """
    print("Performing hybrid intelligent chunking...")
    
    # --- Primary Strategy: Split by Legal Articles ---
    legal_pattern = r'(?=\n\s*(?:ماده|تبصره|اصل|بند)\s+[\w\d()]+)'
    legal_chunks = [chunk.strip() for chunk in re.split(legal_pattern, cleaned_text) if chunk.strip()]
    
    # --- Heuristic Check ---
    # If we have more than one chunk, the legal split was likely successful.
    if len(legal_chunks) > 1:
        print(f"Successfully split text into {len(legal_chunks)} legal article chunks.")
        return legal_chunks
        
    # --- Fallback Strategy: Recursive Splitting ---
    print("Legal structure not prominent. Falling back to recursive splitting.")
    separators = ["\n\n", "\n", ". ", "، ", " "]
    recursive_chunks = _recursive_split(cleaned_text, separators)
    
    # Apply overlap to the recursive chunks
    final_chunks_with_overlap = []
    for i in range(len(recursive_chunks)):
        chunk = recursive_chunks[i]
        if i > 0 and CHUNK_OVERLAP > 0:
            previous_chunk = recursive_chunks[i-1]
            overlap = previous_chunk[-CHUNK_OVERLAP:]
            chunk = overlap + chunk
        final_chunks_with_overlap.append(chunk)

    print(f"Successfully split text into {len(final_chunks_with_overlap)} recursive chunks.")
    return final_chunks_with_overlap

# --- Main function to be called from other modules ---
def split_text(text: str) -> List[str]:
    """
    High-level function to use the new hybrid splitter.
    """
    return split_text_hybrid(text)

