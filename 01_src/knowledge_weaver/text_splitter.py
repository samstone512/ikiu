# src/knowledge_weaver/text_splitter.py
# --- FINAL OPTIMIZATION: Hyper-Semantic Chunking ---

import re
from typing import List

def split_text_intelligent(cleaned_text: str) -> List[str]:
    """
    Splits text using a hyper-semantic strategy. It splits by major structural
    markers first, and then further splits those chunks by sentences,
    creating highly focused and contextually rich chunks.
    """
    print("Performing hyper-semantic chunking...")
    
    # Split by major structural markers first (articles, clauses, etc.)
    structural_pattern = r'(?=\n\s*(?:ماده|تبصره|اصل|بند|فصل|مقدمه|تعاریف|اهداف)\s+[\w\d()]+|\n\s*(?:[الف-ی]-|[ا-ی]\)|\d+-)\s+)'
    initial_chunks = re.split(structural_pattern, cleaned_text)
    
    final_chunks = []
    for chunk in initial_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # --- KEY CHANGE: Further split each structural chunk by sentences ---
        # This regex splits by periods, question marks, and newlines, but tries to avoid splitting on abbreviations.
        sentences = re.split(r'(?<=[.!?\n])\s+', chunk)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # We add a threshold to avoid very small, meaningless sentence fragments.
            if len(sentence) > 20: 
                final_chunks.append(sentence)

    if final_chunks:
        print(f"Successfully split text into {len(final_chunks)} hyper-semantic chunks.")
        return final_chunks
    else:
        # Fallback to paragraph splitting if no other structure is found
        print("No prominent structure found. Falling back to paragraph splitting.")
        fallback_chunks = [p.strip() for p in cleaned_text.split('\n\n') if p.strip() and len(p) > 50]
        print(f"Split text into {len(fallback_chunks)} paragraph chunks.")
        return fallback_chunks

def split_text(text: str) -> List[str]:
    """High-level function to use the new hyper-semantic splitter."""
    return split_text_intelligent(text)
