# src/knowledge_weaver/text_splitter.py
# This module contains a professional-grade Recursive Character Text Splitter.

import re
from typing import List, Any

# --- Configuration for the text splitter ---
CHUNK_SIZE = 1500      # The target size of each chunk in characters. Increased for better context.
CHUNK_OVERLAP = 250    # The number of characters to overlap between chunks.

class RecursiveCharacterTextSplitter:
    """
    A professional text splitter that recursively tries to split text based on a
    hierarchy of separators to keep semantically related text together.
    """
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separators are ordered from largest semantic unit to smallest.
        # This is crucial for keeping paragraphs and table rows intact.
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """The core recursive splitting logic."""
        final_chunks = []
        
        # Get the first separator
        separator = separators[0]
        
        # If the separator is empty, we split by character as a last resort.
        if separator == "":
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                final_chunks.append(chunk)
            return final_chunks

        # Split the text by the current separator
        splits = text.split(separator)
        
        # Process each split
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # If a split is too large, we recursively call the function
                # with the *next* separator in the list.
                if len(separators) > 1:
                    deeper_chunks = self._split_text(s, separators[1:])
                    good_splits.extend(deeper_chunks)
                else:
                    # If there are no more separators, add the large chunk as is.
                    good_splits.append(s)
        
        # Merge small splits back together to form chunks of the desired size
        current_chunk = ""
        for split in good_splits:
            # If adding the next split doesn't exceed the chunk size, add it
            if len(current_chunk) + len(split) + (len(separator) if current_chunk else 0) <= self.chunk_size:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                # Otherwise, the current chunk is complete
                final_chunks.append(current_chunk)
                # Start a new chunk with the current split
                current_chunk = split
        
        # Add the last remaining chunk
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Public method to split a long text into smaller, overlapping chunks.
        """
        if not text:
            return []

        # Step 1: Perform the initial recursive split
        initial_chunks = self._split_text(text, self.separators)
        
        # Step 2: Create the final overlapping chunks
        final_chunks_with_overlap = []
        for i in range(len(initial_chunks)):
            chunk = initial_chunks[i]
            # If this isn't the first chunk, add the overlap from the previous one
            if i > 0 and self.chunk_overlap > 0:
                previous_chunk = initial_chunks[i-1]
                overlap = previous_chunk[-self.chunk_overlap:]
                chunk = overlap + chunk
            final_chunks_with_overlap.append(chunk)

        return final_chunks_with_overlap

# --- Main function to be called from other modules ---
def split_text(text: str) -> List[str]:
    """
    High-level function to instantiate and use the recursive splitter.
    """
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_text_into_chunks(text)
