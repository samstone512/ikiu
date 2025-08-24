# src/knowledge_weaver/preprocessor.py
# This new module is dedicated to cleaning the raw OCR text before chunking.

import re
from typing import List

def clean_document_text(raw_text: str) -> str:
    """
    Cleans the raw text extracted from a PDF by removing common headers, footers,
    and other irrelevant administrative text based on a set of regex patterns.

    Args:
        raw_text (str): The raw text content of a document.

    Returns:
        str: The cleaned text, ready for semantic chunking.
    """
    # --- Regex patterns to identify and remove noise from official documents ---
    # These patterns are inspired by your excellent suggestions.
    patterns_to_remove = [
        r'^\s*صفحه \d+ از \d+\s*$',  # Page numbers like "صفحه 2 از 7"
        r'^\s*(?:بسمه تعالی|به نام خدا)\s*$',  # Starting phrases
        r'^\s*(?:شماره|تاریخ|پیوست)\s*:.*',  # Header metadata like "شماره:", "تاریخ:"
        r'.*(?:کد\s*پستی|تلفن|فاکس|دورنگار|صندوق پستی|Website|Email)\s*[:=]?.*', # Contact info
        r'^\s*دانشگاه بین المللی امام خمینی \(ره\).*', # University headers
        r'IMAM KHOMEINI\s+INTERNATIONAL UNIVERSITY', # University headers (English)
        r'^\s*رونوشت به.*', # "Ronevesht be..." sections
        r'.*(?:معاون(?:ت)?|وزیر|رئیس|مدیر کل|امضاء|کامران دانشجو|محمد مهدی نژاد نوری)\s*[:=]?.*', # Signatures and titles
        r'^\s*([a-zA-Z0-9\s-]{10,})$' # Removes lines that are likely stray English headers/footers
    ]

    cleaned_lines = []
    for line in raw_text.splitlines():
        # Check if the line matches any of the removal patterns
        should_remove = any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns_to_remove)
        
        # Keep the line only if it's not noise and not empty
        if not should_remove and line.strip():
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines)
