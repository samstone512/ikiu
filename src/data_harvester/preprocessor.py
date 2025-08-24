# src/data_harvester/preprocessor.py
# This module is dedicated to cleaning the raw OCR text.

import re

def clean_document_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing common headers, footers, and other
    irrelevant administrative text based on a set of regex patterns.

    Args:
        raw_text (str): The raw text content of a document.

    Returns:
        str: The cleaned text, ready for semantic chunking.
    """
    print("Cleaning raw document text...")
    
    # --- Regex patterns to identify and remove noise from official documents ---
    patterns_to_remove = [
        r'^\s*صفحه \d+ از \d+\s*$',  # Page numbers like "صفحه 2 از 7"
        r'^\s*(?:بسمه تعالی|به نام خدا)\s*$',  # Starting phrases
        r'^\s*(?:شماره|تاریخ|پیوست)\s*:.*',  # Header metadata
        r'.*(?:کد\s*پستی|تلفن|فاکس|دورنگار|صندوق پستی|Website|Email|نشانی)\s*[:=]?.*', # Contact info
        r'^\s*دانشگاه بین المللی امام خمینی.*', # University headers
        r'IMAM KHOMEINI\s+INTERNATIONAL UNIVERSITY', # University headers (English)
        r'^\s*رونوشت به.*', # "Ronevesht be..." sections
        r'.*(?:معاون(?:ت)?|وزیر|رئیس|مدیر کل|امضاء|کامران دانشجو|محمد مهدی نژاد نوری|اسحاق جهانگیری)\s*[:=]?.*', # Signatures and titles
        r'^\s*\d+\s*$', # Lines containing only numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', # Email addresses
        r'https?://[^\s/$.?#].[^\s]*' # URLs
    ]

    cleaned_lines = []
    for line in raw_text.splitlines():
        should_remove = any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns_to_remove)
        if not should_remove and line.strip():
            cleaned_lines.append(line)
            
    cleaned_text = "\n".join(cleaned_lines)
    print("Text cleaning complete.")
    return cleaned_text
