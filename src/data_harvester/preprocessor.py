# src/data_harvester/preprocessor.py
# OPTIMIZATION-V2: This module is heavily upgraded to clean structured official documents.

import re

def clean_document_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing common headers, footers, and other
    irrelevant administrative text based on a comprehensive set of regex patterns
    tailored for official university and ministry documents.

    Args:
        raw_text (str): The raw text content of a document.

    Returns:
        str: The cleaned text, ready for semantic chunking.
    """
    print("Performing advanced text cleaning...")
    
    # --- Greatly expanded regex patterns based on document analysis ---
    patterns_to_remove = [
        r'^\s*جمهوری اسلامی ایران\s*$',
        r'^\s*وزارت علوم تحقیقات و فناوری\s*$',
        r'^\s*معاون پژوهشی و فناوری\s*$',
        r'^\s*بسمه تعالی\s*$',
        r'^\s*«بسمه تعالی»\s*$',
        r'^\s*(تاریخ|شماره|پیوست)\s*:?.*$',
        r'^\s*صفحه \d+ از \d+\s*$',
        r'^\s*\d+\s*$', # Lines containing only numbers (likely page numbers)
        r'.*(?:کد\s*پستی|تلفن|نمابر|دورنگار|صندوق پستی|Website|Email|نشانی|Ref|Date)\s*[:=]?.*',
        r'^\s*دانشگاه بین المللی امام خمینی.*',
        r'IMAM KHOMEINI\s+INTERNATIONAL UNIVERSITY',
        r'^\s*رونوشت به.*',
        r'.*(?:دکتر|مهندس|حجت الاسلام والمسلمین)\s+(?:محمد مهدی نژاد نوری|کامران دانشجو|علی خاکی صدیق|عبدالرضا باقری|محمد فریادی)', # Signatures
        r'^\s*(رئیس|سرپرست|معاون|وزیر|رئیس مرکز).*(هیئت های امنا|هیئت های ممیزه|دانشگاه|پژوهشی و فناوری)?\s*$', # Titles
        r'^\s*تصویب شد\s*$',
        r'^\s*مهر مرکز هیئت های امنا.*$',
        r'QAZVIN-IRAN',
        r'Tel\s*\:?\s*\+?98.*$',
        r'Fax\s*\:?\s*\+?98.*$',
        r'P\.?O\s*\.?Box:.*$',
        r'Postal Code:.*$',
        r'Email\s*:\s*President@IKIU\.AC\.IR'
    ]

    cleaned_lines = []
    # First, join the text to handle multi-line patterns and then split
    full_text = "\n".join(raw_text.splitlines())
    
    # Remove all identified patterns
    for pattern in patterns_to_remove:
        full_text = re.sub(pattern, '', full_text, flags=re.MULTILINE | re.IGNORECASE)
        
    # Final cleanup of residual empty lines
    for line in full_text.splitlines():
        if line.strip(): # Keep lines that are not empty
            cleaned_lines.append(line)
            
    cleaned_text = "\n".join(cleaned_lines)
    print("Advanced text cleaning complete.")
    return cleaned_text
