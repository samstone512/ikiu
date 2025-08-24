# src/document_processing/processor.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from tqdm import tqdm
from typing import List, Dict

# --- Configuration for Tesseract ---
# If tesseract is not in your PATH, include the following line
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file using a hybrid approach."""
    print(f"Processing PDF: {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return ""
    full_text = []
    for page_num in tqdm(range(len(doc)), desc="Extracting text from pages"):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if len(text.strip()) < 100:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            try:
                ocr_text = pytesseract.image_to_string(image, lang='fas')
                final_page_text = ocr_text
            except Exception as ocr_error:
                print(f"\nAn error occurred during OCR on page {page_num + 1}: {ocr_error}")
                final_page_text = text
        else:
            final_page_text = text
        full_text.append(final_page_text)
    doc.close()
    print("Text extraction completed.")
    return "\n--- Page Break ---\n".join(full_text)


def clean_extracted_text(full_text: str) -> str:
    """Cleans raw text from administrative letters."""
    print("Performing advanced text cleaning for administrative documents...")
    patterns_to_remove = [
        r'^[\s\d]*(?:شماره|تاریخ|پیوست)\s*[:=].*',
        r'^\s*(?:بسمه تعالی|به نام خدا)\s*$',
        r'.*(?:کد\s*پستی|تلفن|فاکس|دورنگار|صندوق پستی)\s*[:=]\s*[\d\s-–()]+.*',
        r'.*(?:رونوشت|از طرف|مدیر کل|رئیس اداره|وزیر|معاون|رئیس جمهور|امضاء)\s*[:=]?.*',
        r'^\s*\d+\s*$',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-_.-]+\.[A-Z|a-z]{2,}\b',
        r'https?://[^\s/$.?#].[^\s]*'
    ]
    cleaned_text = full_text
    cleaned_text = re.sub(r'\n--- Page Break ---\n', '\n', cleaned_text)
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    lines = cleaned_text.split('\n')
    meaningful_lines = [line for line in lines if len(re.findall(r'[\u0600-\u06FFa-zA-Z0-9]', line)) > 15]
    cleaned_text = '\n'.join(meaningful_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    print("Advanced cleaning completed.")
    return cleaned_text


def intelligent_chunking(cleaned_text: str) -> List[Dict[str, str]]:
    """Splits the cleaned text into meaningful chunks based on legal keywords."""
    print("Performing intelligent chunking...")
    pattern = r'(?=\n\s*(?:ماده|تبصره|اصل)\s+[\d]+(?:-|\s*\.|\s*:|\s+))'
    raw_chunks = re.split(pattern, cleaned_text, flags=re.MULTILINE)
    structured_chunks = []
    chunk_header_pattern = r'^\s*(?P<type>ماده|تبصره|اصل)\s+(?P<id>[\d]+)[\s-–.:]*(?P<content>.*)'

    for chunk in raw_chunks:
        if not chunk.strip():
            continue

        header_match = re.match(chunk_header_pattern, chunk, re.DOTALL)
        
        if header_match:
            chunk_data = header_match.groupdict()
            structured_chunks.append({
                "type": chunk_data["type"].strip(),
                "identifier": chunk_data["id"].strip(),
                "content": chunk_data["content"].strip()
            })
        else:
            # --- THIS IS THE MISSING BLOCK THAT CAUSED THE ERROR ---
            structured_chunks.append({
                "type": "مقدمه",
                "identifier": "0",
                "content": chunk.strip()
            })
            
    print(f"Chunking complete. Found {len(structured_chunks)} structured chunks.")
    return structured_chunks
