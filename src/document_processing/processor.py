
# src/document_processing/processor.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from tqdm import tqdm

# --- Configuration for Tesseract ---
# If tesseract is not in your PATH, include the following line
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux (if not in /usr/bin/): r'/usr/local/bin/tesseract'


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using a hybrid approach.
    Uses PyMuPDF to get text directly from text-based pages.
    Uses Tesseract OCR for pages that are image-based or have very little text.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The full extracted text from the document, with pages separated by newlines.
    """
    print(f"Processing PDF: {pdf_path}...")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return ""

    full_text = []
    
    # Using tqdm for a progress bar
    for page_num in tqdm(range(len(doc)), desc="Extracting text from pages"):
        page = doc.load_page(page_num)
        
        # 1. First, try to extract text directly
        text = page.get_text("text")
        
        # 2. Heuristic: If the extracted text is very short, it might be an image.
        #    We use a threshold of 100 characters (excluding whitespace).
        if len(text.strip()) < 100:
            # Render the page to an image
            pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR results
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            
            # Use Tesseract OCR to extract text from the image
            # We assume the documents are in Persian ('fas').
            try:
                ocr_text = pytesseract.image_to_string(image, lang='fas')
                # We trust the OCR result more in this case
                final_page_text = ocr_text
            except pytesseract.TesseractNotFoundError:
                print("\n\nWarning: Tesseract is not installed or not in your PATH.")
                print("Skipping OCR and using the minimal text found.")
                final_page_text = text # Fallback to the original short text
            except Exception as ocr_error:
                print(f"\n\nAn error occurred during OCR on page {page_num + 1}: {ocr_error}")
                final_page_text = text # Fallback to the original short text
        else:
            final_page_text = text
            
        full_text.append(final_page_text)
        
    doc.close()
    
    print("Text extraction completed.")
    return "\n--- Page Break ---\n".join(full_text)
