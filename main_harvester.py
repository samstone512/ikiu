# main_harvester.py
# --- FOCUSED EXPERIMENT (FINAL VERSION): Using the new multi-page OCR function ---

import os
import sys
import json
import logging
from pathlib import Path

import google.generativeai as genai

import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
# --- KEY CHANGE: Import the new whole-document OCR function ---
from src.data_harvester.ocr import extract_text_from_document
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- Define the filename of the single book to process ---
TARGET_BOOK_FILENAME = 'کتاب آیین نامه ی ارتقای مرتبه اعضای هیأت علمی آموزشی، پژوهشی و فناوری.pdf'

def main():
    """Main function for the focused data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting FOCUSED EXPERIMENT (V4)    ")
    logging.info(f"    Targeting single document with multi-page OCR: {TARGET_BOOK_FILENAME}    ")
    logging.info("======================================================")

    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.GEMINI_VISION_MODEL_NAME)
        logging.info("Successfully configured Google Gemini API.")
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        sys.exit(1)
    
    # --- Ensure directories are clean for the experiment ---
    # For a clean run, it's best to manually delete the 'images', 'processed_text', 
    # 'knowledge_graph', 'vector_db', and 'optimization_data' folders from Drive.
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    target_pdf_path = config.RAW_PDFS_DIR / TARGET_BOOK_FILENAME
    if not target_pdf_path.exists():
        logging.error(f"FATAL: Target book '{TARGET_BOOK_FILENAME}' not found in {config.RAW_PDFS_DIR}")
        sys.exit(1)
    
    # --- This function remains the same ---
    convert_pdfs_to_images(config.RAW_PDFS_DIR, config.IMAGES_DIR)

    logging.info("--- Starting WHOLE DOCUMENT OCR Processing ---")
    
    folder = config.IMAGES_DIR / target_pdf_path.stem
    if not folder.is_dir():
        logging.error(f"FATAL: Image folder for '{target_pdf_path.stem}' not found.")
        sys.exit(1)

    pdf_name = folder.name
    json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"

    # --- KEY CHANGE: Process all pages at once ---
    image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    
    # Call the new function that sends all images in one go
    full_structured_text = extract_text_from_document(image_paths, model)
    
    # The preprocessor now cleans the structured Markdown text
    cleaned_full_text = clean_document_text(full_structured_text)

    output_data = {
        "pdf_filename": f"{pdf_name}.pdf",
        "total_pages": len(image_paths),
        "cleaned_full_text": cleaned_full_text
        # We no longer need to store raw pages individually
    }

    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved structured, cleaned text for the entire book to: {json_output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: FOCUSED EXPERIMENT Harvester Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
