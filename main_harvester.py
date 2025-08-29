# main_harvester.py
# --- FOCUSED EXPERIMENT (FINAL CORRECTED VERSION) ---

import os
import sys
import json
import logging
from pathlib import Path

import google.generativeai as genai

import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
from src.data_harvester.ocr import extract_text_from_document
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

TARGET_BOOK_FILENAME = 'کتاب آیین نامه ی ارتقای مرتبه اعضای هیأت علمی آموزشی، پژوهشی و فناوری.pdf'

def main():
    """Main function for the focused data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting FOCUSED EXPERIMENT (V4.1)    ")
    logging.info(f"    Targeting single document: {TARGET_BOOK_FILENAME}    ")
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
    
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    target_pdf_path = config.RAW_PDFS_DIR / TARGET_BOOK_FILENAME
    if not target_pdf_path.exists():
        logging.error(f"FATAL: Target book '{TARGET_BOOK_FILENAME}' not found in {config.RAW_PDFS_DIR}")
        sys.exit(1)
    
    # This list will now contain only our single target file
    pdf_files_to_process = [target_pdf_path]

    # --- KEY CHANGE: Pass the specific list to the conversion function ---
    convert_pdfs_to_images(
        pdf_dir=config.RAW_PDFS_DIR,
        image_dir=config.IMAGES_DIR,
        specific_files=pdf_files_to_process
    )

    logging.info("--- Starting WHOLE DOCUMENT OCR Processing ---")
    
    folder = config.IMAGES_DIR / target_pdf_path.stem
    if not folder.is_dir():
        logging.error(f"FATAL: Image folder for '{target_pdf_path.stem}' not found.")
        sys.exit(1)

    pdf_name = folder.name
    json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"

    image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    full_structured_text = extract_text_from_document(image_paths, model)
    cleaned_full_text = clean_document_text(full_structured_text)

    output_data = {
        "pdf_filename": f"{pdf_name}.pdf",
        "total_pages": len(image_paths),
        "cleaned_full_text": cleaned_full_text
    }

    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved structured, cleaned text for the book to: {json_output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: FOCUSED EXPERIMENT Harvester Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
