# main_harvester.py
# Main executable script for the Data Harvester phase of Project Danesh.
# --- FOCUSED EXPERIMENT: Processing ONLY the main book to test new OCR prompt ---

import os
import sys
import json
import logging
from pathlib import Path

import google.generativeai as genai

# Import project-specific modules and configurations
import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
from src.data_harvester.ocr import extract_text_from_image
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- IMPORTANT: Define the filename of the single book you want to process ---
# --- PLEASE REPLACE 'نام_فایل_کتاب.pdf' with the actual filename ---
TARGET_BOOK_FILENAME = 'کتاب آیین نامه ی ارتقای مرتبه اعضای هیأت علمی آموزشی، پژوهشی و فناوری.pdf' 

def main():
    """Main function to orchestrate the focused data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting FOCUSED EXPERIMENT    ")
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

    config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # --- MODIFIED: Find only the target PDF file ---
    target_pdf_path = config.RAW_PDFS_DIR / TARGET_BOOK_FILENAME
    if not target_pdf_path.exists():
        logging.error(f"FATAL: Target book '{TARGET_BOOK_FILENAME}' not found in {config.RAW_PDFS_DIR}")
        sys.exit(1)
    
    pdf_files = [target_pdf_path]
    logging.info(f"Found the target PDF file to process.")

    convert_pdfs_to_images(config.RAW_PDFS_DIR, config.IMAGES_DIR)

    logging.info("--- Starting OCR Processing and Text Cleaning for the target book ---")
    
    # Process only the folder corresponding to our target book
    folder = config.IMAGES_DIR / target_pdf_path.stem
    if not folder.is_dir():
        logging.error(f"FATAL: Image folder for '{target_pdf_path.stem}' not found. PDF conversion might have failed.")
        sys.exit(1)

    pdf_name = folder.name
    json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"

    logging.info(f"Processing images for PDF: '{pdf_name}'")
    image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    
    raw_pages_data = []
    for image_path in image_paths:
        # Using the new, powerful OCR prompt from config.py
        extracted_text = extract_text_from_image(image_path, model)
        page_number = int(image_path.stem.split('_')[-1])
        raw_pages_data.append({
            "page_number": page_number,
            "image_filename": image_path.name,
            "extracted_text": extracted_text
        })

    full_raw_text = "\n\n".join(page['extracted_text'] for page in raw_pages_data)
    cleaned_full_text = clean_document_text(full_raw_text)

    output_data = {
        "pdf_filename": f"{pdf_name}.pdf",
        "total_pages": len(image_paths),
        "cleaned_full_text": cleaned_full_text,
        "raw_pages": raw_pages_data
    }

    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Successfully saved structured, cleaned text to: {json_output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file for '{pdf_name}'. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: FOCUSED EXPERIMENT Harvester Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
