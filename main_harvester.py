# main_harvester.py
# Main executable script for the Data Harvester phase of Project Danesh.
# This version is upgraded to include the text cleaning preprocessing step.

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
# --- NEW: Import the text cleaning module ---
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to orchestrate the entire data harvesting and cleaning pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Phase 01 - Data Harvester    ")
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

    pdf_files = list(config.RAW_PDFS_DIR.glob('*.pdf'))
    if not pdf_files:
        logging.error(f"FATAL: No PDF files found in {config.RAW_PDFS_DIR}")
        sys.exit(1)
    logging.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    convert_pdfs_to_images(config.RAW_PDFS_DIR, config.IMAGES_DIR)

    logging.info("--- Starting OCR Processing and Text Cleaning ---")
    image_folders = [d for d in config.IMAGES_DIR.iterdir() if d.is_dir()]

    for folder in image_folders:
        pdf_name = folder.name
        json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"

        if json_output_path.exists():
            logging.info(f"JSON for '{pdf_name}' already exists. Skipping.")
            continue

        logging.info(f"Processing images for PDF: '{pdf_name}'")
        image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
        
        raw_pages_data = []
        for image_path in image_paths:
            extracted_text = extract_text_from_image(image_path, model)
            page_number = int(image_path.stem.split('_')[-1])
            raw_pages_data.append({
                "page_number": page_number,
                "image_filename": image_path.name,
                "extracted_text": extracted_text
            })

        # --- NEW: Consolidate and clean the text ---
        full_raw_text = "\n".join(page['extracted_text'] for page in raw_pages_data)
        cleaned_full_text = clean_document_text(full_raw_text)

        output_data = {
            "pdf_filename": f"{pdf_name}.pdf",
            "total_pages": len(image_paths),
            "cleaned_full_text": cleaned_full_text, # Store the clean version
            "raw_pages": raw_pages_data # Keep raw pages for reference
        }

        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved cleaned text to: {json_output_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON file for '{pdf_name}'. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: Data Harvester Phase Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
