# main_harvester.py
# --- FINAL VERSION: Using a robust Tesseract-based pipeline ---

import os
import sys
import json
import logging
from pathlib import Path

import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
from src.data_harvester.ocr import extract_text_from_image_tesseract
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Data Harvester (Tesseract Final Version)    ")
    logging.info("======================================================")
    
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    all_pdf_files = list(config.RAW_PDFS_DIR.glob("*.pdf"))
    if not all_pdf_files:
        logging.info("No PDF files found to process. Exiting.")
        return
        
    logging.info(f"Found {len(all_pdf_files)} PDF(s) to process.")
    
    convert_pdfs_to_images(pdf_dir=config.RAW_PDFS_DIR, image_dir=config.IMAGES_DIR)

    for pdf_path in all_pdf_files:
        try:
            logging.info(f"\n--- Processing Document: {pdf_path.name} ---")
            folder = config.IMAGES_DIR / pdf_path.stem
            if not folder.is_dir():
                logging.warning(f"  - Image folder not found for {pdf_path.name}. Skipping.")
                continue

            json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_path.stem}.json"
            if json_output_path.exists():
                logging.info(f"  - JSON file already exists for {pdf_path.name}. Skipping.")
                continue

            image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
            
            all_pages_text = []
            for image_path in image_paths:
                # Call the new Tesseract-based OCR function for each page
                extracted_text = extract_text_from_image_tesseract(image_path)
                all_pages_text.append(extracted_text)

            full_raw_text = "\n\n".join(all_pages_text)
            cleaned_full_text = clean_document_text(full_raw_text)

            output_data = {
                "pdf_filename": pdf_path.name,
                "total_pages": len(image_paths),
                "cleaned_full_text": cleaned_full_text
            }

            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"  ✅ Successfully saved complete text for {pdf_path.name}")

        except Exception as e:
            logging.error(f"  ❌ FAILED to process document {pdf_path.name}. Error: {e}", exc_info=True)
            continue

    logging.info("\n======================================================")
    logging.info("    Project Danesh: Data Harvester Phase Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
