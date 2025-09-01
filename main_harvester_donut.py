# main_harvester_donut.py
# --- PHASE 6: A dedicated runner for the Donut-based OCR pipeline ---

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path to allow importing modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
from src.data_harvester.ocr import extract_text_with_donut
from src.data_harvester.preprocessor import clean_document_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to run the Donut-based data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Harvester (V2 - Donut Model)    ")
    logging.info("======================================================")
    
    # Use the new, isolated directories for the Donut experiment
    images_dir = config.IMAGES_DIR_DONUT
    processed_text_dir = config.PROCESSED_TEXT_DIR_DONUT
    
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_text_dir.mkdir(parents=True, exist_ok=True)

    all_pdf_files = list(config.RAW_PDFS_DIR.glob("*.pdf"))
    if not all_pdf_files:
        logging.info("No PDF files found to process. Exiting.")
        return
        
    logging.info(f"Found {len(all_pdf_files)} PDF(s) to process.")
    
    # Convert PDFs to images, saving them in the new 'images_donut' directory
    convert_pdfs_to_images(pdf_dir=config.RAW_PDFS_DIR, image_dir=images_dir)

    for pdf_path in all_pdf_files:
        try:
            logging.info(f"\n--- Processing Document with Donut: {pdf_path.name} ---")
            folder = images_dir / pdf_path.stem
            if not folder.is_dir():
                logging.warning(f"  - Image folder not found for {pdf_path.name}. Skipping.")
                continue

            json_output_path = processed_text_dir / f"{pdf_path.stem}.json"
            if json_output_path.exists():
                logging.info(f"  - JSON file already exists for {pdf_path.name}. Skipping.")
                continue

            image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
            
            all_pages_text = []
            for image_path in image_paths:
                extracted_text = extract_text_with_donut(image_path)
                all_pages_text.append(extracted_text)

            full_raw_text = "\n\n".join(all_pages_text)
            # We still use the same preprocessor to clean up the final text
            cleaned_full_text = clean_document_text(full_raw_text)

            output_data = {
                "pdf_filename": pdf_path.name,
                "total_pages": len(image_paths),
                "cleaned_full_text": cleaned_full_text
            }

            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"  ✅ Successfully saved complete text for {pdf_path.name} to {json_output_path}")

        except Exception as e:
            logging.error(f"  ❌ FAILED to process document {pdf_path.name}. Error: {e}", exc_info=True)
            continue

    logging.info("\n======================================================")
    logging.info("    Project Danesh: Donut Harvester Phase Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
