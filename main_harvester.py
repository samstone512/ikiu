# main_harvester.py
# --- FOCUSED EXPERIMENT (V5 - FINAL ROBUST VERSION): Implementing intelligent batch processing ---

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
# --- Configuration for batch processing ---
BATCH_SIZE = 15  # Number of pages to process in each API call
BATCH_OVERLAP = 2 # Number of pages to overlap between batches to maintain context

def main():
    """Main function for the focused, robust, batch-processing data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting FOCUSED EXPERIMENT (V5 - Batch Processing)    ")
    logging.info(f"    Targeting: {TARGET_BOOK_FILENAME} with Batch Size: {BATCH_SIZE}, Overlap: {BATCH_OVERLAP}")
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
        logging.error(f"FATAL: Target book '{target_pdf_path.stem}' not found.")
        sys.exit(1)
    
    pdf_files_to_process = [target_pdf_path]
    convert_pdfs_to_images(
        pdf_dir=config.RAW_PDFS_DIR,
        image_dir=config.IMAGES_DIR,
        specific_files=pdf_files_to_process
    )

    logging.info("--- Starting BATCHED Document OCR Processing ---")
    
    folder = config.IMAGES_DIR / target_pdf_path.stem
    if not folder.is_dir():
        logging.error(f"FATAL: Image folder for '{target_pdf_path.stem}' not found.")
        sys.exit(1)

    image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    
    # --- BATCHING LOGIC ---
    all_extracted_text_parts = []
    for i in range(0, len(image_paths), BATCH_SIZE - BATCH_OVERLAP):
        batch_start = i
        batch_end = i + BATCH_SIZE
        image_batch = image_paths[batch_start:batch_end]
        
        if not image_batch:
            continue
            
        logging.info(f"--> Processing Batch: Pages {image_batch[0].stem.split('_')[-1]} to {image_batch[-1].stem.split('_')[-1]}")
        
        # The same multi-page OCR function is used, but on a smaller batch of images
        extracted_text_batch = extract_text_from_document(image_batch, model)
        all_extracted_text_parts.append(extracted_text_batch)

    # --- Combine and clean the results from all batches ---
    full_structured_text = "\n\n---\n\n".join(all_extracted_text_parts)
    cleaned_full_text = clean_document_text(full_structured_text)

    output_data = {
        "pdf_filename": f"{target_pdf_path.name}",
        "total_pages": len(image_paths),
        "cleaned_full_text": cleaned_full_text
    }

    json_output_path = config.PROCESSED_TEXT_DIR / f"{target_pdf_path.stem}.json"
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Successfully saved batched and cleaned text to: {json_output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: FOCUSED EXPERIMENT Harvester Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
