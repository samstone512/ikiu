# main_harvester_donut.py
# --- PHASE 6: Refactored to load prompts from files ---

import os
import sys
import json
import logging
from pathlib import Path
import google.generativeai as genai

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

# --- NEW: Function to load any prompt template from a file ---
def load_prompt_template(prompt_path: Path) -> str:
    """Reads the content of a prompt template file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"FATAL: Prompt file not found at {prompt_path}.")
        raise
    except Exception as e:
        logging.error(f"FATAL: Could not read prompt file at {prompt_path}. Error: {e}")
        raise

def refine_text_with_llm(text: str, model: genai.GenerativeModel, prompt_template: str) -> str:
    """Uses a generative model to clean and refine OCR'd text."""
    if not text.strip():
        return ""
    logging.info("    - Refining OCR text with language model...")
    try:
        prompt = prompt_template.format(ocr_text=text)
        response = model.generate_content(prompt)
        logging.info("    - Text refinement successful.")
        return response.text.strip()
    except Exception as e:
        logging.warning(f"    - Text refinement failed. Error: {e}. Returning original text.")
        return text

def main():
    """Main function to run the Donut-based data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Harvester (V2 - Donut Model)    ")
    logging.info("======================================================")

    # --- NEW: Load the refinement prompt template at the beginning ---
    refinement_prompt_template = load_prompt_template(config.REFINEMENT_PROMPT_PATH)

    images_dir = config.IMAGES_DIR_DONUT
    processed_text_dir = config.PROCESSED_TEXT_DIR_DONUT
    
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_text_dir.mkdir(parents=True, exist_ok=True)

    all_pdf_files = list(config.RAW_PDFS_DIR.glob("*.pdf"))
    if not all_pdf_files:
        logging.info("No PDF files found to process. Exiting.")
        return
        
    logging.info(f"Found {len(all_pdf_files)} PDF(s) to process.")
    
    convert_pdfs_to_images(pdf_dir=config.RAW_PDFS_DIR, image_dir=images_dir)

    # Initialize the Gemini model for the refinement step
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key: sys.exit("FATAL: 'GOOGLE_API_KEY' not set.")
        genai.configure(api_key=api_key)
        text_model = genai.GenerativeModel(config.GEMINI_TEXT_MODEL_NAME)
        logging.info("Successfully configured Gemini API for text refinement.")
    except Exception as e:
        sys.exit(f"FATAL: Failed to configure Gemini API. Error: {e}")

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
                # --- NEW: Refine text immediately after extraction ---
                refined_text = refine_text_with_llm(extracted_text, text_model, refinement_prompt_template)
                all_pages_text.append(refined_text)

            full_raw_text = "\n\n".join(all_pages_text)
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
