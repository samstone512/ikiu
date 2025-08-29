# main_harvester.py
# --- V6 (FINAL OPTIMIZATION): Adding an AI-powered refinement step after OCR ---

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
BATCH_SIZE = 15
BATCH_OVERLAP = 2

# --- NEW: Prompt for refining the OCR'd Markdown text ---
REFINEMENT_PROMPT = """
You are a text-processing expert. You will be given a block of Markdown text that was extracted via OCR. The text may contain errors like joined words, misspellings, or formatting issues.
Your task is to carefully review and correct the text to produce a clean, perfectly formatted Markdown output.

**Instructions:**
1.  **Correct OCR Errors:** Fix spelling mistakes and separate words that are incorrectly joined together (e.g., "اینیکمتن" should be "این یک متن").
2.  **Preserve Markdown Structure:** Do NOT alter the Markdown table structures (|---|---|). Ensure they remain valid.
3.  **Ensure Readability:** Make the text clean and easy to read.
4.  **Do NOT add new content or commentary.** Only correct and refine the existing text.

**Original OCR Text:**
---
{ocr_text}
---

**Refined Text:**
"""

def refine_text_with_llm(text: str, model: genai.GenerativeModel) -> str:
    """Uses a generative model to clean and refine OCR'd text."""
    if not text.strip():
        return ""
    
    logging.info("    - Refining OCR text with language model...")
    try:
        prompt = REFINEMENT_PROMPT.format(ocr_text=text)
        response = model.generate_content(prompt)
        logging.info("    - Text refinement successful.")
        return response.text.strip()
    except Exception as e:
        logging.warning(f"    - Text refinement failed. Error: {e}. Returning original text.")
        return text # Fallback to the original text if refinement fails

def main():
    """Main function for the final, refined data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting FOCUSED EXPERIMENT (V6 - AI Refinement)    ")
    logging.info(f"    Targeting: {TARGET_BOOK_FILENAME}")
    logging.info("======================================================")

    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            sys.exit("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
        genai.configure(api_key=api_key)
        vision_model = genai.GenerativeModel(config.GEMINI_VISION_MODEL_NAME)
        # --- NEW: We also need a text model for refinement ---
        text_model = genai.GenerativeModel(config.GEMINI_TEXT_MODEL_NAME)
        logging.info("Successfully configured Google Gemini API for vision and text models.")
    except Exception as e:
        sys.exit(f"FATAL: Failed to configure Gemini API. Error: {e}")
    
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    target_pdf_path = config.RAW_PDFS_DIR / TARGET_BOOK_FILENAME
    if not target_pdf_path.exists():
        sys.exit(f"FATAL: Target book '{target_pdf_path.stem}' not found.")
    
    pdf_files_to_process = [target_pdf_path]
    convert_pdfs_to_images(
        pdf_dir=config.RAW_PDFS_DIR,
        image_dir=config.IMAGES_DIR,
        specific_files=pdf_files_to_process
    )

    logging.info("--- Starting BATCHED Document OCR & Refinement Processing ---")
    
    folder = config.IMAGES_DIR / target_pdf_path.stem
    image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
    
    all_refined_text_parts = []
    for i in range(0, len(image_paths), BATCH_SIZE - BATCH_OVERLAP):
        batch_start, batch_end = i, i + BATCH_SIZE
        image_batch = image_paths[batch_start:batch_end]
        
        if not image_batch: continue
            
        logging.info(f"--> Processing Batch: Pages {image_batch[0].stem.split('_')[-1]} to {image_batch[-1].stem.split('_')[-1]}")
        
        # Step 1: Extract structured text from images
        extracted_text_batch = extract_text_from_document(image_batch, vision_model)
        
        # --- STEP 2 (NEW): Refine the extracted text using the text model ---
        refined_text_batch = refine_text_with_llm(extracted_text_batch, text_model)
        
        all_refined_text_parts.append(refined_text_batch)

    full_structured_text = "\n\n---\n\n".join(all_refined_text_parts)
    # The preprocessor now cleans the AI-refined markdown text
    cleaned_full_text = clean_document_text(full_structured_text)

    output_data = {
        "pdf_filename": f"{target_pdf_path.name}",
        "total_pages": len(image_paths),
        "cleaned_full_text": cleaned_full_text
    }

    json_output_path = config.PROCESSED_TEXT_DIR / f"{target_pdf_path.stem}.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    logging.info(f"✅ Successfully saved refined and cleaned text to: {json_output_path}")

    logging.info("======================================================")
    logging.info("    Project Danesh: FOCUSED EXPERIMENT Harvester Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
