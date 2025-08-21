# main_harvester.py
# Main executable script for the Data Harvester phase of Project Danesh.

import os
import sys
import json
import logging
from pathlib import Path

# This special block handles running in Google Colab and importing the API key.
try:
    from google.colab import userdata
    import google.generativeai as genai
except ImportError:
    # This helps prevent errors if run outside of Colab, though it's designed for it.
    pass

# Import project-specific modules and configurations
import config
from src.data_harvester.pdf_processor import convert_pdfs_to_images
from src.data_harvester.ocr import extract_text_from_image

# --- 1. SETUP LOGGING ---
# Configure logging to provide clear, step-by-step status updates in the Colab output.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to orchestrate the entire data harvesting pipeline."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Phase 01 - Data Harvester    ")
    logging.info("======================================================")

    # --- 2. CONFIGURE GEMINI API ---
    # Securely read the API key from Colab's secret manager.
    try:
        api_key = userdata.get('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' not found in Colab secrets. Please add it.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        logging.info("Successfully configured Google Gemini API.")
    except NameError:
        logging.error("FATAL: Script is not running in a Google Colab environment or 'userdata' is unavailable.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        sys.exit(1)

    # --- 3. CREATE DATA DIRECTORIES ON GOOGLE DRIVE ---
    try:
        logging.info(f"Ensuring data directories exist in Google Drive at: {config.DRIVE_BASE_PATH}")
        config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
        config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Data directories are ready.")
    except Exception as e:
        logging.error(f"FATAL: Could not create directories on Google Drive. Check permissions. Error: {e}")
        sys.exit(1)

    # --- 4. CHECK FOR SOURCE PDFS ---
    pdf_files = list(config.RAW_PDFS_DIR.glob('*.pdf'))
    if not pdf_files:
        logging.error(f"FATAL: No PDF files found in the source directory: {config.RAW_PDFS_DIR}")
        logging.error("Please upload your PDF files to this directory in Google Drive and run the script again.")
        sys.exit(1)
    logging.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    # --- 5. RUN PDF-TO-IMAGE CONVERSION ---
    convert_pdfs_to_images(config.RAW_PDFS_DIR, config.IMAGES_DIR)

    # --- 6. RUN OCR AND SAVE JSON OUTPUTS ---
    logging.info("--- Starting OCR Processing for All Converted Images ---")
    image_folders = [d for d in config.IMAGES_DIR.iterdir() if d.is_dir()]

    for folder in image_folders:
        pdf_name = folder.name
        json_output_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"

        # Optimization: Skip if the final JSON already exists.
        if json_output_path.exists():
            logging.info(f"JSON for '{pdf_name}' already exists. Skipping OCR.")
            continue

        logging.info(f"Processing images for PDF: '{pdf_name}'")
        image_paths = sorted(list(folder.glob('*.png')), key=lambda p: int(p.stem.split('_')[-1]))
        if not image_paths:
            logging.warning(f"No images found in {folder}, skipping.")
            continue

        output_data = {
            "pdf_filename": f"{pdf_name}.pdf",
            "total_pages": len(image_paths),
            "pages": []
        }

        for image_path in image_paths:
            extracted_text = extract_text_from_image(image_path, model)
            page_number = int(image_path.stem.split('_')[-1])
            output_data["pages"].append({
                "page_number": page_number,
                "image_filename": image_path.name,
                "extracted_text": extracted_text
            })

        # Save the final structured data to a JSON file.
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved extracted text to: {json_output_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON file for '{pdf_name}'. Error: {e}")

    logging.info("======================================================")
    logging.info("    Project Danesh: Data Harvester Phase Completed!     ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
