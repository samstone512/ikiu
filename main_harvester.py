# main_harvester.py

import os
import json
import logging
from google.colab import userdata # To securely get API key in Colab
import google.generativeai as genai

# Import configurations and functions from our modules
import config
from src.data_harvester import crawler, pdf_processor, ocr

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function to run the entire data harvesting pipeline."""
    logging.info("==============================================")
    logging.info("         Starting Phase 1: Data Harvester     ")
    logging.info("==============================================")

    # --- 1. API Key Configuration ---
    try:
        GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Google API Key configured successfully.")
        vision_model = genai.GenerativeModel('gemini-pro-vision')
    except Exception as e:
        logging.error(f"Could not configure Google API: {e}")
        logging.error("Please make sure you have set the 'GOOGLE_API_KEY' secret in Colab.")
        return

    # --- 2. Create Directories ---
    config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Data directories are ready.")

    # --- 3. Crawl for PDFs (Optional) ---
    # Note: For this phase, we will manually upload PDFs.
    # The crawler function is available for future use.
    # To use it, uncomment the following line and set the URL in config.py
    # crawler.crawl_and_download_pdfs(config.CRAWL_URL, config.UNIVERSITY_DOMAIN, config.RAW_PDFS_DIR)
    logging.warning("Skipping web crawl. Please manually upload PDFs to the 'data/raw_pdfs' folder in Colab.")
    
    # Check if there are any PDFs to process
    if not any(config.RAW_PDFS_DIR.glob("*.pdf")):
        logging.error("No PDFs found in 'data/raw_pdfs'. Please upload files and restart the script.")
        return

    # --- 4. Convert PDFs to Images ---
    pdf_processor.convert_pdfs_to_images(config.RAW_PDFS_DIR, config.IMAGES_DIR)

    # --- 5. OCR Images and Save Text ---
    logging.info("Starting text extraction from images...")
    for pdf_folder in config.IMAGES_DIR.iterdir():
        if not pdf_folder.is_dir():
            continue

        pdf_name = pdf_folder.name
        output_json_path = config.PROCESSED_TEXT_DIR / f"{pdf_name}.json"
        
        if output_json_path.exists():
            logging.info(f"JSON for '{pdf_name}' already exists. Skipping.")
            continue

        all_pages_data = []
        image_files = sorted(list(pdf_folder.glob("*.png")), key=lambda p: int(p.stem.split('_')[1]))

        for image_path in image_files:
            page_num = int(image_path.stem.split('_')[1])
            extracted_text = ocr.extract_text_from_image(image_path, vision_model)
            page_data = {
                "pdf_name": f"{pdf_name}.pdf",
                "page": page_num,
                "text": extracted_text
            }
            all_pages_data.append(page_data)

        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_pages_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully saved extracted text to '{output_json_path}'")
        except IOError as e:
            logging.error(f"Error writing JSON file for {pdf_name}: {e}")

    logging.info("==============================================")
    logging.info("          Phase 1: Data Harvester Complete    ")
    logging.info(f"Processed text files are in: {config.PROCESSED_TEXT_DIR}")
    logging.info("==============================================")


if __name__ == "__main__":
    main()
