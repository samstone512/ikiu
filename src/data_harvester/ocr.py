  # src/data_harvester/ocr.py

import time
from pathlib import Path
from PIL import Image
import google.generativeai as genai
import logging

def extract_text_from_image(image_path: Path, model) -> str:
    """
    Sends an image to the Gemini Pro Vision API and returns extracted text.
    """
    logging.info(f"  - Extracting text from '{image_path.name}'...")
    try:
        image = Image.open(image_path)
        prompt = "You are an expert OCR system. Extract all the Persian text from this image exactly as it appears. Do not add any commentary or explanation. Just provide the raw text."
        response = model.generate_content([prompt, image])
        time.sleep(2) # Respect API rate limits
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API Error for {image_path.name}: {e}")
        return f"Error: Could not extract text. Details: {e}"
