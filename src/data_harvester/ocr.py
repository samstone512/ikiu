# src/data_harvester/ocr.py
# Module for performing OCR on images using the Gemini Pro Vision API.

import logging
import time
from pathlib import Path
from PIL import Image
import google.generativeai as genai

# Import the specific prompt from our central config file
from config import OCR_PROMPT

def extract_text_from_image(image_path: Path, model) -> str:
    """
    Performs OCR on a single image file using the configured Gemini model.

    Args:
        image_path (Path): The path to the image file.
        model: The initialized Gemini Pro Vision model instance.

    Returns:
        str: The extracted text from the image. Returns an empty string if an
             error occurs or if the API returns no text.
    """
    try:
        logging.info(f"  - Processing image: {image_path.name}")
        img = Image.open(image_path)

        # Send the request to the Gemini API with the image and our specific prompt.
        # The model expects a list containing the prompt and the image object.
        response = model.generate_content([OCR_PROMPT, img])

        # IMPORTANT: Add a small delay to respect API rate limits and avoid errors.
        time.sleep(2)

        if response and response.text:
            logging.debug(f"Successfully extracted text from {image_path.name}.")
            return response.text.strip()
        else:
            logging.warning(f"Gemini API returned an empty response for {image_path.name}.")
            return ""

    except Exception as e:
        # Catch any potential errors during the API call or image processing.
        logging.error(f"An error occurred while processing '{image_path.name}'. Error: {e}")
        return ""
