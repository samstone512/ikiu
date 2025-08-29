# src/data_harvester/ocr.py
# OPTIMIZATION-V4: This module is upgraded to process a whole document (all its page images)
# in a single API call to leverage the multi-page context understanding of Gemini 1.5 Pro.

import logging
import time
from pathlib import Path
from typing import List
from PIL import Image
import google.generativeai as genai

# Import the specific prompt from our central config file
from config import OCR_PROMPT

def extract_text_from_document(image_paths: List[Path], model) -> str:
    """
    Performs OCR on a whole document by sending all its page images in a single request.
    This allows the model to understand the context across pages, like reconstructing broken tables.

    Args:
        image_paths (List[Path]): A sorted list of paths to the image files of the document pages.
        model: The initialized Gemini Pro Vision model instance.

    Returns:
        str: The extracted and structured text (in Markdown) for the entire document.
             Returns an empty string if an error occurs.
    """
    if not image_paths:
        logging.warning("No image paths provided to OCR function. Returning empty string.")
        return ""

    try:
        logging.info(f"  - Processing {len(image_paths)} pages for a single document...")
        
        # --- KEY CHANGE: Build a single request with the prompt and ALL images ---
        # The model expects a list where the first element is the prompt
        # and the subsequent elements are the images.
        request_content = [OCR_PROMPT]
        for image_path in image_paths:
            img = Image.open(image_path)
            request_content.append(img)

        # Send the single, comprehensive request to the Gemini API.
        response = model.generate_content(request_content)

        # A short delay is still good practice.
        time.sleep(2)

        if response and response.text:
            logging.info(f"  - Successfully extracted structured text from the entire document.")
            return response.text.strip()
        else:
            logging.warning(f"  - Gemini API returned an empty response for the document.")
            return ""

    except Exception as e:
        # Catch any potential errors during the API call or image processing.
        logging.error(f"  - An error occurred while processing the document. Error: {e}", exc_info=True)
        return ""
