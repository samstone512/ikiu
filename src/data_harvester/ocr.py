# src/data_harvester/ocr.py
# OPTIMIZATION-V4.2: Added safety settings to prevent premature truncation of the output.

import logging
import time
from pathlib import Path
from typing import List
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import the specific prompt from our central config file
from config import OCR_PROMPT

def extract_text_from_document(image_paths: List[Path], model) -> str:
    """
    Performs OCR on a whole document by sending all its page images in a single request.
    This allows the model to understand the context across pages, like reconstructing broken tables.
    """
    if not image_paths:
        logging.warning("No image paths provided to OCR function. Returning empty string.")
        return ""

    try:
        logging.info(f"  - Processing {len(image_paths)} pages for a single document...")
        
        request_content = [OCR_PROMPT]
        for image_path in image_paths:
            img = Image.open(image_path)
            request_content.append(img)

        # --- KEY CHANGE: Add safety settings to prevent the API from stopping early ---
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Send the single, comprehensive request to the Gemini API with safety settings.
        response = model.generate_content(
            request_content,
            safety_settings=safety_settings
        )

        time.sleep(2)

        if response and response.text:
            logging.info(f"  - Successfully extracted structured text from the entire document.")
            return response.text.strip()
        else:
            # Added more detailed logging for when the response is empty
            logging.warning(f"  - Gemini API returned an empty or blocked response. Prompt feedback: {response.prompt_feedback}")
            return ""

    except Exception as e:
        logging.error(f"  - An error occurred while processing the document. Error: {e}", exc_info=True)
        return ""
