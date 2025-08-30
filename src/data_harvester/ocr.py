# src/data_harvester/ocr.py
# --- FINAL VERSION: Using the robust and reliable Tesseract OCR engine ---

import logging
from pathlib import Path
from PIL import Image
import pytesseract
import cv2

def preprocess_image_for_ocr(image_path: Path):
    """
    Improves image quality for better OCR results using OpenCV.
    """
    try:
        img = cv2.imread(str(image_path))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get a binary image
        _, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return Image.fromarray(binary_img)
    except Exception as e:
        logging.warning(f"  - Could not preprocess image {image_path.name}. Using original. Error: {e}")
        return Image.open(image_path)

def extract_text_from_image_tesseract(image_path: Path) -> str:
    """
    Performs OCR on a single image file using the Tesseract engine.
    """
    try:
        logging.info(f"  - Processing image with Tesseract: {image_path.name}")
        
        # 1. Preprocess the image to improve quality
        preprocessed_img = preprocess_image_for_ocr(image_path)
        
        # 2. Use Tesseract to extract text, specifying Persian language
        text = pytesseract.image_to_string(preprocessed_img, lang='fas')
        
        logging.debug(f"Successfully extracted text from {image_path.name}.")
        return text.strip()

    except Exception as e:
        logging.error(f"  - An error occurred while processing '{image_path.name}' with Tesseract. Error: {e}")
        return ""
