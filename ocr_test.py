# ocr_test.py
# A dedicated script for testing the Donut OCR model on a single image.
# This helps in isolating and debugging OCR issues without running the full pipeline.

import os
from dotenv import load_dotenv
from pathlib import Path

# --- IMPORTANT: Ensure you have run 'pip install Pillow transformers torch sentencepiece' in your .venv ---

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# لطفا نام پوشه عکس‌ها و شماره صفحه مورد نظر را در اینجا تنظیم کنید
# این پوشه باید در data/images_donut وجود داشته باشد
PDF_NAME = "کتاب آیین نامه ی ارتقای مرتبه اعضای هیأت علمی آموزشی، پژوهشی و فناوری"
PAGE_NUMBER = 42  # شماره صفحه‌ای که می‌خواهید تست کنید

# --- Do not change the code below this line ---

# Dynamically build the image path
# Note: The image filenames are 1-based and padded to 3 digits (e.g., page_042.png)
IMAGE_FILENAME = f"page_{PAGE_NUMBER:03d}.png"
IMAGE_PATH_TO_TEST = Path(__file__).parent / "data" / "images_donut" / PDF_NAME / IMAGE_FILENAME

# Import the OCR function from our project's src directory
# This ensures we are testing the exact same code used in the main harvester
try:
    from src.data_harvester.ocr import extract_text_with_donut
except ImportError:
    print("FATAL: Could not import the OCR function. Make sure you are running this script from the project root.")
    exit()

def run_single_image_test():
    """
    Runs the OCR process on the single image specified in the configuration.
    """
    print("--- Starting Single Page OCR Test ---")

    if not IMAGE_PATH_TO_TEST.exists():
        print(f"Error: Image file not found at the specified path!")
        print(f"Searched for: {IMAGE_PATH_TO_TEST}")
        print("\nPlease ensure that:")
        print("1. You have already run the main harvester at least once to generate the images.")
        print("2. The PDF_NAME and PAGE_NUMBER variables are set correctly.")
        return

    print(f"Target Image: {IMAGE_PATH_TO_TEST}")
    print("Running extraction... This may take a moment as the model needs to be loaded.")

    try:
        # Call the OCR function
        extracted_text = extract_text_with_donut(IMAGE_PATH_TO_TEST)

        print("\n--- OCR Extraction Result ---")
        print(extracted_text)
        print("-----------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred during the OCR process: {e}")
        print("This could be due to issues with model loading or the CUDA environment.")

if __name__ == "__main__":
    run_single_image_test()