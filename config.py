# config.py
# Centralized configuration file for Project Danesh.

from pathlib import Path

# --- CRITICAL SETUP: GOOGLE DRIVE INTEGRATION ---
# This configuration is now set to your specific Google Drive path.
# All data will be read from and written to this directory.
DRIVE_BASE_PATH = Path("/content/drive/MyDrive/IKIU")

# --- SUB-DIRECTORY DEFINITIONS ---
# Defines the folder structure for our data within your IKIU folder.
# The main script will create these folders if they don't exist.

# 1. RAW_PDFS_DIR:
#    !!! IMPORTANT !!!
#    Place your source PDF files inside this directory before running the script.
#    Path: /content/drive/MyDrive/IKIU/raw_pdfs/
RAW_PDFS_DIR = DRIVE_BASE_PATH / "raw_pdfs"

# 2. IMAGES_DIR:
#    This directory will store the PNG images converted from the PDF pages.
#    Path: /content/drive/MyDrive/IKIU/images/
IMAGES_DIR = DRIVE_BASE_PATH / "images"

# 3. PROCESSED_TEXT_DIR:
#    The final extracted text will be saved here as structured JSON files.
#    Path: /content/drive/MyDrive/IKIU/processed_text/
PROCESSED_TEXT_DIR = DRIVE_BASE_PATH / "processed_text"

# --- API & MODEL CONFIGURATION ---
# Defines the model we'll use for OCR.
GEMINI_MODEL_NAME = 'gemini-pro-vision'

# The prompt sent to the Gemini API for text extraction.
# This prompt is designed to get only the raw Persian text without extra commentary.
OCR_PROMPT = "You are an expert OCR system. Extract all the Persian text from this image exactly as it appears. Do not add any commentary or explanation. Just provide the raw text."
