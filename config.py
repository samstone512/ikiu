# config.py
# Centralized configuration file for Project Danesh.

from pathlib import Path

# --- Base Paths ---
# We define two separate base paths for clarity and portability:
# 1. PROJECT_ROOT: The root directory of the Git repository. This is used for
#    locating project files like prompts. It makes the code independent of where it's run.
# 2. DRIVE_BASE_PATH: The specific path on Google Drive where all runtime data
#    (PDFs, images, JSONs, databases) is stored. This is specific to the Colab environment.

PROJECT_ROOT = Path(__file__).parent.resolve()
DRIVE_BASE_PATH = Path("/content/drive/MyDrive/IKIU")

# --- Project-Relative Directories (Located in the Git Repo) ---
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# --- Google Drive Data Directories (Located in Google Drive) ---
RAW_PDFS_DIR = DRIVE_BASE_PATH / "raw_pdfs"
IMAGES_DIR = DRIVE_BASE_PATH / "images"
PROCESSED_TEXT_DIR = DRIVE_BASE_PATH / "processed_text"
KNOWLEDGE_GRAPH_DIR = DRIVE_BASE_PATH / "knowledge_graph"
VECTOR_DB_DIR = DRIVE_BASE_PATH / "vector_db"

# --- API & MODEL CONFIGURATION ---
# Phase 01: Model for OCR (Vision)
GEMINI_VISION_MODEL_NAME = 'gemini-pro-vision'

# Phase 02: Models for Text Analysis and Embedding
GEMINI_TEXT_MODEL_NAME = 'gemini-1.5-flash'
GEMINI_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'

# --- PROMPT ENGINEERING ---
# Phase 01: Prompt for OCR
OCR_PROMPT = "You are an expert OCR system. Extract all the Persian text from this image exactly as it appears. Do not add any commentary or explanation. Just provide the raw text."

# Phase 02: Path to the prompt file for Entity and Relationship Extraction
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
