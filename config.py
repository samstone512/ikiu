# config.py
# Centralized configuration file for Project Danesh.

from pathlib import Path

# --- Base Paths ---
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
GEMINI_VISION_MODEL_NAME = 'gemini-1.5-pro-latest'
GEMINI_TEXT_MODEL_NAME = 'gemini-1.5-flash'
GEMINI_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
GEMINI_GENERATION_MODEL_NAME = 'gemini-1.5-flash'

# --- PROMPT ENGINEERING ---
OCR_PROMPT = "You are an expert OCR system. Extract all the Persian text from this image exactly as it appears. Do not add any commentary or explanation. Just provide the raw text."
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"

# --- RAG PIPELINE CONFIGURATION ---
CHROMA_COLLECTION_NAME = "ikiu_regulations"
# --- UPDATED: Retrieve more candidates for re-ranking ---
VECTOR_SEARCH_TOP_K = 10 
GRAPH_SEARCH_DEPTH = 2
# --- ADDED: Number of documents to keep after re-ranking ---
RERANK_TOP_N = 3
