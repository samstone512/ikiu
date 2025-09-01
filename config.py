# config.py
# --- PHASE 6: Adding refinement prompt path ---

from pathlib import Path

# --- Base Paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DRIVE_BASE_PATH = Path("/content/drive/MyDrive/IKIU")

# --- Project-Relative Directories (Located in the Git Repo) ---
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# --- Google Drive Data Directories (Located in Google Drive) ---
RAW_PDFS_DIR = DRIVE_BASE_PATH / "raw_pdfs"

# --- V1 (Original) Data Directories ---
IMAGES_DIR = DRIVE_BASE_PATH / "images"
PROCESSED_TEXT_DIR = DRIVE_BASE_PATH / "processed_text"

# --- V2 (Donut Experiment) Data Directories ---
IMAGES_DIR_DONUT = DRIVE_BASE_PATH / "images_donut"
PROCESSED_TEXT_DIR_DONUT = DRIVE_BASE_PATH / "processed_text_donut"

# --- Common Knowledge & Vector DB Directories ---
KNOWLEDGE_GRAPH_DIR = DRIVE_BASE_PATH / "knowledge_graph"
VECTOR_DB_DIR = DRIVE_BASE_PATH / "vector_db"
KNOWLEDGE_BASE_DIR = DRIVE_BASE_PATH / "knowledge_base"

# --- API & MODEL CONFIGURATION ---
GEMINI_VISION_MODEL_NAME = 'gemini-1.5-pro-latest'
GEMINI_TEXT_MODEL_NAME = 'gemini-1.5-flash'
GEMINI_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
GEMINI_GENERATION_MODEL_NAME = 'gemini-1.5-flash'

# --- PROMPT ENGINEERING ---
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"
# --- NEW: Path to the refinement prompt ---
REFINEMENT_PROMPT_PATH = PROMPTS_DIR / "refinement_prompt.txt"

# --- RAG PIPELINE CONFIGURATION ---
CHROMA_COLLECTION_NAME = "ikiu_regulations"
VECTOR_SEARCH_TOP_K = 10
RERANK_TOP_N = 3
