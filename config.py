# config.py
# --- MODIFIED FOR LOCAL EXECUTION ---

from pathlib import Path

# --- Base Paths ---
# آدرس ریشه پروژه شما
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- NEW: Base path for all our local data ---
# تمام داده‌های پروژه در این پوشه ذخیره خواهند شد
DATA_BASE_PATH = PROJECT_ROOT / "data"

# --- Project-Relative Directories (Located in the Git Repo) ---
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# --- Local Data Directories (Located in D:\ProjectDanesh\data) ---
RAW_PDFS_DIR = DATA_BASE_PATH / "raw_pdfs"

# --- V1 (Original) Data Directories ---
IMAGES_DIR = DATA_BASE_PATH / "images"
PROCESSED_TEXT_DIR = DATA_BASE_PATH / "processed_text"

# --- V2 (Donut Experiment) Data Directories ---
IMAGES_DIR_DONUT = DATA_BASE_PATH / "images_donut"
PROCESSED_TEXT_DIR_DONUT = DATA_BASE_PATH / "processed_text_donut"

# --- Common Knowledge & Vector DB Directories ---
KNOWLEDGE_GRAPH_DIR = DATA_BASE_PATH / "knowledge_graph"
VECTOR_DB_DIR = DATA_BASE_PATH / "vector_db"
KNOWLEDGE_BASE_DIR = DATA_BASE_PATH / "knowledge_base"

# --- NEW: Directory for optimization data ---
OPTIMIZATION_DATA_DIR = DATA_BASE_PATH / "optimization_data"


# --- API & MODEL CONFIGURATION ---
GEMINI_VISION_MODEL_NAME = 'gemini-1.5-pro-latest'
GEMINI_TEXT_MODEL_NAME = 'gemini-1.5-flash'
GEMINI_EMBEDDING_MODEL_NAME = 'models/text-embedding-004'
GEMINI_GENERATION_MODEL_NAME = 'gemini-1.5-flash'

# --- PROMPT ENGINEERING ---
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"
REFINEMENT_PROMPT_PATH = PROMPTS_DIR / "refinement_prompt.txt"

# --- RAG PIPELINE CONFIGURATION ---
CHROMA_COLLECTION_NAME = "ikiu_regulations"
VECTOR_SEARCH_TOP_K = 10
RERANK_TOP_N = 3