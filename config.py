# config.py
# Centralized configuration file for Project Danesh.
# --- OPTIMIZATION-V3: Upgrading OCR prompt to be structure-aware (especially for tables) ---

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
# --- NEW POWERFUL, STRUCTURE-AWARE OCR PROMPT ---
OCR_PROMPT = """
You are an expert document digitization assistant. Your task is to analyze the provided image of a document page and convert its content into clean, structured Markdown text. Pay close attention to tables.

**Instructions:**
1.  **Extract all Persian text.**
2.  **Preserve Structure:** Maintain the original structure of headings, lists, and paragraphs.
3.  **Convert Tables to Markdown:** This is the most important instruction. If you detect a table, you MUST represent it using Markdown table format. Do not just extract the text line-by-line. Capture the rows and columns accurately.
    Example of a Markdown Table:
    | هدر ۱ | هدر ۲ | هدر ۳ |
    |---|---|---|
    | ردیف ۱، ستون ۱ | ردیف ۱، ستون ۲ | ردیف ۱، ستون ۳ |
    | ردیف ۲، ستون ۱ | ردیف ۲، ستون ۲ | ردیف ۲، ستون ۳ |
4.  **Do NOT add any commentary or explanation.** Your output should only be the clean Markdown text representing the document's content.
"""
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"

# --- RAG PIPELINE CONFIGURATION ---
CHROMA_COLLECTION_NAME = "ikiu_regulations"
VECTOR_SEARCH_TOP_K = 10 
GRAPH_SEARCH_DEPTH = 2
RERANK_TOP_N = 3
