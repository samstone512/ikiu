# config.py
# Centralized configuration file for Project Danesh.
# --- OPTIMIZATION-V4: Multi-page, context-aware OCR prompt for whole-document understanding ---

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
# --- NEW, WHOLE-DOCUMENT AWARE OCR PROMPT ---
OCR_PROMPT = """
You are an expert document digitization AI. You will be given a sequence of images that represent the pages of a single document. Your task is to analyze all pages together and produce one single, coherent, and clean Markdown text for the entire document.

**CRITICAL INSTRUCTIONS:**
1.  **Acknowledge Multi-Page Context:** Understand that paragraphs, lists, and especially **tables** can be split across page breaks. Your primary goal is to intelligently stitch these broken elements together.
2.  **Reconstruct Broken Tables:** If a table starts on one page and continues on the next, you MUST merge them into a single, complete Markdown table in your final output. Do not output two separate, incomplete tables. Preserve the headers and row continuity.
3.  **Maintain Logical Flow:** Ensure the text flows logically from one page to the next without interruption.
4.  **Format as Clean Markdown:** Use Markdown for headings, lists, and tables.
    Example of a correctly reconstructed Markdown Table:
    | هدر ۱ | هدر ۲ | هدر ۳ |
    |---|---|---|
    | ردیف ۱، ستون ۱ (از صفحه ۱) | ردیف ۱، ستون ۲ (از صفحه ۱) | ردیف ۱، ستون ۳ (از صفحه ۱) |
    | ردیف ۲، ستون ۱ (ادامه در صفحه ۲) | ردیف ۲، ستون ۲ (ادامه در صفحه ۲) | ردیف ۲، ستون ۳ (ادامه در صفحه ۲) |
5.  **Do NOT add any commentary.** Your output should be the single, final Markdown text for the entire document.
"""
ENTITY_EXTRACTION_PROMPT_PATH = PROMPTS_DIR / "entity_extraction.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"

# --- RAG PIPELINE CONFIGURATION ---
CHROMA_COLLECTION_NAME = "ikiu_regulations"
VECTOR_SEARCH_TOP_K = 10
GRAPH_SEARCH_DEPTH = 2
RERANK_TOP_N = 3
