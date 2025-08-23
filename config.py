import os

# --- Google Drive Paths ---
# Base path for the project in Google Drive
# Note: You'll need to mount your Google Drive in Colab and adjust this base path.
# Example: '/content/drive/MyDrive/Danesh_Project_v2'
DRIVE_BASE_PATH = '/content/drive/MyDrive/Danesh_Project_v2' 

# Data paths
DATA_PATH = os.path.join(DRIVE_BASE_PATH, 'data/')
PROCESSED_DATA_PATH = os.path.join(DRIVE_BASE_PATH, 'processed_data/')

# Artifact paths
KNOWLEDGE_GRAPH_PATH = os.path.join(DRIVE_BASE_PATH, 'knowledge_graph/')
VECTOR_STORE_PATH = os.path.join(DRIVE_BASE_PATH, 'vector_store/')

# --- Model & API Keys ---
# Securely load the Gemini API Key from Google Colab's secrets.
# This prevents exposing the key directly in the code.
GEMINI_API_KEY = None
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    if GEMINI_API_KEY is None:
        print("Warning: 'GEMINI_API_KEY' not found in Colab secrets.")
except ImportError:
    print("Warning: Not in a Colab environment. API key loading from secrets is skipped.")

# --- Document Processing ---
# Settings for Tesseract OCR, if needed
# Example: TESSERACT_CMD_PATH = '/usr/bin/tesseract' # for Linux/Colab

# --- Intelligent Chunking Parameters ---
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# --- FAISS Vector Store ---
FAISS_INDEX_NAME = "faiss_index.bin"

# --- Knowledge Graph ---
GRAPH_FILE_NAME = "knowledge_graph.gml"
