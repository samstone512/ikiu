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
# This section is updated to be more robust and portable.
# It prioritizes environment variables, which is a best practice.
# If the environment variable is not set, it falls back to Colab secrets.

try:
    from google.colab import userdata
except ImportError:
    userdata = None

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

if GEMINI_API_KEY is None and userdata:
    print("API key not found in environment variables. Trying to read from Colab secrets...")
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

if GEMINI_API_KEY is None:
    print("Warning: 'GEMINI_API_KEY' not found as an environment variable or in Colab secrets.")


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
