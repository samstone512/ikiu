import os
from pathlib import Path

class Config:
    # Project Root Directory
    ROOT_DIR = Path(__file__).parent

    # Data Directories
    DATA_DIR = ROOT_DIR / 'data'
    PDF_DIR = DATA_DIR / 'pdf'
    TEXT_DIR = DATA_DIR / 'text'
    JSON_DIR = DATA_DIR / 'json'
    DONUT_OUTPUT_DIR = DATA_DIR / 'donut_output'
    DOCLING_OUTPUT_DIR = DATA_DIR / 'docling_output' # New directory for Docling JSON outputs

    # Ensure directories exist
    @staticmethod
    def create_dirs():
        for dir_path in [
            Config.DATA_DIR,
            Config.PDF_DIR,
            Config.TEXT_DIR,
            Config.JSON_DIR,
            Config.DONUT_OUTPUT_DIR,
            Config.DOCLING_OUTPUT_DIR # Ensure the new directory is created
        ]:
            os.makedirs(dir_path, exist_ok=True)

# Automatically create directories when this module is imported
Config.create_dirs()

if __name__ == '__main__':
    # You can run this file directly to see the configured paths
    print(f"Project Root: {Config.ROOT_DIR}")
    print(f"Data Directory: {Config.DATA_DIR}")
    print(f"PDF Directory: {Config.PDF_DIR}")
    print(f"Docling Output Directory: {Config.DOCLING_OUTPUT_DIR}")
    print("All necessary directories have been checked/created.")