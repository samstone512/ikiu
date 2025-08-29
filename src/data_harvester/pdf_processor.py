# src/data_harvester/pdf_processor.py
# Module for converting PDF files to a series of PNG images.
# --- OPTIMIZATION-V4.1: Modified to accept a specific list of files to process ---

import logging
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path

def convert_pdfs_to_images(pdf_dir: Path, image_dir: Path, specific_files: Optional[List[Path]] = None):
    """
    Iterates through PDF files and converts them to images.
    If 'specific_files' is provided, it only processes those files.
    Otherwise, it processes all PDFs in the pdf_dir.
    """
    logging.info("--- Starting PDF to Image Conversion Process ---")

    # --- KEY CHANGE: Use the specific list if provided, otherwise scan the directory ---
    if specific_files:
        pdf_files = specific_files
        logging.info(f"Processing {len(pdf_files)} specific PDF file(s).")
    else:
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logging.warning(f"No PDF files found in {pdf_dir}. Skipping conversion.")
            return
        logging.info(f"Found {len(pdf_files)} PDF(s) to process in the directory.")

    for pdf_path in pdf_files:
        try:
            pdf_stem = pdf_path.stem
            output_folder = image_dir / pdf_stem
            output_folder.mkdir(exist_ok=True)

            if any(output_folder.glob("*.png")):
                logging.info(f"Images for '{pdf_path.name}' already exist. Skipping.")
                continue

            logging.info(f"Processing '{pdf_path.name}'...")
            images = convert_from_path(pdf_path, fmt='png')

            for i, image in enumerate(images):
                image_name = f"page_{i+1:03d}.png"
                image_save_path = output_folder / image_name
                image.save(image_save_path, 'PNG')

            logging.info(f"Successfully converted {len(images)} pages from '{pdf_path.name}'.")

        except Exception as e:
            logging.error(f"Failed to process PDF file '{pdf_path.name}'. Error: {e}")
            continue

    logging.info("--- Finished PDF to Image Conversion Process ---")
