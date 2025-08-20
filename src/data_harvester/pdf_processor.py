# src/data_harvester/pdf_processor.py

from pathlib import Path
from pdf2image import convert_from_path
import logging

def convert_pdfs_to_images(pdf_dir: Path, image_dir: Path):
    """
    Converts all PDFs in a directory to PNG images, page by page.
    """
    logging.info("Starting PDF to image conversion...")
    for pdf_path in pdf_dir.glob("*.pdf"):
        pdf_name = pdf_path.stem
        pdf_image_dir = image_dir / pdf_name
        pdf_image_dir.mkdir(exist_ok=True)

        if any(pdf_image_dir.iterdir()):
            logging.info(f"Images for '{pdf_name}' already exist. Skipping.")
            continue

        logging.info(f"Processing '{pdf_name}'...")
        try:
            images = convert_from_path(pdf_path, fmt='png')
            for i, image in enumerate(images):
                image_path = pdf_image_dir / f"page_{i+1}.png"
                image.save(image_path, 'PNG')
            logging.info(f"Converted {len(images)} pages for '{pdf_name}'")
        except Exception as e:
            logging.error(f"Error converting {pdf_path}: {e}")
