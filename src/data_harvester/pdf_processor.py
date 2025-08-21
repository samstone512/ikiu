# src/data_harvester/pdf_processor.py
# Module for converting PDF files to a series of PNG images.

import logging
from pathlib import Path
from pdf2image import convert_from_path

def convert_pdfs_to_images(pdf_dir: Path, image_dir: Path):
    """
    Iterates through all PDF files in a directory, converts each page to a PNG image,
    and saves them into a structured output directory.

    This function is designed to be robust:
    - It logs the start and end of the process.
    - It skips PDFs that have already been converted to avoid redundant work.
    - It handles potential errors during the conversion of a single PDF gracefully
      and continues with the next one.
    """
    logging.info("--- Starting PDF to Image Conversion Process ---")

    # Find all PDF files in the source directory.
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in {pdf_dir}. Skipping conversion.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_path in pdf_files:
        try:
            # Create a specific sub-directory for this PDF's images.
            # The folder name is derived from the PDF filename without its extension.
            pdf_stem = pdf_path.stem
            output_folder = image_dir / pdf_stem
            output_folder.mkdir(exist_ok=True)

            # Optimization: Check if images already exist to avoid reprocessing.
            if any(output_folder.glob("*.png")):
                logging.info(f"Images for '{pdf_path.name}' already exist. Skipping.")
                continue

            logging.info(f"Processing '{pdf_path.name}'...")

            # Perform the conversion. This can be time-consuming for large PDFs.
            images = convert_from_path(pdf_path, fmt='png')

            # Save each page as a consistently named PNG file.
            for i, image in enumerate(images):
                # Naming convention: page_001.png, page_002.png, etc.
                image_name = f"page_{i+1:03d}.png"
                image_save_path = output_folder / image_name
                image.save(image_save_path, 'PNG')

            logging.info(f"Successfully converted {len(images)} pages from '{pdf_path.name}'.")

        except Exception as e:
            logging.error(f"Failed to process PDF file '{pdf_path.name}'. Error: {e}")
            # Continue to the next file even if one fails.
            continue

    logging.info("--- Finished PDF to Image Conversion Process ---")
