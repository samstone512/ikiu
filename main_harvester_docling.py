import logging
from pathlib import Path
from docling import Docling
from config import Config

# --- Setup Logging ---
# A professional setup for logging to see progress and potential issues.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def process_document_with_docling(pdf_path: Path, output_dir: Path, config_path: Path):
    """
    Processes a single PDF document using the specified Docling pipeline.

    Args:
        pdf_path (Path): The full path to the input PDF file.
        output_dir (Path): The directory where the output JSON file will be saved.
        config_path (Path): The path to the docling_config.yml file.
    """
    if not pdf_path.exists():
        logging.error(f"Input PDF not found at: {pdf_path}")
        return

    if not config_path.exists():
        logging.error(f"Docling config file not found at: {config_path}")
        return

    # Define the output path for the JSON file to check if it already exists
    output_json_path = output_dir / f"{pdf_path.stem}.json"
    if output_json_path.exists():
        logging.warning(f"Output file {output_json_path.name} already exists. Skipping.")
        return

    logging.info(f"Starting to process document: {pdf_path.name}")
    logging.info(f"Using config file: {config_path.name}")

    try:
        # Initialize Docling with the path to the config file
        docling_processor = Docling.from_yaml(config_path)
        
        logging.info(f"Output will be saved to: {output_json_path}")

        # Run the pipeline
        # The 'path' argument is the key for the input file.
        # The 'output_path' argument tells the JsonWriter where to save the file.
        docling_processor.run(
            pipeline_name="pdf_to_json_ocr_fa",
            path=pdf_path,
            output_path=output_json_path
        )
        
        logging.info(f"✅ Successfully processed and saved output for: {pdf_path.name}")

    except Exception as e:
        logging.error(f"❌ An error occurred while processing {pdf_path.name}: {e}", exc_info=True)


if __name__ == "__main__":
    # --- Main Execution (Now processes all PDFs in the directory) ---
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Docling Harvester    ")
    logging.info("======================================================")

    # Find all PDF files in the designated PDF directory
    pdf_files_to_process = list(Config.PDF_DIR.glob("*.pdf"))
    
    if not pdf_files_to_process:
        logging.warning(f"No PDF files found in '{Config.PDF_DIR}'. Nothing to process.")
    else:
        logging.info(f"Found {len(pdf_files_to_process)} PDF(s) to process.")
        docling_config_file = Config.ROOT_DIR / "docling_config.yml"

        for pdf_file in pdf_files_to_process:
            process_document_with_docling(
                pdf_path=pdf_file,
                output_dir=Config.DOCLING_OUTPUT_DIR,
                config_path=docling_config_file
            )

    logging.info("\n======================================================")
    logging.info("    Docling Harvester Phase Completed!     ")
    logging.info("======================================================")