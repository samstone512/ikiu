import logging
from pathlib import Path

# --- Final, Working Imports ---
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

from config import Config

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def process_document_with_docling(pdf_path: Path, output_dir: Path):
    """
    Processes a single PDF using the DocumentConverter.
    """
    if not pdf_path.exists():
        logging.error(f"Input PDF not found at: {pdf_path}")
        return

    output_json_path = output_dir / f"{pdf_path.stem}.json"
    if output_json_path.exists():
        logging.warning(f"Output file {output_json_path.name} already exists. Skipping.")
        return

    logging.info(f"Starting to process document: {pdf_path.name}")

    try:
        # --- Correct Configuration ---
        pipeline_options = PdfPipelineOptions()
        pipeline_options.ocr_options.lang = ["fa"]
        
        # --- Correct Initialization ---
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # --- Run the conversion pipeline ---
        logging.info("Pipeline is running... This may take some time.")
        result = doc_converter.convert(source=str(pdf_path))
        processed_document = result.document
        
        # --- Serialize and Save the result ---
        logging.info("Serializing processed document to JSON...")
        json_output = processed_document.model_dump_json(indent=2)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        logging.info(f"✅ Successfully processed and saved output to: {output_json_path.name}")

    except Exception as e:
        logging.error(f"❌ An error occurred while processing {pdf_path.name}: {e}", exc_info=True)


if __name__ == "__main__":
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Docling Harvester (Single File Mode)    ")
    logging.info("======================================================")
    
    # --- MODIFICATION ---
    # We are now targeting a single, specific file for focused analysis.
    target_pdf_name = "shivename.pdf"
    
    # Construct the full path to the target file
    pdf_to_process = Config.PDF_DIR / target_pdf_name

    if not pdf_to_process.exists():
        logging.error(f"Target PDF file not found: '{pdf_to_process}'")
        logging.error("Please ensure the file exists in the 'data/pdf' directory.")
    else:
        logging.info(f"Found target file: {target_pdf_name}")
        
        # Ensure the output directory exists
        Config.DOCLING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Process the single specified file
        process_document_with_docling(
            pdf_path=pdf_to_process,
            output_dir=Config.DOCLING_OUTPUT_DIR
        )

    logging.info("\n======================================================")
    logging.info("    Docling Harvester Phase Completed!     ")
    logging.info("======================================================")