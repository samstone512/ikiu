import logging
from pathlib import Path

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
    Processes a single PDF, gets the result as a Document object,
    and then saves it to a JSON file.
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
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.ocr_options.lang = ["fa"]  # Set language to Farsi/Persian
        
        # Initialize document converter with PDF format options
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # Run the conversion pipeline
        logging.info("Pipeline is running... This may take some time depending on the document size.")
        result = doc_converter.convert(source=str(pdf_path))
        processed_document = result.document
        
        # Serialize the result to a human-readable JSON string
        logging.info("Serializing processed document to JSON...")
        json_output = processed_document.model_dump_json(indent=2)
        
        # Write the JSON string to our target file, ensuring UTF-8 encoding for Farsi text
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        logging.info(f"✅ Successfully processed and saved output to: {output_json_path.name}")

    except Exception as e:
        logging.error(f"❌ An error occurred while processing {pdf_path.name}: {e}", exc_info=True)


if __name__ == "__main__":
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Docling Harvester    ")
    logging.info("======================================================")
    
    Config.DOCLING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files_to_process = list(Config.PDF_DIR.glob("*.pdf"))
    
    if not pdf_files_to_process:
        logging.warning(f"No PDF files found in '{Config.PDF_DIR}'. Nothing to process.")
    else:
        logging.info(f"Found {len(pdf_files_to_process)} PDF(s) to process.")

        for pdf_file in pdf_files_to_process:
            process_document_with_docling(
                pdf_path=pdf_file,
                output_dir=Config.DOCLING_OUTPUT_DIR
            )


    logging.info("\n======================================================")
    logging.info("    Docling Harvester Phase Completed!     ")
    logging.info("======================================================")