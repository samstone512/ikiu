# run_harvester_v2.py

import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import pypdfium2 as pdfium

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
INPUT_PDF_DIR = os.getenv("INPUT_PDF_DIR", "data/raw")
OUTPUT_TEXT_DIR = os.getenv("OUTPUT_TEXT_DIR", "data/processed")
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa" # A powerful model for document understanding

# --- Main Harvester Logic ---

def setup_model():
    """Initializes and returns the Donut model and processor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        processor = DonutProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
        return processor, model, device
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def convert_pdf_to_images(pdf_path):
    """Converts a PDF file into a list of PIL Images."""
    try:
        images = pdfium.render_pdf_to_pil(pdf_path)
        return list(images)
    except Exception as e:
        logging.error(f"Could not convert PDF {pdf_path} to images. Error: {e}")
        return []

def process_image(image, processor, model, device):
    """Processes a single image using the Donut model and returns the extracted text."""
    # Prepare image for the model
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # Generate transcription
    task_prompt = "<s_docvqa>" # Document Visual Question Answering prompt
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode the generated ids to text
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # The model may return the prompt, so we remove it.
    processed_text = sequence.split(task_prompt)[-1].strip()
    return processed_text


def run_harvester(pdf_dir, output_dir):
    """
    Main function to orchestrate the data harvesting process.
    It finds PDFs, converts them to images, and uses a VDU model to extract text.
    """
    logging.info("--- Starting Phase 6: Stability Master Harvester ---")
    os.makedirs(output_dir, exist_ok=True)

    processor, model, device = setup_model()
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"No PDF files found in {pdf_dir}. Exiting.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        logging.info(f"Processing document: {pdf_path}")
        
        # 1. Convert PDF to a list of images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            continue

        full_document_text = []
        # 2. Process each image (page)
        for i, page_image in enumerate(tqdm(images, desc=f"Pages of {pdf_file}", leave=False)):
            try:
                page_text = process_image(page_image, processor, model, device)
                full_document_text.append(f"--- PAGE {i+1} ---\n{page_text}")
            except Exception as e:
                logging.error(f"Failed to process page {i+1} of {pdf_file}. Error: {e}")
                full_document_text.append(f"--- PAGE {i+1} FAILED TO PROCESS ---")
        
        # 3. Save the combined text to a file
        output_filename = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_document_text))
            
        logging.info(f"Successfully processed and saved output to {output_path}")

    logging.info("--- Harvester run completed. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the new Data Harvester using a Visual Document Understanding model.")
    parser.add_argument("--input_dir", type=str, default=INPUT_PDF_DIR, help="Directory containing raw PDF files.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_TEXT_DIR, help="Directory to save the processed text files.")
    args = parser.parse_args()

    run_harvester(pdf_dir=args.input_dir, output_dir=args.output_dir)
