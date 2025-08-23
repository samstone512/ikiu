# main_processing_pipeline.py

import os
import json
from tqdm import tqdm

# Import configurations from our config file
import config

# Import the functions we built in our processor module
from src.document_processing.processor import (
    extract_text_from_pdf,
    clean_extracted_text,
    intelligent_chunking,
)

def main():
    """
    The main function to run the entire document processing pipeline.
    """
    print("--- Starting Phase 1: Intelligent Document Processing ---")
    
    # Ensure the output directory exists
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    print(f"Output will be saved to: {config.PROCESSED_DATA_PATH}")

    # Find all PDF files in the data directory
    try:
        pdf_files = [f for f in os.listdir(config.DATA_PATH) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"Warning: No PDF files found in {config.DATA_PATH}. Please add some PDFs to process.")
            return
    except FileNotFoundError:
        print(f"Error: The data directory was not found at {config.DATA_PATH}")
        print("Please ensure your Google Drive is mounted and the path in config.py is correct.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    
    # This dictionary will hold the processed data for all documents
    all_documents_data = {}

    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing all documents"):
        file_path = os.path.join(config.DATA_PATH, pdf_file)
        
        print(f"\n--- Processing: {pdf_file} ---")
        
        try:
            # Step 1: Extract raw text
            raw_text = extract_text_from_pdf(file_path)
            
            # Step 2: Clean the extracted text
            cleaned_text = clean_extracted_text(raw_text)
            
            # Step 3: Perform intelligent chunking
            structured_chunks = intelligent_chunking(cleaned_text)
            
            # Store the results for this document
            all_documents_data[pdf_file] = structured_chunks
            
            print(f"Successfully processed and chunked {pdf_file}.")

        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")
            # Optionally, store error information
            all_documents_data[pdf_file] = {"error": str(e)}

    # Define the path for the final JSON output
    output_json_path = os.path.join(config.PROCESSED_DATA_PATH, "processed_documents.json")
    
    # Save all the processed data into a single JSON file
    print(f"\nSaving all processed data to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents_data, f, ensure_ascii=False, indent=4)
        
    print("--- Phase 1: Intelligent Document Processing Completed Successfully! ---")


if __name__ == "__main__":
    main()
