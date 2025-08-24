# main_processing_pipeline.py (نسخه مقاوم در برابر خطا)

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
    The main function to run the robust document processing pipeline.
    It processes each PDF individually and saves a separate JSON for each.
    """
    print("--- Starting Phase 1: Intelligent Document Processing (Robust Version) ---")
    
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
    
    successful_files = 0
    failed_files = 0

    # Process each PDF file individually
    for pdf_file in tqdm(pdf_files, desc="Processing all documents"):
        
        # Define paths for this specific file
        input_path = os.path.join(config.DATA_PATH, pdf_file)
        # Create a clean filename for the output JSON
        output_filename = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join(config.PROCESSED_DATA_PATH, output_filename)
        
        print(f"\n--- Processing: {pdf_file} ---")
        
        try:
            # Step 1: Extract raw text
            raw_text = extract_text_from_pdf(input_path)
            
            # Step 2: Clean the extracted text
            cleaned_text = clean_extracted_text(raw_text)
            
            # Step 3: Perform intelligent chunking
            structured_chunks = intelligent_chunking(cleaned_text)
            
            # Save the successful result to its own JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_chunks, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Successfully processed and saved to {output_filename}")
            successful_files += 1

        except Exception as e:
            # If any step fails, log the error and save it to a JSON file
            print(f"❌ An error occurred while processing {pdf_file}: {e}")
            error_data = {"error": str(e), "file": pdf_file}
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=4)
            
            print(f"Error details saved to {output_filename}")
            failed_files += 1

    print("\n--- Pipeline Execution Summary ---")
    print(f"Total files processed: {len(pdf_files)}")
    print(f"✅ Successful: {successful_files}")
    print(f"❌ Failed: {failed_files}")
    print("--- Phase 1: Intelligent Document Processing Completed! ---")


if __name__ == "__main__":
    main()
