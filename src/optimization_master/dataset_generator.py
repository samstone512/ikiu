# src/optimization_master/dataset_generator.py
# This script automatically generates a question-answer evaluation dataset
# from the clean text chunks processed in the Data Harvester phase.

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Add the project root to the Python path ---
# This allows us to import modules from other directories, like 'config'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import config
from src.knowledge_weaver.json_loader import load_processed_texts
from src.knowledge_weaver.text_splitter import split_text

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- 2. DEFINE THE PROMPT FOR DATASET GENERATION ---
DATASET_GENERATION_PROMPT = """
You are an expert AI assistant tasked with creating a high-quality evaluation dataset for a Persian RAG (Retrieval-Augmented Generation) system.
Based on the "Text Chunk" provided below, generate a precise question-answer pair in JSON format.

**INSTRUCTIONS:**
1.  **Read the Text Chunk Carefully:** Understand the main topic and the specific details within the text.
2.  **Generate a Relevant Question:** The question MUST be answerable *only* from the provided text chunk. The question should be something a real user (student, professor) might ask.
3.  **Generate a Ground Truth Answer:** The answer must be a direct, concise, and accurate summary of the information present in the text chunk that addresses the question.
4.  **Include the Source:** The source of the answer is the text chunk itself.
5.  **Format as JSON:** The final output must be a single, valid JSON object with the keys "question", "ground_truth_answer", and "source_chunk".

**Text Chunk:**
---
{text_chunk}
---

**JSON Output:**
"""

# --- 3. CONFIGURE THE GENERATIVE MODEL ---
def get_generative_model() -> Optional[genai.GenerativeModel]:
    """Configures and returns the Gemini generative model."""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
            return None
        genai.configure(api_key=api_key)
        
        # We will use JSON Mode to ensure the output is always a valid JSON object
        generation_config = {
            "response_mime_type": "application/json",
        }
        
        # Safety settings to minimize blocking
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            config.GEMINI_TEXT_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info("Successfully configured Google Gemini API for dataset generation.")
        return model
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        return None

# --- 4. MAIN GENERATION LOGIC ---
def generate_qa_pairs(
    chunks: List[str],
    model: genai.GenerativeModel
) -> List[Dict[str, Any]]:
    """
    Iterates through text chunks and generates a question-answer pair for each one.
    """
    qa_dataset = []
    total_chunks = len(chunks)
    logging.info(f"Starting QA pair generation for {total_chunks} text chunks.")

    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{total_chunks}...")
        if not chunk.strip():
            logging.warning(f"  - Chunk {i + 1} is empty. Skipping.")
            continue

        prompt = DATASET_GENERATION_PROMPT.format(text_chunk=chunk)
        
        try:
            # Generate content using the model
            response = model.generate_content(prompt)
            
            # The response.text is a guaranteed valid JSON string in JSON Mode
            qa_pair = json.loads(response.text)
            
            # Add the source chunk for reference
            qa_pair["source_chunk"] = chunk
            
            qa_dataset.append(qa_pair)
            logging.info(f"  - Successfully generated QA pair for chunk {i + 1}.")
            
            # Respect API rate limits
            time.sleep(2)

        except Exception as e:
            logging.error(f"  - Failed to generate QA pair for chunk {i + 1}. Error: {e}")
            # Optional: Add a retry mechanism here if needed

    return qa_dataset

# --- 5. SCRIPT EXECUTION ---
def main():
    """Main function to run the dataset generation pipeline."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Evaluation Dataset Generator    ")
    logging.info("=========================================================")

    model = get_generative_model()
    if not model:
        sys.exit(1)

    # 1. Load all processed documents from Phase 01
    documents = load_processed_texts(config.PROCESSED_TEXT_DIR)
    if not documents:
        logging.warning("No processed text documents found. Exiting.")
        sys.exit(0)

    # 2. Split all documents into chunks using the same logic as the Knowledge Weaver
    all_chunks = []
    for doc in documents:
        # We use the 'full_text' which is the cleaned text
        cleaned_text = doc.get('full_text', '')
        if cleaned_text:
            chunks = split_text(cleaned_text)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        logging.error("No text chunks could be created from the documents. Cannot generate dataset.")
        sys.exit(1)

    # 3. Generate the question-answer pairs
    evaluation_dataset = generate_qa_pairs(all_chunks, model)

    # 4. Save the dataset to a file
    if evaluation_dataset:
        output_path = project_root / "evaluation_dataset.json"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_dataset, f, ensure_ascii=False, indent=4)
            logging.info(f"Successfully saved {len(evaluation_dataset)} QA pairs to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save the evaluation dataset. Error: {e}")
    else:
        logging.warning("No QA pairs were generated. The output file will not be created.")

    logging.info("=========================================================")
    logging.info("    Project Danesh: Evaluation Dataset Generation Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
