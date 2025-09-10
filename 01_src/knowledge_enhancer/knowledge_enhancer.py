# src/knowledge_enhancer/knowledge_enhancer.py
# This script enriches the knowledge base by generating a question for each text chunk.

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import config
from src.knowledge_weaver.json_loader import load_processed_texts
from src.knowledge_weaver.text_splitter import split_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Prompt for generating a single, high-quality question from a text chunk
QUESTION_GENERATION_PROMPT = """
You are an expert AI assistant for creating training data for a question-answering system.
Based on the provided "Text Chunk" from a university regulation document, your task is to generate a single, clear, and relevant question that this chunk directly answers.

**INSTRUCTIONS:**
1.  **Analyze the Chunk:** Read the text chunk carefully to understand the specific rule, definition, or piece of information it contains.
2.  **Formulate a Question:** Create a question that a student, professor, or university staff member would realistically ask. The answer to the question must be found entirely within the provided text chunk.
3.  **Output ONLY the Question:** Your entire output should be just the question text itself, without any prefixes, explanations, or JSON formatting.

**Text Chunk:**
---
{text_chunk}
---

**Question:**
"""

def get_generative_model() -> Optional[genai.GenerativeModel]:
    """Configures and returns the Gemini generative model for text generation."""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
            return None
        genai.configure(api_key=api_key)
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(config.GEMINI_TEXT_MODEL_NAME, safety_settings=safety_settings)
        logging.info("Successfully configured Google Gemini API for question generation.")
        return model
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        return None

def generate_question_answer_pairs(chunks: List[str], model: genai.GenerativeModel) -> List[Dict[str, str]]:
    """
    Iterates through text chunks and generates a corresponding question for each.
    """
    qa_pairs = []
    total_chunks = len(chunks)
    logging.info(f"Starting question generation for {total_chunks} text chunks.")

    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{total_chunks}...")
        if not chunk.strip():
            logging.warning(f"  - Chunk {i + 1} is empty. Skipping.")
            continue

        prompt = QUESTION_GENERATION_PROMPT.format(text_chunk=chunk)
        
        try:
            response = model.generate_content(prompt)
            question = response.text.strip()
            
            if not question:
                logging.warning(f"  - Model returned an empty question for chunk {i + 1}. Skipping.")
                continue

            qa_pairs.append({
                "question": question,
                "answer": chunk  # The original chunk is the answer
            })
            logging.info(f"  - Successfully generated question for chunk {i + 1}.")
            time.sleep(1) # Respect API rate limits

        except Exception as e:
            logging.error(f"  - Failed to generate question for chunk {i + 1}. Error: {e}")

    return qa_pairs

def main():
    """Main function to run the knowledge enhancement pipeline."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Phase 05 - Knowledge Enhancer    ")
    logging.info("=========================================================")

    model = get_generative_model()
    if not model:
        sys.exit(1)

    # 1. Load the clean, processed documents
    documents = load_processed_texts(config.PROCESSED_TEXT_DIR)
    if not documents:
        logging.warning("No processed text documents found. Exiting.")
        sys.exit(0)

    # 2. Split all documents into chunks
    all_chunks = []
    for doc in documents:
        cleaned_text = doc.get('full_text', '')
        if cleaned_text:
            chunks = split_text(cleaned_text)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        logging.error("No text chunks could be created from the documents. Cannot generate QA pairs.")
        sys.exit(1)

    # 3. Generate the question-answer pairs
    knowledge_pairs = generate_question_answer_pairs(all_chunks, model)

    # 4. Save the enriched knowledge to a new file in Google Drive
    if knowledge_pairs:
        output_dir = config.DRIVE_BASE_PATH / "knowledge_base"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "enriched_knowledge.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_pairs, f, ensure_ascii=False, indent=4)
            logging.info(f"✅ Successfully saved {len(knowledge_pairs)} enriched QA pairs to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save the enriched knowledge file. Error: {e}")
    else:
        logging.warning("No QA pairs were generated. The output file will not be created.")

    logging.info("=========================================================")
    logging.info("    Project Danesh: Knowledge Enhancement Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
