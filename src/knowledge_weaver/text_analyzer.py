# src/knowledge_weaver/text_analyzer.py
# Module for analyzing text to extract entities and relationships using an LLM.

import logging
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional

import google.generativeai as genai

# --- ADDED: Configuration for retry mechanism ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

def load_prompt_template(prompt_path: Path) -> str:
    """
    Reads the content of a prompt template file.
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"FATAL: Prompt file not found at {prompt_path}. Please ensure it exists.")
        raise
    except Exception as e:
        logging.error(f"FATAL: Could not read prompt file at {prompt_path}. Error: {e}")
        raise

def analyze_text_for_entities(
    text_chunk: str,
    model: genai.GenerativeModel,
    prompt_template: str
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a chunk of text using a Gemini model to extract entities and
    relationships. This version includes a retry mechanism for API calls.
    """
    if not text_chunk.strip():
        logging.warning("Input text is empty. Skipping analysis.")
        return None

    # --- ADDED: Retry Loop ---
    for attempt in range(MAX_RETRIES):
        try:
            # Format the prompt with the actual text chunk.
            prompt = prompt_template.format(text_chunk=text_chunk)

            logging.info(f"Sending text chunk to Gemini for entity extraction (Attempt {attempt + 1}/{MAX_RETRIES})...")

            safety_settings = {
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }

            response = model.generate_content(
                prompt,
                safety_settings=safety_settings
            )
            
            # --- Robust JSON Cleaning ---
            raw_text = response.text
            json_match = re.search(r'```(json)?\s*({.*?})\s*```', raw_text, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(2)
            else:
                start_index = raw_text.find('{')
                end_index = raw_text.rfind('}')
                if start_index != -1 and end_index != -1:
                    cleaned_response = raw_text[start_index:end_index+1]
                else:
                    # This triggers a retry if no JSON is found
                    raise ValueError("No valid JSON object found in the model's response.")

            structured_data = json.loads(cleaned_response)

            if 'entities' in structured_data and 'relationships' in structured_data:
                logging.info("Successfully extracted structured data from text chunk.")
                return structured_data # Success! Exit the loop and function.
            else:
                # This also triggers a retry if JSON is valid but keys are missing
                raise ValueError("Parsed JSON is missing 'entities' or 'relationships' keys.")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            logging.warning(f"Attempt {attempt + 1} failed. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error("All retry attempts failed for this text chunk.")
                logging.debug(f"Final failed raw response was: {response.text if 'response' in locals() else 'No response'}")
                return None # Return None after all retries have failed

    return None # Should not be reached, but as a fallback
