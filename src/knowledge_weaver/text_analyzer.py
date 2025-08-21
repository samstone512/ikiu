# src/knowledge_weaver/text_analyzer.py
# Module for analyzing text to extract entities and relationships using an LLM.

import logging
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional

import google.generativeai as genai

def load_prompt_template(prompt_path: Path) -> str:
    """
    Reads the content of a prompt template file.

    Args:
        prompt_path (Path): The path to the text file containing the prompt.

    Returns:
        str: The content of the prompt file.
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
    relationships based on a provided prompt template. This version includes
    more robust JSON cleaning and safety settings.
    """
    if not text_chunk.strip():
        logging.warning("Input text is empty. Skipping analysis.")
        return None

    try:
        # Format the prompt with the actual text chunk.
        prompt = prompt_template.format(text_chunk=text_chunk)

        logging.info("Sending text chunk to Gemini for entity extraction...")

        # --- ADDED SAFETY SETTINGS ---
        # This prevents the model from blocking responses that might be
        # falsely flagged, which can result in an empty or malformed response.
        # This is crucial when expecting a strict JSON output.
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings # Apply the settings here
        )

        # IMPORTANT: Add a delay to respect API rate limits.
        time.sleep(2)

        # --- Robust JSON Cleaning ---
        # The model's response might be enclosed in markdown backticks or have extra text.
        # We find the first '{' and the last '}' to extract the JSON object.
        raw_text = response.text
        
        # Use regex to find the JSON block, even with markdown
        json_match = re.search(r'```(json)?\s*({.*?})\s*```', raw_text, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(2)
        else:
            # Fallback for cases where markdown is not used
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                cleaned_response = raw_text[start_index:end_index+1]
            else:
                logging.error("No valid JSON object found in the model's response.")
                logging.debug(f"Model raw response was: {raw_text}")
                return None

        # Attempt to parse the cleaned string as JSON.
        structured_data = json.loads(cleaned_response)

        # Basic validation to ensure the expected keys are present.
        if 'entities' in structured_data and 'relationships' in structured_data:
            logging.info("Successfully extracted structured data from text chunk.")
            return structured_data
        else:
            logging.warning("Parsed JSON is missing 'entities' or 'relationships' keys.")
            return None

    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the model's response even after cleaning.")
        logging.debug(f"Model raw response was: {response.text}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during text analysis: {e}")
        return None
