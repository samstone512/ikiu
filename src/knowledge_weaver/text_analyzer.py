# src/knowledge_weaver/text_analyzer.py
# Module for analyzing text to extract entities and relationships using an LLM.

import logging
import json
import time
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
    relationships based on a provided prompt template.

    Args:
        text_chunk (str): The piece of text to analyze.
        model (genai.GenerativeModel): The initialized Gemini model instance.
        prompt_template (str): The prompt template string, which must contain
                               a '{text_chunk}' placeholder.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing 'entities' and
                                  'relationships' if successful, otherwise None.
    """
    if not text_chunk.strip():
        logging.warning("Input text is empty. Skipping analysis.")
        return None

    try:
        # Format the prompt with the actual text chunk.
        prompt = prompt_template.format(text_chunk=text_chunk)

        logging.info("Sending text chunk to Gemini for entity extraction...")
        response = model.generate_content(prompt)

        # IMPORTANT: Add a delay to respect API rate limits.
        time.sleep(2)

        # The model's response might be enclosed in markdown backticks for JSON.
        # We need to clean it before parsing.
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

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
        logging.error("Failed to decode JSON from the model's response.")
        logging.debug(f"Model raw response was: {response.text}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during text analysis: {e}")
        return None
