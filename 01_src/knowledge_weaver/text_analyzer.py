# src/knowledge_weaver/text_analyzer.py
# Module for analyzing text to extract entities and relationships using an LLM.

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configuration for retry mechanism
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
    Analyzes a chunk of text using a Gemini model in JSON Mode to ensure
    a valid JSON output, with a retry mechanism for transient errors.
    """
    if not text_chunk.strip():
        logging.warning("Input text is empty. Skipping analysis.")
        return None

    # --- ADDED: Define the exact JSON schema the model MUST follow ---
    json_schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["id", "type"]
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["source", "target", "type"]
                }
            }
        },
        "required": ["entities", "relationships"]
    }

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": json_schema
    }

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for attempt in range(MAX_RETRIES):
        try:
            prompt = prompt_template.format(text_chunk=text_chunk)
            logging.info(f"Sending text chunk to Gemini for entity extraction (Attempt {attempt + 1}/{MAX_RETRIES})...")

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # In JSON mode, the response.text is a guaranteed valid JSON string
            structured_data = json.loads(response.text)

            logging.info("Successfully extracted structured data from text chunk using JSON Mode.")
            return structured_data

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed. Error: {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error("All retry attempts failed for this text chunk.")
                return None

    return None
