# main_query_master.py
# Main executable script for the Query Master phase of Project Danesh.

import os
import sys
import logging

import google.generativeai as genai

# Import the function that creates our UI
from src.query_master.ui import create_chatbot_interface

# --- 1. SETUP LOGGING ---
# Configure logging to provide clear, step-by-step status updates.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to configure the API and launch the chatbot UI."""
    logging.info("======================================================")
    logging.info("    Project Danesh: Starting Phase 03 - Query Master    ")
    logging.info("======================================================")

    # --- 2. CONFIGURE GEMINI API ---
    # Securely read the API key from an environment variable.
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set. Please set it in your Colab notebook.")
            sys.exit(1)
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Google Gemini API.")
    except Exception as e:
        logging.error(f"FATAL: Failed to configure Gemini API. Error: {e}")
        sys.exit(1)

    # --- 3. LAUNCH THE CHATBOT INTERFACE ---
    # This function is imported from our ui.py module and will handle
    # everything related to the Gradio interface.
    create_chatbot_interface()

    logging.info("======================================================")
    logging.info("    Project Danesh: Query Master has been shut down.    ")
    logging.info("======================================================")

if __name__ == '__main__':
    main()
