# main_enhancer.py
# Main executable script for the Knowledge Master phase of Project Danesh.

import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the main function from our knowledge enhancer module
from src.knowledge_enhancer.knowledge_enhancer import main as run_knowledge_enhancement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to orchestrate the knowledge enhancement pipeline."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Phase 05 - Knowledge Master    ")
    logging.info("=========================================================")

    # Check for the Google API Key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
        sys.exit(1)
    logging.info("Google API key found.")

    # Run the knowledge enhancement script
    try:
        logging.info("\n--- Kicking off the Knowledge Enhancement process ---")
        run_knowledge_enhancement()
        logging.info("--- Knowledge Enhancement process finished ---")
    except Exception as e:
        logging.error(f"An unexpected error occurred during knowledge enhancement: {e}", exc_info=True)
        sys.exit(1)

    logging.info("=========================================================")
    logging.info("    Project Danesh: Knowledge Master Phase 05 tasks completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
