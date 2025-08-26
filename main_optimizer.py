# main_optimizer.py
# Main executable script for the Optimization Master phase of Project Danesh.
# This script will orchestrate dataset generation, evaluation, and fine-tuning.

import os
import sys
import logging

# --- Add the project root to the Python path ---
# This ensures that we can import modules from our 'src' directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the main function from our dataset generator module
from src.optimization_master.dataset_generator import main as run_dataset_generation

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def main():
    """Main function to orchestrate the entire optimization pipeline."""
    logging.info("=========================================================")
    logging.info("    Project Danesh: Starting Phase 04 - Optimization Master    ")
    logging.info("=========================================================")

    # --- 2. CHECK FOR GOOGLE_API_KEY ---
    # The dataset generator requires the API key to be set.
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
        logging.error("Please set it in your Colab environment before running.")
        sys.exit(1)
    logging.info("Google API key found.")

    # --- 3. RUN THE DATASET GENERATION SCRIPT ---
    # In this phase, our first task is to generate the evaluation dataset.
    # We simply call the main function we imported from the generator script.
    try:
        logging.info("\n--- Kicking off the Evaluation Dataset Generation process ---")
        run_dataset_generation()
        logging.info("--- Evaluation Dataset Generation process finished ---")
    except Exception as e:
        logging.error(f"An unexpected error occurred during dataset generation: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Future steps for this phase will be added here ---
    # For example: run_evaluation(), run_fine_tuning(), etc.

    logging.info("=========================================================")
    logging.info("    Project Danesh: Optimization Master Phase 04 tasks completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
