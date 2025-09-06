# main_optimizer.py
# Main executable script for the Optimization Master phase of Project Danesh.
# This script now orchestrates both dataset generation and pipeline evaluation.

import os
import sys
import logging
import argparse # Added to allow choosing which step to run
from dotenv import load_dotenv
load_dotenv()
# --- Add the project root to the Python path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Import the main functions from our optimization modules ---
from src.optimization_master.dataset_generator import main as run_dataset_generation
from src.optimization_master.evaluator import main as run_evaluation

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

    # --- NEW: Add command-line arguments to select the task ---
    parser = argparse.ArgumentParser(description="Run the Optimization Master pipeline for Project Danesh.")
    parser.add_argument(
        'task',
        choices=['generate', 'evaluate', 'full'],
        help="The task to perform: 'generate' dataset, 'evaluate' the pipeline, or run the 'full' pipeline."
    )
    args = parser.parse_args()

    # --- 2. CHECK FOR GOOGLE_API_KEY ---
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logging.error("FATAL: 'GOOGLE_API_KEY' environment variable not set.")
        sys.exit(1)
    logging.info("Google API key found.")

    # --- 3. RUN THE SELECTED TASK ---
    if args.task == 'generate' or args.task == 'full':
        try:
            logging.info("\n--- Kicking off the Evaluation Dataset Generation process ---")
            run_dataset_generation()
            logging.info("--- Evaluation Dataset Generation process finished ---\n")
        except Exception as e:
            logging.error(f"An unexpected error occurred during dataset generation: {e}", exc_info=True)
            sys.exit(1)

    if args.task == 'evaluate' or args.task == 'full':
        try:
            logging.info("\n--- Kicking off the RAG Pipeline Evaluation process ---")
            run_evaluation()
            logging.info("--- RAG Pipeline Evaluation process finished ---")
        except Exception as e:
            logging.error(f"An unexpected error occurred during evaluation: {e}", exc_info=True)
            sys.exit(1)

    logging.info("=========================================================")
    logging.info("    Project Danesh: Optimization Master Phase 04 tasks completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
