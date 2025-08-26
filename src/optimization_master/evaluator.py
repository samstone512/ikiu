# src/optimization_master/evaluator.py
# This script automates the evaluation of the RAG pipeline using the generated dataset.

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# --- Add the project root to the Python path ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import config
from src.query_master.rag_pipeline import QueryMaster

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- 2. EVALUATION LOGIC ---
class RagEvaluator:
    def __init__(self):
        """Initializes the evaluator by loading the RAG pipeline."""
        logging.info("Initializing the RAG Evaluator...")
        try:
            # We instantiate QueryMaster which loads all necessary models and data.
            # This is the system we are going to test.
            self.query_master = QueryMaster()
            logging.info("QueryMaster (the RAG system under test) has been loaded successfully.")
        except Exception as e:
            logging.error(f"FATAL: Could not instantiate QueryMaster. Evaluation cannot proceed. Error: {e}", exc_info=True)
            raise

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs the evaluation pipeline on the provided dataset.

        Args:
            dataset: A list of question-answer pairs from our evaluation dataset.

        Returns:
            A dictionary containing the evaluation results and detailed logs.
        """
        total_questions = len(dataset)
        logging.info(f"Starting evaluation for {total_questions} questions...")

        retrieval_hits = 0
        evaluation_details = []

        for i, item in enumerate(dataset):
            question = item.get("question")
            ground_truth_chunk = item.get("source_chunk")
            
            logging.info(f"--- Processing Question {i + 1}/{total_questions}: '{question}' ---")

            if not question or not ground_truth_chunk:
                logging.warning("Skipping item due to missing 'question' or 'source_chunk'.")
                continue

            # --- This is the core of the evaluation ---
            # We will "peek" inside the RAG pipeline to check the retrieved documents
            # before the final answer is generated.

            # 1. Generate hypothetical answer (HyDE)
            hypothetical_answer = self.query_master._generate_hypothetical_answer(question)
            
            # 2. Search the vector store to get the initial retrieved documents
            retrieved_docs = self.query_master._search_vector_store(hypothetical_answer)
            
            # 3. Check for a "hit"
            is_hit = any(ground_truth_chunk in doc['text'] for doc in retrieved_docs)

            if is_hit:
                retrieval_hits += 1
                logging.info("  [✔️] SUCCESS: The ground truth source chunk was found in the retrieved documents.")
            else:
                logging.info("  [❌] FAILURE: The ground truth source chunk was NOT found.")

            # Get the final answer from the chatbot for logging purposes
            generated_answer = self.query_master.answer_question(question)

            evaluation_details.append({
                "question": question,
                "is_hit": is_hit,
                "retrieved_sources": [doc['metadata'].get('source', 'N/A') for doc in retrieved_docs],
                "generated_answer": generated_answer,
                "ground_truth_answer": item.get("ground_truth_answer")
            })

        # Calculate final metrics
        retrieval_accuracy = (retrieval_hits / total_questions) * 100 if total_questions > 0 else 0

        results = {
            "total_questions": total_questions,
            "retrieval_hits": retrieval_hits,
            "retrieval_accuracy": f"{retrieval_accuracy:.2f}%",
            "details": evaluation_details
        }
        
        logging.info("--- Evaluation Complete ---")
        logging.info(f"Retrieval Accuracy: {results['retrieval_accuracy']}")
        
        return results

# --- 3. SCRIPT EXECUTION ---
def main():
    """Main function to load the dataset and run the evaluation."""
    logging.info("=========================================================")
    logging.info("      Project Danesh: Starting RAG Pipeline Evaluator      ")
    logging.info("=========================================================")

    # 1. Load the generated evaluation dataset from Google Drive
    dataset_path = config.DRIVE_BASE_PATH / "optimization_data" / "evaluation_dataset.json"
    if not dataset_path.exists():
        logging.error(f"FATAL: Evaluation dataset not found at '{dataset_path}'.")
        logging.error("Please run the 'dataset_generator' script first.")
        sys.exit(1)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            evaluation_dataset = json.load(f)
        logging.info(f"Successfully loaded {len(evaluation_dataset)} items from the evaluation dataset.")
    except Exception as e:
        logging.error(f"Failed to load or parse the dataset file. Error: {e}")
        sys.exit(1)

    # 2. Initialize and run the evaluator
    try:
        evaluator = RagEvaluator()
        evaluation_results = evaluator.evaluate(evaluation_dataset)
    except Exception as e:
        logging.error(f"An error occurred during the evaluation process: {e}", exc_info=True)
        sys.exit(1)
        
    # 3. Save the detailed results to a file in Google Drive
    output_path = config.DRIVE_BASE_PATH / "optimization_data" / "evaluation_results.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ Successfully saved detailed evaluation results to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results. Error: {e}")

    logging.info("=========================================================")
    logging.info("      Project Danesh: RAG Pipeline Evaluation Completed!     ")
    logging.info("=========================================================")

if __name__ == '__main__':
    main()
