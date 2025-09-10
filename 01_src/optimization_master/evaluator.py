# src/optimization_master/evaluator.py
# --- V8 (FINAL CORRECTED VERSION): Aligned with the new direct search pipeline ---

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import config
from src.query_master.rag_pipeline import QueryMaster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class RagEvaluator:
    def __init__(self):
        logging.info("Initializing the RAG Evaluator...")
        try:
            self.query_master = QueryMaster()
            logging.info("QueryMaster (the RAG system under test) has been loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"FATAL: Could not instantiate QueryMaster. Error: {e}")

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
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

            # --- KEY CHANGE: Use the direct search method, same as the new pipeline ---
            # We no longer generate a hypothetical answer. We search with the question directly.
            retrieved_docs = self.query_master._search_vector_store(question)
            
            is_hit = any(ground_truth_chunk in doc['text'] for doc in retrieved_docs)

            if is_hit:
                retrieval_hits += 1
                logging.info("  [✔️] SUCCESS: The ground truth source chunk was found in the retrieved documents.")
            else:
                logging.info("  [❌] FAILURE: The ground truth source chunk was NOT found.")

            generated_answer = self.query_master.answer_question(question)

            evaluation_details.append({
                "question": question,
                "is_hit": is_hit,
                "retrieved_sources": [doc['metadata'].get('source', 'N/A') for doc in retrieved_docs],
                "generated_answer": generated_answer,
                "ground_truth_answer": item.get("ground_truth_answer")
            })

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

def main():
    logging.info("=========================================================")
    logging.info("      Project Danesh: Starting RAG Pipeline Evaluator (V8)      ")
    logging.info("=========================================================")

    dataset_path = config.DRIVE_BASE_PATH / "optimization_data" / "evaluation_dataset.json"
    if not dataset_path.exists():
        sys.exit(f"FATAL: Evaluation dataset not found at '{dataset_path}'. Please run the 'generate' task first.")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            evaluation_dataset = json.load(f)
        logging.info(f"Successfully loaded {len(evaluation_dataset)} items from the evaluation dataset.")
    except Exception as e:
        sys.exit(f"Failed to load or parse the dataset file. Error: {e}")

    try:
        evaluator = RagEvaluator()
        evaluation_results = evaluator.evaluate(evaluation_dataset)
    except Exception as e:
        sys.exit(f"An error occurred during the evaluation process: {e}")
        
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
