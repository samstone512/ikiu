# src/query_master/rag_pipeline.py
# --- PHASE 5 UPGRADE: Using a direct QA-based retrieval strategy ---

import logging
import networkx as nx
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any
import re

import config

class QueryMaster:
    def __init__(self):
        logging.info("Initializing QueryMaster (Phase 5 - QA Model)...")
        self.config = config
        self.text_model = genai.GenerativeModel(self.config.GEMINI_GENERATION_MODEL_NAME)
        self.embedding_model_name = self.config.GEMINI_EMBEDDING_MODEL_NAME
        self._load_vector_store()
        self._load_prompt_template()
        # Graph is temporarily disabled to focus on retrieval accuracy
        # self._load_knowledge_graph() 
        logging.info("QueryMaster initialization complete.")

    def _load_vector_store(self):
        try:
            client = chromadb.PersistentClient(path=str(self.config.VECTOR_DB_DIR))
            self.collection = client.get_collection(name=self.config.CHROMA_COLLECTION_NAME)
            logging.info(f"Successfully connected to QA ChromaDB collection: '{self.config.CHROMA_COLLECTION_NAME}'.")
        except Exception as e:
            raise RuntimeError(f"FATAL: Failed to connect to ChromaDB. Error: {e}")

    def _load_prompt_template(self):
        try:
            with open(self.config.RAG_PROMPT_PATH, 'r', encoding='utf-8') as f:
                self.rag_prompt_template = f.read()
            logging.info("Successfully loaded RAG prompt template.")
        except Exception as e:
            raise RuntimeError(f"FATAL: Could not read RAG prompt file. Error: {e}")

    def _search_vector_store(self, user_question: str) -> List[Dict[str, Any]]:
        """
        Searches the vector store by embedding the user's question and finding
        the most similar questions in our knowledge base.
        """
        logging.info(f"Performing QA-based vector search for: '{user_question}'")
        try:
            query_embedding = genai.embed_content(
                model=self.embedding_model_name,
                content=user_question,
                task_type="RETRIEVAL_QUERY"
            )['embedding']
            
            # Find the most similar QUESTIONS in the database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.VECTOR_SEARCH_TOP_K 
            )
            
            # The documents returned are the ANSWERS associated with those questions
            retrieved_docs = []
            for i, doc_text in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "text": doc_text, # This is the original text chunk (the answer)
                    "metadata": results['metadatas'][0][i]
                })
            logging.info(f"Found {len(retrieved_docs)} relevant answers by matching questions.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error during QA vector search: {e}")
            return []

    def answer_question(self, user_question: str) -> str:
        """
        The main method to answer a user's question using the new QA retrieval architecture.
        """
        # Step 1: Search the QA vector store directly with the user's question.
        # No re-ranking is needed initially as the retrieval should be very accurate.
        relevant_docs = self._search_vector_store(user_question)
        
        if not relevant_docs:
            return "متاسفانه اطلاعات مرتبطی برای پاسخ به سوال شما در منابع موجود یافت نشد."

        # We take the top N results directly
        top_docs = relevant_docs[:self.config.RERANK_TOP_N]

        vector_context_str = "".join(f"Source: {doc['metadata']['source']}\nContent: {doc['text']}\n\n" for doc in top_docs)
        
        final_prompt = self.rag_prompt_template.format(
            vector_context=vector_context_str,
            graph_context="Graph context is not used in this version.", # Graph is disabled
            user_question=user_question
        )
        
        logging.info("Generating final answer with Gemini...")
        try:
            response = self.text_model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error during final answer generation: {e}")
            return "خطایی در هنگام تولید پاسخ رخ داد. لطفاً دوباره تلاش کنید."
