# src/query_master/rag_pipeline.py
# --- FINAL OPTIMIZATION: Re-enabling Re-ranking on top of the QA retrieval ---

import logging
import networkx as nx
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any
import re

import config

class QueryMaster:
    def __init__(self):
        logging.info("Initializing QueryMaster (Phase 5 - QA Model with Re-ranking)...")
        self.config = config
        self.text_model = genai.GenerativeModel(self.config.GEMINI_GENERATION_MODEL_NAME)
        self.embedding_model_name = self.config.GEMINI_EMBEDDING_MODEL_NAME
        self._load_vector_store()
        self._load_prompt_template()
        logging.info("QueryMaster initialization complete.")

    def _load_vector_store(self):
        try:
            client = chromadb.PersistentClient(path=str(self.config.VECTOR_DB_DIR))
            self.collection = client.get_collection(name=self.config.CHROMA_COLLECTION_NAME)
            logging.info(f"Successfully connected to QA ChromaDB collection.")
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
        logging.info(f"Performing QA-based vector search for: '{user_question}'")
        try:
            query_embedding = genai.embed_content(
                model=self.embedding_model_name,
                content=user_question,
                task_type="RETRIEVAL_QUERY"
            )['embedding']
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.VECTOR_SEARCH_TOP_K 
            )
            
            retrieved_docs = []
            for i, doc_text in enumerate(results['documents'][0]):
                retrieved_docs.append({ "text": doc_text, "metadata": results['metadatas'][0][i] })
            logging.info(f"Found {len(retrieved_docs)} candidate answers.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error during QA vector search: {e}")
            return []
            
    def _rerank_documents(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """A robust re-ranking function to find the best documents from a list of candidates."""
        logging.info(f"Re-ranking {len(docs)} documents for relevance...")
        if not docs: return []
        docs_str = "".join(f"Document {i+1}:\n{doc['text']}\n\n" for i, doc in enumerate(docs))
        prompt = f"""From the following documents, identify the top {self.config.RERANK_TOP_N} that are MOST relevant to the user's question.
        User Question: "{query}"
        Documents:\n{docs_str}
        Respond with a comma-separated list of the document numbers. Example: "3,1,5" """
        try:
            response = self.text_model.generate_content(prompt)
            numbers = re.findall(r'\d+', response.text)
            if not numbers: return docs[:self.config.RERANK_TOP_N]
            relevant_indices = [int(n) - 1 for n in numbers]
            reranked_docs = [docs[i] for i in relevant_indices if 0 <= i < len(docs)]
            logging.info(f"Re-ranked documents. Selected top {len(reranked_docs)}.")
            return reranked_docs
        except Exception as e:
            logging.error(f"Error during re-ranking: {e}. Falling back.")
            return docs[:self.config.RERANK_TOP_N]

    def answer_question(self, user_question: str) -> str:
        """
        Main method to answer a question using a two-stage retrieval (Search + Re-rank).
        """
        # --- KEY CHANGE: Re-enabling the two-stage process ---
        # Stage 1: Fast vector search to get candidate documents
        candidate_docs = self._search_vector_store(user_question)
        if not candidate_docs:
            return "متاسفانه اطلاعات مرتبطی برای پاسخ به سوال شما در منابع موجود یافت نشد."

        # Stage 2: Slower, more powerful re-ranking to find the best documents
        relevant_docs = self._rerank_documents(candidate_docs, user_question)
        if not relevant_docs:
            return "پس از بررسی، اطلاعات دقیقی برای پاسخ به سوال شما یافت نشد."

        vector_context_str = "".join(f"Source: {doc['metadata']['source']}\nContent: {doc['text']}\n\n" for doc in relevant_docs)
        
        final_prompt = self.rag_prompt_template.format(
            vector_context=vector_context_str,
            graph_context="Graph context is not used in this version.",
            user_question=user_question
        )
        
        logging.info("Generating final answer with Gemini...")
        try:
            response = self.text_model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error during final answer generation: {e}")
            return "خطایی در هنگام تولید پاسخ رخ داد. لطفاً دوباره تلاش کنید."
