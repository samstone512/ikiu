# src/query_master/rag_pipeline.py
# Final version with Hypothetical Document Embeddings (HyDE) for superior retrieval.

import logging
import networkx as nx
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any

import config

class QueryMaster:
    def __init__(self):
        logging.info("Initializing QueryMaster...")
        self.config = config
        self.text_model = None
        self.embedding_model_name = None
        self.collection = None
        self.graph = None
        self.rag_prompt_template = ""
        
        self._load_models()
        self._load_vector_store()
        self._load_knowledge_graph()
        self._load_prompt_template()
        logging.info("QueryMaster initialization complete.")

    def _load_models(self):
        try:
            self.text_model = genai.GenerativeModel(self.config.GEMINI_GENERATION_MODEL_NAME)
            self.embedding_model_name = self.config.GEMINI_EMBEDDING_MODEL_NAME
            logging.info("Successfully loaded Gemini models.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load Gemini models. Error: {e}")
            raise

    def _load_vector_store(self):
        try:
            client = chromadb.PersistentClient(path=str(self.config.VECTOR_DB_DIR))
            self.collection = client.get_collection(name=self.config.CHROMA_COLLECTION_NAME)
            logging.info(f"Successfully connected to ChromaDB collection: '{self.config.CHROMA_COLLECTION_NAME}'.")
        except Exception as e:
            logging.error(f"FATAL: Failed to connect to ChromaDB. Error: {e}")
            raise

    def _load_knowledge_graph(self):
        try:
            graph_path = self.config.KNOWLEDGE_GRAPH_DIR / "knowledge_graph.graphml"
            self.graph = nx.read_graphml(graph_path)
            logging.info(f"Successfully loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load knowledge graph. Error: {e}")
            raise
            
    def _load_prompt_template(self):
        try:
            with open(self.config.RAG_PROMPT_PATH, 'r', encoding='utf-8') as f:
                self.rag_prompt_template = f.read()
            logging.info("Successfully loaded RAG prompt template.")
        except Exception as e:
            logging.error(f"FATAL: Could not read RAG prompt file. Error: {e}")
            raise

    def _generate_hypothetical_answer(self, query_text: str) -> str:
        """
        NEW (HyDE): Generates a hypothetical, detailed answer to the user's query
        to be used for embedding-based search.
        """
        logging.info("Generating hypothetical answer for HyDE...")
        prompt = f"""
        لطفاً به این سوال یک پاسخ جامع و کامل بدهید. این پاسخ برای جستجوی اسناد مرتبط استفاده خواهد شد، بنابراین باید شامل کلمات کلیدی و مفاهیم احتمالی باشد.
        سوال: {query_text}
        """
        try:
            response = self.text_model.generate_content(prompt)
            logging.info("Hypothetical answer generated successfully.")
            return response.text
        except Exception as e:
            logging.error(f"Error generating hypothetical answer: {e}")
            return query_text # Fallback to the original query

    def _search_vector_store(self, search_text: str) -> List[Dict[str, Any]]:
        """
        Performs vector search using the provided text (which can be the original
        query or a hypothetical answer).
        """
        logging.info(f"Performing vector search...")
        try:
            query_embedding = genai.embed_content(
                model=self.embedding_model_name,
                content=search_text,
                task_type="RETRIEVAL_DOCUMENT" # Use DOCUMENT type for the rich hypo-answer
            )['embedding']
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.VECTOR_SEARCH_TOP_K 
            )
            
            retrieved_docs = []
            for i, doc_text in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "text": doc_text,
                    "metadata": results['metadatas'][0][i]
                })
            logging.info(f"Found {len(retrieved_docs)} initial candidates in vector store.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []

    def _rerank_documents(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        # This function remains the same, but now it receives much better candidates.
        logging.info(f"Re-ranking {len(docs)} documents for relevance...")
        if not docs: return []
        docs_str = "".join(f"Document {i+1}:\n{doc['text']}\n\n" for i, doc in enumerate(docs))
        prompt = f"""From the following documents, identify the top {self.config.RERANK_TOP_N} that are MOST relevant to the user's question.
        User Question: "{query}"
        Documents:\n{docs_str}
        Respond with a comma-separated list of the document numbers. Example: "3,1,5" """
        try:
            response = self.text_model.generate_content(prompt)
            relevant_indices = [int(i.strip()) - 1 for i in response.text.split(',')]
            reranked_docs = [docs[i] for i in relevant_indices if i < len(docs)]
            logging.info(f"Re-ranked documents. Selected top {len(reranked_docs)}.")
            return reranked_docs
        except Exception as e:
            logging.error(f"Error during re-ranking: {e}. Falling back to original list.")
            return docs[:self.config.RERANK_TOP_N]

    def _search_knowledge_graph(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        # This function also remains the same.
        logging.info("Searching knowledge graph for additional context...")
        all_nodes = list(self.graph.nodes())
        found_nodes = set()
        for doc in retrieved_docs:
            for node in all_nodes:
                if node.lower() in doc['text'].lower():
                    found_nodes.add(node)
        if not found_nodes: return "No additional context found."
        graph_context = ""
        for node in found_nodes:
            neighbors = list(nx.bfs_edges(self.graph, source=node, depth_limit=self.config.GRAPH_SEARCH_DEPTH))
            related_items = [n[1] for n in neighbors]
            graph_context += f"- For topic '{node}', related concepts are: {', '.join(related_items) if related_items else 'None'}\n"
        logging.info(f"Found context for {len(found_nodes)} nodes.")
        return graph_context

    def answer_question(self, user_question: str) -> str:
        """
        The main method to answer a user's question, now using the HyDE strategy.
        """
        # Step 1: NEW (HyDE) - Generate a hypothetical answer
        hypothetical_answer = self._generate_hypothetical_answer(user_question)

        # Step 2: Search the vector store using the hypothetical answer
        initial_docs = self._search_vector_store(hypothetical_answer)
        if not initial_docs:
            return "متاسفانه اطلاعات مرتبطی برای پاسخ به سوال شما در منابع موجود یافت نشد."

        # Step 3: Re-rank the documents to find the most relevant ones
        relevant_docs = self._rerank_documents(initial_docs, user_question)
        if not relevant_docs:
            return "پس از بررسی، اطلاعات دقیقی برای پاسخ به سوال شما یافت نشد."

        # Step 4 & 5: Build context and generate the final answer (unchanged)
        vector_context_str = "".join(f"Source: {doc['metadata']['source']}\nContent: {doc['text']}\n\n" for doc in relevant_docs)
        graph_context_str = self._search_knowledge_graph(relevant_docs)
        final_prompt = self.rag_prompt_template.format(
            vector_context=vector_context_str,
            graph_context=graph_context_str,
            user_question=user_question
        )
        logging.info("Generating final answer with Gemini...")
        try:
            response = self.text_model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error during final answer generation: {e}")
            return "خطایی در هنگام تولید پاسخ رخ داد. لطفاً دوباره تلاش کنید."
