# src/query_master/rag_pipeline.py
# This module contains the core logic for the GraphRAG pipeline.

import logging
import networkx as nx
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any

# Import the centralized configuration
import config

class QueryMaster:
    """
    Orchestrates the entire GraphRAG pipeline from user query to final answer.
    """
    def __init__(self):
        """
        Initializes all necessary components for the RAG pipeline upon creation.
        This includes loading models, connecting to databases, and loading the graph.
        """
        logging.info("Initializing QueryMaster...")
        self.config = config
        self.text_model = None
        self.embedding_model = None
        self.collection = None
        self.graph = None
        self.rag_prompt_template = ""
        
        self._load_models()
        self._load_vector_store()
        self._load_knowledge_graph()
        self._load_prompt_template()
        logging.info("QueryMaster initialization complete.")

    def _load_models(self):
        """Loads the Gemini models for generation and embedding."""
        try:
            self.text_model = genai.GenerativeModel(self.config.GEMINI_GENERATION_MODEL_NAME)
            # The embedding model is accessed via the top-level genai function,
            # so we just confirm its name is set.
            self.embedding_model_name = self.config.GEMINI_EMBEDDING_MODEL_NAME
            logging.info("Successfully loaded Gemini models.")
        except Exception as e:
            logging.error(f"FATAL: Failed to load Gemini models. Error: {e}")
            raise

    def _load_vector_store(self):
        """Connects to the persistent ChromaDB and gets the collection."""
        try:
            client = chromadb.PersistentClient(path=str(self.config.VECTOR_DB_DIR))
            self.collection = client.get_collection(name=self.config.CHROMA_COLLECTION_NAME)
            logging.info(f"Successfully connected to ChromaDB collection: '{self.config.CHROMA_COLLECTION_NAME}'.")
        except Exception as e:
            logging.error(f"FATAL: Failed to connect to ChromaDB. Error: {e}")
            raise

    def _load_knowledge_graph(self):
        """Loads the knowledge graph from the GraphML file."""
        try:
            graph_path = self.config.KNOWLEDGE_GRAPH_DIR / "knowledge_graph.graphml"
            self.graph = nx.read_graphml(graph_path)
            logging.info(f"Successfully loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        except FileNotFoundError:
            logging.error(f"FATAL: Knowledge graph file not found at {graph_path}.")
            raise
        except Exception as e:
            logging.error(f"FATAL: Failed to load knowledge graph. Error: {e}")
            raise
            
    def _load_prompt_template(self):
        """Loads the RAG prompt template from the text file."""
        try:
            with open(self.config.RAG_PROMPT_PATH, 'r', encoding='utf-8') as f:
                self.rag_prompt_template = f.read()
            logging.info("Successfully loaded RAG prompt template.")
        except Exception as e:
            logging.error(f"FATAL: Could not read RAG prompt file. Error: {e}")
            raise

    def _search_vector_store(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Performs a similarity search in the vector store.
        
        Args:
            query_text (str): The user's question.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the retrieved
                                  documents and their metadata.
        """
        logging.info(f"Performing vector search for query: '{query_text}'")
        try:
            query_embedding = genai.embed_content(
                model=self.embedding_model_name,
                content=query_text,
                task_type="RETRIEVAL_QUERY"
            )['embedding']
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.config.VECTOR_SEARCH_TOP_K
            )
            
            # Combine documents and metadata into a more usable format
            retrieved_docs = []
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i]
                })
            logging.info(f"Found {len(retrieved_docs)} relevant documents in vector store.")
            return retrieved_docs
        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []

    def _search_knowledge_graph(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Searches the knowledge graph for nodes mentioned in the retrieved text
        and finds their neighbors to build additional context.

        Args:
            retrieved_docs (List[Dict[str, Any]]): The documents from the vector search.

        Returns:
            str: A formatted string of context from the knowledge graph.
        """
        logging.info("Searching knowledge graph for additional context...")
        all_nodes = list(self.graph.nodes())
        found_nodes = set()
        
        # Find which nodes from our graph are mentioned in the retrieved text
        for doc in retrieved_docs:
            for node in all_nodes:
                if node.lower() in doc['text'].lower():
                    found_nodes.add(node)
        
        if not found_nodes:
            logging.info("No relevant nodes found in the knowledge graph.")
            return "No additional context found in the knowledge graph."

        # For each found node, get its neighbors up to a certain depth
        graph_context = ""
        for node in found_nodes:
            neighbors = list(nx.bfs_edges(self.graph, source=node, depth_limit=self.config.GRAPH_SEARCH_DEPTH))
            context_line = f"- For topic '{node}', related concepts are: "
            related_items = [n[1] for n in neighbors]
            context_line += ", ".join(related_items) if related_items else "None"
            graph_context += context_line + "\n"
            
        logging.info(f"Found context for {len(found_nodes)} nodes in the knowledge graph.")
        return graph_context

    def answer_question(self, user_question: str) -> str:
        """
        The main method to answer a user's question by running the full RAG pipeline.

        Args:
            user_question (str): The question from the user.

        Returns:
            str: The generated answer from the language model.
        """
        # Step 1: Search the vector store
        vector_context_docs = self._search_vector_store(user_question)
        if not vector_context_docs:
            return "متاسفانه اطلاعات مرتبطی برای پاسخ به سوال شما در منابع موجود یافت نشد."

        # Format the vector context for the prompt
        vector_context_str = ""
        for doc in vector_context_docs:
            vector_context_str += f"Source: {doc['metadata']['source']}\nContent: {doc['text']}\n\n"

        # Step 2: Search the knowledge graph using the results from the vector search
        graph_context_str = self._search_knowledge_graph(vector_context_docs)
        
        # Step 3: Construct the final prompt
        final_prompt = self.rag_prompt_template.format(
            vector_context=vector_context_str,
            graph_context=graph_context_str,
            user_question=user_question
        )
        
        # Step 4: Generate the final answer using the LLM
        logging.info("Generating final answer with Gemini...")
        try:
            response = self.text_model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error during final answer generation: {e}")
            return "خطایی در هنگام تولید پاسخ رخ داد. لطفاً دوباره تلاش کنید."
