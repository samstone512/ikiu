import pickle
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
from transformers import pipeline, Pipeline

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent
INPUT_DIR = ROOT_DIR / "03_output"
GRAPH_DATA_PATH = INPUT_DIR / "graph_data.pkl"

# --- Data Structure Definitions ---
@dataclass(frozen=True, eq=True)
class EntityNode:
    text: str
    type: str

# --- KnowledgeGraph Class (Manages graph data and searches) ---
class KnowledgeGraph:
    def __init__(self, nodes: Set[EntityNode], edges: Set[Tuple[EntityNode, EntityNode]]):
        self._nodes = nodes
        self._adjacency_list: Dict[EntityNode, Set[EntityNode]] = {node: set() for node in nodes}
        for node1, node2 in edges:
            self._adjacency_list[node1].add(node2)
            self._adjacency_list[node2].add(node1)
        logging.info(f"Knowledge Graph initialized with {len(self._nodes)} nodes and {len(edges)} edges.")

    def find_neighbors(self, node_text: str) -> Set[EntityNode]:
        target_node = next((node for node in self._nodes if node.text == node_text), None)
        return self._adjacency_list.get(target_node, set()) if target_node else set()

# --- NERPipeline Class (Manages the NER model) ---
class NERPipeline:
    def __init__(self, model_name="HooshvareLab/bert-fa-zwnj-base-ner"):
        logging.info(f"Initializing NER pipeline with model: {model_name}")
        self._pipeline: Pipeline = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple"
        )
        logging.info("NER pipeline initialized successfully.")

    def extract_entities(self, text: str) -> List[EntityNode]:
        try:
            entities = self._pipeline(text)
            return [EntityNode(text=e['word'], type=e['entity_group']) for e in entities]
        except Exception as e:
            logging.error(f"Error during NER extraction: {e}")
            return []

# --- QueryPipeline Class (The main RAG orchestrator) ---
class QueryPipeline:
    def __init__(self, kg: KnowledgeGraph, ner_pipeline: NERPipeline):
        self._kg = kg
        self._ner = ner_pipeline

    def execute(self, question: str) -> str:
        logging.info(f"\n--- Executing new query: '{question}' ---")
        
        # 1. Retrieval Step: Extract entities from the question
        question_entities = self._ner.extract_entities(question)
        if not question_entities:
            return "متاسفانه هیچ موجودیت کلیدی در سوال شما پیدا نکردم. لطفا سوال خود را با کلمات مشخص‌تری مانند نام سازمان‌ها یا افراد بپرسید."
        
        logging.info(f"Found entities in question: {[e.text for e in question_entities]}")

        # 2. Augmentation Step: Gather context from the knowledge graph
        context = ""
        for entity in question_entities:
            neighbors = self._kg.find_neighbors(entity.text)
            if neighbors:
                context += f"اطلاعات مرتبط با '{entity.text}':\n"
                for neighbor in neighbors:
                    context += f"- {neighbor.text} (نوع: {neighbor.type})\n"
        
        if not context:
            return f"موجودیت '{question_entities[0].text}' در گراف دانش پیدا شد، اما هیچ اطلاعات مرتبطی برای آن یافت نشد."

        # 3. Generation Step: Create a prompt and generate the answer
        prompt = f"""
شما یک دستیار هوش مصنوعی هستید که به سوالات مربوط به آیین‌نامه‌های دانشگاهی پاسخ می‌دهید.
بر اساس اطلاعات و بافتار زیر، به سوال کاربر پاسخ دهید. پاسخ شما باید دقیق و فقط بر اساس اطلاعات ارائه شده باشد.

[بافتار بازیابی شده از گراف دانش]
{context}

[سوال کاربر]
{question}

[پاسخ شما]
"""
        
        logging.info("--- Generated Prompt for LLM ---")
        logging.info(prompt)
        
        # In a real application, this prompt would be sent to an LLM like Gemini.
        # Here, we simulate the response for demonstration purposes.
        simulated_answer = "پاسخ شبیه‌سازی شده: بر اساس اطلاعات گراف، موجودیت‌های مرتبط با سوال شما پیدا شد و در پرامپت بالا برای تولید پاسخ نهایی آماده گردید."

        return simulated_answer

def main():
    """Main function to set up and run the full query pipeline."""
    logging.info("--- Starting Phase 09: Query Master (Full RAG Pipeline) ---")

    # Load all components
    graph_data = pickle.load(open(GRAPH_DATA_PATH, 'rb'))
    kg = KnowledgeGraph(graph_data['nodes'], graph_data['edges'])
    ner = NERPipeline()
    
    # Create and run the pipeline
    query_pipeline = QueryPipeline(kg, ner)
    
    # --- Test Case ---
    user_question = "اطلاعاتی در مورد وزارت بهداشت و وزارت علوم به من بده."
    final_answer = query_pipeline.execute(user_question)
    
    logging.info("\n--- FINAL ANSWER ---")
    logging.info(final_answer)

if __name__ == "__main__":
    main()