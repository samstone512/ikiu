# src/query_master/ui.py
# This module builds the Gradio web interface for the chatbot.

import gradio as gr
import logging

# Import our main RAG pipeline class
from src.query_master import QueryMaster

def create_chatbot_interface():
    """
    Creates and launches the Gradio ChatInterface for Project Danesh.
    """
    logging.info("--- Creating Gradio Chatbot Interface ---")

    # Step 1: Instantiate the QueryMaster.
    # This is the most important step. We create the object once, so all the
    # heavy models and data are loaded only at the start of the application.
    try:
        query_master = QueryMaster()
        logging.info("QueryMaster instance created successfully.")
    except Exception as e:
        # If the backend fails to load, we display an error and stop.
        logging.error(f"FATAL: Could not instantiate QueryMaster. UI cannot start. Error: {e}", exc_info=True)
        print("\n" + "="*50)
        print(" ERROR: Backend failed to initialize. See logs for details. ")
        print("="*50 + "\n")
        return

    # Step 2: Define the function that will be called on every user interaction.
    # Gradio's ChatInterface expects a function that takes `message` and `history` as arguments.
    def chat_response(message, history):
        """
        This function is the bridge between the Gradio UI and our RAG backend.
        
        Args:
            message (str): The user's input from the chatbox.
            history (List[List[str]]): The conversation history.

        Returns:
            str: The chatbot's response.
        """
        logging.info(f"User query received: '{message}'")
        # Call the main method of our QueryMaster to get the answer
        answer = query_master.answer_question(message)
        logging.info(f"Chatbot response generated.")
        return answer

    # Step 3: Configure and launch the Gradio ChatInterface.
    logging.info("Configuring Gradio interface...")
    chatbot_ui = gr.ChatInterface(
        fn=chat_response,
        title="🎓 Project Danesh Chatbot",
        description="""
        Welcome to Danesh, your AI assistant for university regulations.
        Ask me any questions about the university's rules and I will try to answer based on the official documents.
        به دانِش، دستیار هوش مصنوعی شما برای آیین‌نامه‌های دانشگاه خوش آمدید.
        هر سوالی در مورد قوانین دانشگاه دارید بپرسید، من تلاش می‌کنم بر اساس اسناد رسمی به آن پاسخ دهم.
        """,
        # --- UPDATED EXAMPLES ---
        # These examples are now directly related to the provided PDF documents.
        examples=[
            ["شرایط استفاده از فرصت مطالعاتی برای اعضای هیئت علمی چیست؟"],
            ["معیارهای انتخاب پژوهشگر برگزیده دانشگاه چیست؟"],
            ["سرقت علمی به چه معناست و چه مصادیقی دارد؟"]
        ],
        theme="soft",
    )

    logging.info("Launching Gradio interface...")
    # The launch() method creates a local web server and provides a public link if needed.
    chatbot_ui.launch(share=True, debug=True)
