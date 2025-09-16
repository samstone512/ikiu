import os
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# --- Constants ---
PDF_FILE_PATH = "00_data/pdf/Regulations.pdf"
# Page number to test (e.g., page 12 which was two-column)
TARGET_PAGE_NUMBER = 12
# Let's save outputs to the 03_output directory to keep the project clean
OUTPUT_IMAGE_PATH = f"03_output/page_{TARGET_PAGE_NUMBER}_image.png"

def setup_gemini():
    """Configures the Gemini API key from environment variables."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the .env file or environment.")
        genai.configure(api_key=api_key)
        # Create output directory if it doesn't exist
        os.makedirs("03_output", exist_ok=True)
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

def convert_pdf_page_to_image(pdf_path, page_number):
    """Converts a specific page of a PDF file to a PIL Image."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return None

    if page_number < 1 or page_number > len(doc):
        print(f"Error: Page number '{page_number}' is out of range. Total pages: {len(doc)}")
        return None
    
    # Page numbering in PyMuPDF is 0-indexed
    page = doc.load_page(page_number - 1) 
    
    # Increase image resolution for better OCR (DPI: Dots Per Inch)
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Save the image for review
    pix.save(OUTPUT_IMAGE_PATH)
    print(f"Image of page {page_number} saved to '{OUTPUT_IMAGE_PATH}'")

    # Convert to bytes for the Gemini library
    image_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(image_bytes))

def main():
    """Main function to run the content extraction process."""
    print("Starting the content extraction process from PDF page...")

    if not setup_gemini():
        print("Halting execution due to API key configuration error.")
        return

    # Step 1: Convert PDF page to image
    page_image = convert_pdf_page_to_image(PDF_FILE_PATH, TARGET_PAGE_NUMBER)

    if page_image is None:
        print("Halting execution due to image conversion failure.")
        return

    # Step 2: Prepare and send request to the Gemini model
    # UPDATED: Using a newer, more stable model name
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = """
    You are a document analysis expert.
    Your task is to accurately extract the text content from the provided image.
    Strictly follow these rules:
    1.  Preserve the document's structure completely.
    2.  If the page has two columns, extract the content of the right column first, then the left column.
    3.  Convert tables into a clean and readable Markdown format.
    4.  Do not summarize any part of the text; return all content in full.
    5.  Identify headings, subheadings, and lists using Markdown syntax.
    """

    print("Sending request to Gemini... (this may take a moment)")
    try:
        response = model.generate_content([prompt, page_image])
        extracted_text = response.text

        # Step 3: Display the result
        print("\n--- Result from Gemini ---\n")
        print(extracted_text)
        print("\n--------------------------\n")
        print("Process completed successfully.")

    except Exception as e:
        print(f"An error occurred while communicating with Gemini: {e}")


if __name__ == "__main__":
    main()