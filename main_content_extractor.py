import os
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv
from tqdm import tqdm # Library to show a progress bar

# --- Configuration ---
load_dotenv()

# --- Constants ---
PDF_FILE_PATH = "00_data/pdf/Regulations.pdf"
OUTPUT_DIR = "03_output"
# The final clean, structured text will be saved here
FINAL_OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "full_regulations.md")
# A temporary directory to store page images
TEMP_IMAGE_DIR = os.path.join(OUTPUT_DIR, "temp_page_images")


def setup_gemini():
    """Configures the Gemini API key and creates necessary directories."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in the .env file or environment.")
        genai.configure(api_key=api_key)
        
        # Create output directories if they don't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Error during setup: {e}")
        return False

def process_pdf_page(page, page_number):
    """Converts a single PDF page to an image and sends it to Gemini for extraction."""
    # Increase image resolution for better OCR
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Save the image temporarily for potential review/debugging
    temp_image_path = os.path.join(TEMP_IMAGE_DIR, f"page_{page_number}.png")
    pix.save(temp_image_path)

    image_bytes = pix.tobytes("png")
    page_image = Image.open(io.BytesIO(image_bytes))

    # Prepare and send request to the Gemini model
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
    
    try:
        response = model.generate_content([prompt, page_image])
        return response.text
    except Exception as e:
        print(f"  - Error processing page {page_number}: {e}")
        return f"\n\n--- ERROR ON PAGE {page_number} ---\n\n"


def main():
    """Main function to run the full content extraction process."""
    print("Starting the full content extraction process...")

    if not setup_gemini():
        print("Halting execution due to setup error.")
        return

    try:
        doc = fitz.open(PDF_FILE_PATH)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return

    # Open the output file in write mode
    with open(FINAL_OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        # Using tqdm for a nice progress bar
        for page_num in tqdm(range(1, len(doc) + 1), desc="Processing Pages"):
            page = doc.load_page(page_num - 1)
            extracted_text = process_pdf_page(page, page_num)
            
            # Write the result of each page to the file
            f.write(extracted_text)
            f.write("\n\n---\n\n") # Add a separator between pages

    print(f"\nProcess completed successfully!")
    print(f"Full extracted text saved to '{FINAL_OUTPUT_FILE_PATH}'")
    print("Please review the file to ensure accuracy.")

if __name__ == "__main__":
    main()