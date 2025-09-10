# src/data_harvester/ocr.py
# --- FINAL & ROBUST VERSION: Using a powerful Document Understanding Transformer (Donut) ---
# --- PROFESSIONAL FIX V2: Switched to a more suitable model for general document parsing ---

import logging
from pathlib import Path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re

# --- Global variables to load the model only once ---
DONUT_PROCESSOR = None
DONUT_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_donut_model():
    """Loads the Donut model and processor into memory."""
    global DONUT_PROCESSOR, DONUT_MODEL
    if DONUT_MODEL is None:
        logging.info("Loading Donut model for the first time... This may take a few minutes.")
        try:
            # --- KEY CHANGE: Switched from the DocVQA model to a Document Parsing model ---
            # این مدل برای استخراج متن کامل از اسناد مناسب‌تر است
            model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
            
            DONUT_PROCESSOR = DonutProcessor.from_pretrained(model_name)
            DONUT_MODEL = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
            logging.info(f"Donut model '{model_name}' loaded successfully and moved to {DEVICE}.")
        except Exception as e:
            logging.error(f"FATAL: Could not load the Donut model from Hugging Face. Check internet connection and library versions. Error: {e}")
            raise

def extract_text_with_donut(image_path: Path) -> str:
    """
    Extracts structured text from a single image using the Donut model.
    """
    global DONUT_PROCESSOR, DONUT_MODEL
    try:
        # Ensure the model is loaded
        if DONUT_MODEL is None:
            load_donut_model()

        logging.info(f"  - Processing image with Donut: {image_path.name}")
        image = Image.open(image_path).convert("RGB")

        # Prepare the image for the model
        pixel_values = DONUT_PROCESSOR(image, return_tensors="pt").pixel_values
        
        # The task prompt for Donut is crucial. For this model, we use its specific prompt.
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = DONUT_PROCESSOR.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        # Generate the output from the model
        outputs = DONUT_MODEL.generate(
            pixel_values.to(DEVICE),
            decoder_input_ids=decoder_input_ids.to(DEVICE),
            max_length=DONUT_MODEL.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=DONUT_PROCESSOR.tokenizer.pad_token_id,
            eos_token_id=DONUT_PROCESSOR.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[DONUT_PROCESSOR.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode the generated ids to a string
        sequence = DONUT_PROCESSOR.batch_decode(outputs.sequences)[0]
        
        # Clean up the sequence from special tokens
        sequence = sequence.replace(DONUT_PROCESSOR.tokenizer.eos_token, "").replace(DONUT_PROCESSOR.tokenizer.pad_token, "")
        
        # The model sometimes outputs structured data tags. We will clean them up to get the raw text content.
        clean_text = re.sub(r"<.*?>", "", sequence).strip()
        
        logging.info(f"  - Successfully extracted text from {image_path.name}")
        return clean_text

    except Exception as e:
        logging.error(f"  - An error occurred while processing '{image_path.name}' with Donut. Error: {e}", exc_info=True)
        return ""