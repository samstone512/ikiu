# config.py
# Centralized configuration file for Project Danesh

from pathlib import Path

# --- IMPORTANT: USER CONFIGURATION ---
# The user needs to set their Google API Key in Google Colab's secrets.
# This script will read it from there.
# The target URL for crawling must also be set by the user.

# The base URL of the university's regulations page
# EXAMPLE: "https://www.university.edu/rules"
CRAWL_URL = "https://www.example.edu/regulations"  # <--- !!! CHANGE THIS !!!
UNIVERSITY_DOMAIN = "https://www.example.edu"   # <--- !!! CHANGE THIS for relative links !!!

# --- DIRECTORY SETUP ---
# Defines the folder structure for our data.
# Using pathlib makes paths work on any operating system.
DATA_DIR = Path("data")
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_TEXT_DIR = DATA_DIR / "processed_text"
