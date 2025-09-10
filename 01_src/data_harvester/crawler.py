# src/data_harvester/crawler.py

import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crawl_and_download_pdfs(start_url: str, domain: str, save_dir: Path):
    """
    Crawls a website to find and download PDF files.
    NOTE: This function is a TEMPLATE and MUST be adapted by the user.
    """
    logging.info(f"Starting crawl at: {start_url}")
    try:
        response = requests.get(start_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all anchor tags (<a>) with a link ending in '.pdf'
        pdf_links = soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))

        if not pdf_links:
            logging.warning("No PDF links found. User may need to adjust scraping logic.")
            return

        logging.info(f"Found {len(pdf_links)} PDF links.")
        for link in pdf_links:
            pdf_url = link['href']
            if not pdf_url.startswith('http'):
                pdf_url = f"{domain}{pdf_url}"

            file_name = Path(pdf_url).name
            save_path = save_dir / file_name

            if save_path.exists():
                logging.info(f"Skipping '{file_name}', already downloaded.")
                continue

            logging.info(f"Downloading '{file_name}'...")
            try:
                pdf_response = requests.get(pdf_url, timeout=30)
                pdf_response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(pdf_response.content)
                logging.info(f"Successfully saved to '{save_path}'")
            except requests.RequestException as e:
                logging.error(f"Error downloading {pdf_url}: {e}")
            time.sleep(1)

    except requests.RequestException as e:
        logging.error(f"Error accessing start URL {start_url}: {e}")
