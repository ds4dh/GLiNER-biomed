import os
import requests
from time import sleep
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    filename="./data/html_download.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Constants
SETIDS_FILE = "./data/hpl_setids.txt"
OUTPUT_DIR = "./data/hpl_html_files"
BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid={setid}"
MAX_RETRIES = 3
BATCH_SIZE = 100  # Number of Set IDs to process at a time
MAX_WORKERS = 5  # Number of threads for parallel processing

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_setids(file_path):
    """
    Load Set IDs from a file into a list.
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_already_processed_setids(output_dir):
    """
    Identify already processed Set IDs based on existing files in the output directory.
    """
    processed_files = os.listdir(output_dir)
    return {os.path.splitext(filename)[0] for filename in processed_files if filename.endswith(".html")}

def save_html_content(setid, content):
    """
    Save HTML content to a file named after the Set ID.
    """
    file_path = os.path.join(OUTPUT_DIR, f"{setid}.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def fetch_html(setid):
    """
    Fetch HTML content for a given Set ID with retries.
    """
    url = BASE_URL.format(setid=setid)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                logging.warning(f"Failed to fetch {setid}: HTTP {response.status_code}")
        except requests.RequestException as e:
            logging.warning(f"Request failed for {setid}: {e}")
        retries += 1
        sleep(1)  # Backoff before retry
    logging.error(f"Failed to fetch {setid} after {MAX_RETRIES} retries")
    return None

def process_setid(setid):
    """
    Process a single Set ID: fetch and save HTML content.
    """
    html_content = fetch_html(setid)
    if html_content:
        save_html_content(setid, html_content)
        logging.info(f"Successfully processed {setid}")
    else:
        logging.error(f"Failed to process {setid}")

def process_setids_in_batches(setids):
    """
    Process Set IDs in batches using ThreadPoolExecutor for parallelism.
    """
    for i in tqdm(range(0, len(setids), BATCH_SIZE), desc="Processing Batches"):
        batch = setids[i:i + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(process_setid, batch)

if __name__ == "__main__":
    # Load all Set IDs
    setids = load_setids(SETIDS_FILE)
    print(f"Loaded {len(setids)} total Set IDs.")

    # Identify already processed Set IDs
    processed_setids = get_already_processed_setids(OUTPUT_DIR)
    print(f"Found {len(processed_setids)} already processed Set IDs.")

    # Filter Set IDs to process
    remaining_setids = [setid for setid in setids if setid not in processed_setids]
    print(f"{len(remaining_setids)} Set IDs remaining to process.")

    # Process remaining Set IDs
    process_setids_in_batches(remaining_setids)

    print(f"HTML content for remaining Set IDs has been saved to {OUTPUT_DIR}.")
    logging.info("Processing complete.")