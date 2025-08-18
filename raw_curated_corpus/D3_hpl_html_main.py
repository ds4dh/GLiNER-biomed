import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from D2_hpl_html_worker import process_html_file

def process_all_html_files(directory: Path):
    """
    Process all HTML files in the given directory using multiprocessing and save results to a CSV file.
    """
    # Get all .html files
    html_files = list(directory.glob("*.html"))
    if not html_files:
        print("No HTML files found in the specified directory.")
        return

    # Use multiprocessing to process files
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_html_file, html_files), total=len(html_files)))

    # Flatten results from all processes
    flattened_results = [item for sublist in results for item in sublist]

    # Write results to a CSV file
    output_csv_path = "./data/hpl_html_raw_text.csv"
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=["filename", "info_details", "text"])
        csv_writer.writeheader()
        csv_writer.writerows(flattened_results)

    print(f"Results have been written to {output_csv_path}")

if __name__ == "__main__":
    folder_path = Path("./data/hpl_html_files")
    process_all_html_files(folder_path)