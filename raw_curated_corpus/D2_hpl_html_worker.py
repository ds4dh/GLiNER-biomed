import logging
import re
from pathlib import Path
from typing import List, Optional
from lxml import html
import html2text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

def remove_tables(html_content: str) -> str:
    """
    Remove all <table> elements and their contents from the given HTML content.
    """
    tree = html.fromstring(html_content)
    for table in tree.xpath("//table"):
        parent = table.getparent()
        if parent is not None:
            parent.remove(table)
    return html.tostring(tree, encoding='unicode')

def remove_headings(html_content: str) -> str:
    """
    Remove <h1> and <h2> tags and their content from the HTML.
    """
    html_content = re.sub(r'<h1\b[^>]*>.*?</h1>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    html_content = re.sub(r'<h2\b[^>]*>.*?</h2>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    return html_content

def initialize_html_parser() -> html2text.HTML2Text:
    """
    Initialize and configure the html2text.HTML2Text parser.
    """
    html_parser = html2text.HTML2Text()
    html_parser.ignore_links = True
    html_parser.ignore_images = True
    html_parser.ignore_tables = True
    html_parser.inline_links = False
    html_parser.drop_white_space = True
    html_parser.body_width = 0
    html_parser.ignore_emphasis = True
    return html_parser

def html_to_text(html_content: str, html_parser: html2text.HTML2Text) -> str:
    """
    Convert HTML content to a readable text format using html2text.
    """
    html_content = remove_headings(html_content)
    return html_parser.handle(html_content).strip()

def chunk_by_h1_h2(html_content: str) -> List[str]:
    """
    Chunk the HTML content by <h1> and <h2> tags. Each chunk starts at either
    <h1> or <h2> and includes everything up to the next <h1>/<h2> or the end of the HTML.
    No overlap, no repetition.

    If there are no <h1> or <h2> tags, returns an empty list.
    """
    heading_positions = [m.start() for m in re.finditer(r'<h[12]\b', html_content, flags=re.IGNORECASE)]
    
    if not heading_positions:
        return []

    chunks = []
    for i in range(len(heading_positions)):
        start = heading_positions[i]
        end = heading_positions[i+1] if (i+1 < len(heading_positions)) else len(html_content)
        chunk_html = html_content[start:end]
        chunks.append(chunk_html)

    return chunks

def process_html_file(file_path: Path) -> List[dict]:
    """
    Process a single HTML file and return a list of dictionaries with chunk data.
    """
    if not file_path.is_file():
        logging.warning(f"File {file_path} does not exist or is not a regular file.")
        return []

    try:
        with file_path.open('r', encoding='utf-8') as f:
            original_html = f.read()

        cleaned_html = remove_tables(original_html)
        html_chunks = chunk_by_h1_h2(cleaned_html)

        html_parser = initialize_html_parser()
        return [
            {"filename": file_path.name, "info_details": f"Chunk {i+1}", "text": html_to_text(chunk, html_parser)}
            for i, chunk in enumerate(html_chunks)
        ]
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return []