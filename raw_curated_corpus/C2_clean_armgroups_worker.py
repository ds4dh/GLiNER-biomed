import re
from nltk.tokenize import sent_tokenize

# Define the regex pattern
pattern = re.compile(r'^[A-Z].*\d.*\.$')

def passes_simple_filtering(text, min_sentences=2):
    """
    Checks if the given text passes the filtering logic:
    - Matches the specified regex pattern.
    - Has at least `min_sentences` sentences.
    """
    # Check if the text matches the regex pattern
    if not pattern.match(text):
        return False

    # Tokenize into sentences and check the count
    sentences = sent_tokenize(text)
    return len(sentences) >= min_sentences


def process_single_text(text, min_sentences=2):
    """
    Process a single text to determine if it passes the filtering logic.
    Returns the text and a boolean indicating whether it passes.
    """
    if not isinstance(text, str):
        return text, False  # Non-string inputs fail automatically

    # Apply the filtering logic
    passes_filter = passes_simple_filtering(text, min_sentences)
    return text, passes_filter