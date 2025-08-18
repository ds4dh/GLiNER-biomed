import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load stopwords
stop_words = set(stopwords.words('english'))


def compute_quality_scores(text: str, tokens=None, sentences=None) -> dict:
    """
    Compute all heuristic scores for a given text.
    Tokenization and sentence splitting are done once.
    """
    if tokens is None:
        tokens = word_tokenize(text)
    if sentences is None:
        sentences = sent_tokenize(text)

    total_words = len(tokens)
    total_sentences = len(sentences)

    # Special character ratio
    non_alpha_words = sum(1 for word in tokens if re.search(r'[^a-zA-Z]', word))
    special_char_ratio = non_alpha_words / total_words if total_words > 0 else 0

    # Average words per sentence
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

    # Capitalization ratio
    uppercase_count = sum(1 for char in text if char.isupper())
    total_letters = sum(1 for char in text if char.isalpha())
    capitalization_ratio = uppercase_count / total_letters if total_letters > 0 else 0

    # Lexical diversity
    unique_words = set(tokens)
    lexical_diversity = len(unique_words) / total_words if total_words > 0 else 0

    # Stopword ratio
    stopword_count = sum(1 for word in tokens if word.lower() in stop_words)
    stopword_ratio = stopword_count / total_words if total_words > 0 else 0

    # Repetition score
    word_freq = Counter(tokens)
    max_freq = max(word_freq.values(), default=0)
    repetition_score = max_freq / total_words if total_words > 0 else 0

    # Newline to sentence ratio
    newline_groups = re.findall(r'\n+', text)
    number_of_newline_groups = len(newline_groups)
    newline_to_sentence_ratio = number_of_newline_groups / total_sentences if total_sentences > 0 else 0

    return {
        "special_char_ratio": special_char_ratio,
        "sentence_count": total_sentences,
        "avg_words_per_sentence": avg_words_per_sentence,
        "capitalization_ratio": capitalization_ratio,
        "lexical_diversity": lexical_diversity,
        "stopword_ratio": stopword_ratio,
        "repetition_score": repetition_score,
        "newline_to_sentence_ratio": newline_to_sentence_ratio,
    }


def process_single_text(text: str, thresholds) -> (str, bool, dict):
    """
    Process a single text and determine if it meets quality thresholds.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str):
        # Return early for invalid text
        return text, False, {}

    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)
    scores = compute_quality_scores(text, tokens, sentences)

    passed = True
    for heuristic, (comp_type, threshold) in thresholds.items():
        score = scores.get(heuristic, 0)
        if comp_type == 'min' and score < threshold:
            passed = False
        elif comp_type == 'max' and score > threshold:
            passed = False

    return text, passed, scores