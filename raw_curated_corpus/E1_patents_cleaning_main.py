import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from E0_patents_cleaning_worker import process_single_text

def evaluate_text(args):
    """
    Wrapper for passing arguments to `process_single_text` via multiprocessing.
    """
    text, thresholds = args
    return process_single_text(text, thresholds)


def process_csv(input_csv, output_csv, thresholds):
    """
    Load, process, and save a CSV file with quality-based filtering, using multiprocessing.
    For the biomedical patents dataset, we will:
      - read the patents CSV
      - rename columns: publication_number -> filename, title -> info_details, description -> text
      - apply cleaning and thresholds
      - output a CSV with filename, info_details, text
    """
    # Load the CSV
    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv, usecols=["publication_number", "title", "description"])
    
    # Check if necessary columns exist
    required_cols = ["publication_number", "title", "description"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the CSV.")

    # Rename columns
    df = df.rename(columns={
        "publication_number": "filename",
        "title": "info_details",
        "description": "text"
    })
    
    # Preprocess the text column: remove NaNs, ensure strings, strip whitespace
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    total_rows = len(df)

    # Extract text data
    text_list = df["text"].tolist()

    # Prepare arguments for multiprocessing
    args_list = [(text, thresholds) for text in text_list]

    # Multiprocessing setup
    print("Processing texts using multiprocessing...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_text, args_list), total=total_rows))

    # Collect passed texts and removal reasons
    passed_texts = []
    removal_reasons = {key: 0 for key in thresholds.keys()}
    for text, passed, scores in results:
        if passed:
            passed_texts.append(text)
        else:
            for heuristic, (comp_type, threshold) in thresholds.items():
                score = scores.get(heuristic, 0)
                if comp_type == 'min' and score < threshold:
                    removal_reasons[heuristic] += 1
                elif comp_type == 'max' and score > threshold:
                    removal_reasons[heuristic] += 1

    # Filter DataFrame to include only passed texts
    filtered_df = df[df["text"].isin(passed_texts)]

    # Deduplicate the filtered DataFrame by text column
    filtered_df = filtered_df.drop_duplicates(subset="text")

    remaining_rows = len(filtered_df)

    # Save the cleaned DataFrame
    # The user wants the final CSV to have exactly 3 columns: filename, info_details, text
    cleaned_df = filtered_df[["filename", "info_details", "text"]]
    print(f"Saving cleaned CSV to: {output_csv}")
    cleaned_df.to_csv(output_csv, index=False)
    print("Cleaned CSV saved successfully!")

    # Generate and print summary report
    print("\nSummary Report:")
    print(f"Total rows before filtering: {total_rows}")
    print(f"Total rows after filtering: {remaining_rows}")
    print(f"Total rows removed: {total_rows - remaining_rows}")
    print("Reasons for removal:")
    for heuristic, count in removal_reasons.items():
        print(f"  {heuristic}: {count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and clean a patents CSV file with quality heuristics.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input patents CSV file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument('--special_char_ratio', type=float, default=0.3, help="Maximum special character ratio.")
    parser.add_argument('--min_sentence_count', type=int, default=6, help="Minimum number of sentences.")
    parser.add_argument('--min_avg_words_per_sentence', type=float, default=10, help="Minimum average words per sentence.")
    parser.add_argument('--max_capitalization_ratio', type=float, default=0.2, help="Maximum capitalization ratio.")
    parser.add_argument('--min_lexical_diversity', type=float, default=0.1, help="Minimum lexical diversity.")
    parser.add_argument('--min_stopword_ratio', type=float, default=0.05, help="Minimum stopword ratio.")
    parser.add_argument('--max_repetition_score', type=float, default=0.2, help="Maximum repetition score.")
    parser.add_argument('--max_newline_to_sentence_ratio', type=float, default=0.15, help="Maximum newline to sentence ratio.")

    args = parser.parse_args()

    thresholds = {
        "special_char_ratio": ("max", args.special_char_ratio),
        "sentence_count": ("min", args.min_sentence_count),
        "avg_words_per_sentence": ("min", args.min_avg_words_per_sentence),
        "capitalization_ratio": ("max", args.max_capitalization_ratio),
        "lexical_diversity": ("min", args.min_lexical_diversity),
        "stopword_ratio": ("min", args.min_stopword_ratio),
        "repetition_score": ("max", args.max_repetition_score),
        "newline_to_sentence_ratio": ("max", args.max_newline_to_sentence_ratio),
    }

    process_csv(args.input_csv, args.output_csv, thresholds)