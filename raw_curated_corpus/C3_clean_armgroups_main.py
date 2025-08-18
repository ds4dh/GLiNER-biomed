import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from C2_clean_armgroups_worker import process_single_text


def evaluate_text(args):
    """
    Wrapper for passing arguments to `process_single_text` via multiprocessing.
    """
    text, min_sentences = args
    return process_single_text(text, min_sentences)


def process_csv(input_csv, output_csv, min_sentences=2, text_column="text"):
    """
    Load, process, and save a CSV file with the simplified filtering logic, using multiprocessing.
    """
    # Load the CSV
    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the CSV.")

    # Preprocess the text column: remove NaNs, ensure all values are strings, and strip whitespace
    df[text_column] = df[text_column].fillna("").astype(str).str.strip()
    total_rows = len(df)

    # Extract text data
    text_list = df[text_column].tolist()

    # Prepare arguments for multiprocessing
    args_list = [(text, min_sentences) for text in text_list]

    # Multiprocessing setup
    print("Processing texts using multiprocessing...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_text, args_list), total=total_rows))

    # Collect passed texts
    passed_texts = [text for text, passed in results if passed]

    # Filter DataFrame to include only passed texts
    filtered_df = df[df[text_column].isin(passed_texts)]
    # Remove duplicates from the filtered DataFrame
    filtered_df = filtered_df.drop_duplicates(subset=text_column)
    remaining_rows = len(filtered_df)

    # Save the cleaned DataFrame
    print(f"Saving cleaned CSV to: {output_csv}")
    filtered_df.to_csv(output_csv, index=False)
    print("Cleaned CSV saved successfully!")

    # Generate and print summary report
    print("\nSummary Report:")
    print(f"Total rows before filtering: {total_rows}")
    print(f"Total rows after filtering: {remaining_rows}")
    print(f"Total rows removed: {total_rows - remaining_rows}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and clean a CSV file with simplified filtering logic.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument('--min_sentences', type=int, default=2, help="Minimum number of sentences (default: 2).")

    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv, args.min_sentences)