import os
import pandas as pd
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from C0_ct_armgroups_worker import process_arm_groups


def main():
    """
    Main function to process clinical trial armGroups descriptions using multiprocessing.
    """
    parser = argparse.ArgumentParser(description="Extract armGroups descriptions from clinical trial JSON files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing JSON files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file (including filename)."
    )
    args = parser.parse_args()

    # List all JSON files in the input directory
    input_dir = args.input_dir
    output_file = args.output_file
    json_files = sorted([file for file in os.listdir(input_dir) if file.endswith('.json')])

    # Create arguments for multiprocessing
    file_args = [(file, input_dir) for file in json_files]

    # Use multiprocessing to process files
    results = []
    with Pool(cpu_count()) as pool:
        for file_results in tqdm(pool.imap_unordered(process_arm_groups, file_args), total=len(file_args), desc="Processing files"):
            results.extend(file_results)

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()