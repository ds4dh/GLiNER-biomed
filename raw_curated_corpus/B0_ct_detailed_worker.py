import os
import json


def extract_value_with_path(data, path):
    """
    Helper function to extract a value from a nested JSON structure using a list of keys.
    """
    try:
        for key in path:
            data = data[key]
        return data
    except (KeyError, TypeError):
        return None


def process_detailed_description(args):
    """
    Processes a single JSON file to extract detailedDescription.
    Args:
        args (tuple): (file_name, input_dir)

    Returns:
        dict: A dictionary with the extracted detailedDescription if available.
    """
    file_name, input_dir = args
    file_path = os.path.join(input_dir, file_name)
    results = []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract 'detailedDescription'
        detailed_description_path = ['protocolSection', 'descriptionModule', 'detailedDescription']
        detailed_description = extract_value_with_path(data, detailed_description_path)
        if detailed_description:
            results.append({
                "filename": file_name,
                "info_details": " | ".join(detailed_description_path),
                "text": detailed_description.strip()
            })

    except (json.JSONDecodeError, FileNotFoundError):
        # Skip files that cannot be decoded or are missing
        pass

    return results