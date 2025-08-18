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


def process_arm_groups(args):
    """
    Processes a single JSON file to extract armGroups descriptions.
    Args:
        args (tuple): (file_name, input_dir)

    Returns:
        list: A list of dictionaries with the extracted armGroups descriptions if available.
    """
    file_name, input_dir = args
    file_path = os.path.join(input_dir, file_name)
    results = []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract 'armGroups descriptions'
        arm_groups_path = ['protocolSection', 'armsInterventionsModule', 'armGroups']
        arm_groups = extract_value_with_path(data, arm_groups_path)
        if arm_groups:
            for arm in arm_groups:
                if 'description' in arm:
                    arm_description_path = arm_groups_path + ['description']
                    results.append({
                        "filename": file_name,
                        "info_details": " | ".join(arm_description_path),
                        "text": arm['description'].strip()
                    })

    except (json.JSONDecodeError, FileNotFoundError):
        # Skip files that cannot be decoded or are missing
        pass

    return results