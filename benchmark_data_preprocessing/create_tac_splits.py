import os
import random
import json
import xml.etree.ElementTree as ET
import spacy
from tqdm.auto import tqdm


def load_xml_files_from_folder(folder_path):
    """
    Load and parse all XML files in a specified folder.

    Args:
        folder_path (str): The path to the folder containing XML files.

    Returns:
        dict: A dictionary where keys are filenames and values are the parsed XML root elements.
    """
    xml_files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                xml_files[filename] = root
            except ET.ParseError as e:
                print(f"ParseError in file {filename}: {str(e)}")
    return xml_files


def collect_entity_types_from_xml(xml_files, entity_type_mapping):
    """
    Collect all unique entity types from the XML files, applying the mapping.

    Args:
        xml_files (dict): Dictionary of XML root elements.
        entity_type_mapping (dict): Dictionary to rename entity types.

    Returns:
        set: A set of all unique entity types after mapping.
    """
    all_entity_types = set()

    for file_name, root in xml_files.items():
        mentions = root.findall(".//Mention")
        for mention in mentions:
            mention_type = mention.get("type")
            # Rename entity type if in the mapping
            mention_type = entity_type_mapping.get(mention_type, mention_type)
            all_entity_types.add(mention_type)

    print(f"Collected {len(all_entity_types)} unique entity types.")
    return all_entity_types


def process_xml_files(
    files,
    xml_files,
    output_file,
    split_type,
    consider_discontinuous,
    all_entity_types,
    entity_type_mapping,
):
    """
    Process XML files and prepare data for NER tasks, matching the format and logic of the first code.

    Args:
        files (list): List of filenames to process.
        xml_files (dict): Dictionary of XML root elements.
        output_file (str): Path to the output JSON file.
        split_type (str): Type of data split ('train', 'val', or 'test').
        consider_discontinuous (dict): Dict specifying whether to consider discontinuous entities for each split.
        all_entity_types (set): Set of all unique entity types.
        entity_type_mapping (dict): Dictionary to rename entity types.
    """
    nlp = spacy.load("en_core_web_sm")

    data = []
    excluded_discontinuous_count = 0  # Counter for excluded entities
    total_entities = 0  # Counter for total entities processed

    for file_name in tqdm(files, desc=f"Processing {split_type} set"):
        root = xml_files[file_name]

        # Process each section separately
        sections = root.findall(".//Section")
        for section in sections:
            section_id = section.get("id")
            section_text = section.text or ""

            # Skip empty sections
            if not section_text.strip():
                continue

            # Read mentions for this section
            entities = []
            section_entity_types = set()  # Track entity types in this section

            # Find mentions that belong to this section
            mentions = root.findall(f".//Mention[@section='{section_id}']")

            for mention in mentions:
                mention_type = mention.get("type")
                # Rename entity type if in the mapping
                mention_type = entity_type_mapping.get(mention_type, mention_type)
                section_entity_types.add(mention_type)

                start_positions = mention.get("start").split(",")
                lengths = mention.get("len").split(",")

                if len(start_positions) != len(lengths):
                    print(
                        f"Warning: Mismatch in lengths and starts in mention '{mention.get('id')}'"
                    )
                    continue

                spans = []
                for start_str, length_str in zip(start_positions, lengths):
                    start = int(start_str)
                    length = int(length_str)
                    end = start + length
                    spans.append((start, end))

                # Update entity counts
                total_entities += (
                    1  # Count the mention as one entity regardless of spans
                )

                # Exclude or include discontinuous entities based on setting
                if len(spans) > 1:
                    if not consider_discontinuous[split_type]:
                        # Skip discontinuous spans
                        excluded_discontinuous_count += 1
                        continue
                    else:
                        # Handle discontinuous spans as separate entities
                        for start_char, end_char in spans:
                            # Adjust start_char and end_char relative to the section
                            # For mentions relative to section text, no adjustment needed
                            entities.append(
                                {
                                    "start_char": start_char,
                                    "end_char": end_char,
                                    "type": mention_type,
                                }
                            )
                else:
                    start_char, end_char = spans[0]
                    # Adjust start_char and end_char relative to the section
                    # For mentions relative to section text, no adjustment needed
                    entities.append(
                        {
                            "start_char": start_char,
                            "end_char": end_char,
                            "type": mention_type,
                        }
                    )

            # Tokenize section_text with spaCy
            doc = nlp(section_text)
            tokens = [token.text for token in doc]

            # Map character offsets to token indices using doc.char_span
            token_spans = []
            for entity in entities:
                start_char = entity["start_char"]
                end_char = entity["end_char"]
                entity_type = entity["type"]
                span = doc.char_span(start_char, end_char, alignment_mode="expand")
                if span is not None:
                    start_token_index = span.start
                    end_token_index = span.end - 1  # end index is inclusive
                    token_spans.append(
                        [start_token_index, end_token_index, entity_type]
                    )
                    #print(section_id, start_token_index, end_token_index, entity_type, tokens[start_token_index:end_token_index+1])
                else:
                    print(
                        f"Warning: could not create token span for entity in file {file_name} section {section_id} at chars {start_char}-{end_char}"
                    )

            # Identify absent entity types in this section
            negative_entity_types = list(all_entity_types - section_entity_types)
            to_append = {
                    "tokenized_text": tokens,
                    "ner": token_spans,
                    "negatives": negative_entity_types,
                }
            
            assert negative_entity_types or token_spans
            
            if not negative_entity_types:
                del to_append["negatives"]

            data.append(to_append)

    # Print statistics after processing
    print(f"Processed {len(files)} files for {split_type} set.")
    print(f"Total entities processed: {total_entities}")
    print(
        f"Total entities excluded due to non-continuity: {excluded_discontinuous_count}"
    )

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save data to output_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_and_process_data(
    train_folder,
    test_folder,
    output_base,
    seed,
    consider_discontinuous,
    entity_type_mapping,
):
    """
    Split the data into training, validation, and testing sets, and process each set.

    Args:
        train_folder (str): Path to the folder containing training XML files.
        test_folder (str): Path to the folder containing testing XML files.
        output_base (str): Base path for output JSON files.
        seed (int): Random seed for reproducibility.
        consider_discontinuous (dict): Dict specifying whether to consider discontinuous entities for each split.
        entity_type_mapping (dict): Dictionary to rename entity types.
    """
    # Load XML files
    train_xml_files = load_xml_files_from_folder(train_folder)
    test_xml_files = load_xml_files_from_folder(test_folder)

    # Combine train and val data, then split
    all_train_xml_files = train_xml_files

    # Set the seed for reproducibility
    random.seed(seed)

    # Convert the dictionary items into a list of filenames
    xml_filenames = list(all_train_xml_files.keys())

    # Shuffle the list randomly
    random.shuffle(xml_filenames)

    # Calculate the split point for a 90-10 split
    split_index = int(0.9 * len(xml_filenames))

    # Split the list into train and validation sets
    train_files = xml_filenames[:split_index]
    val_files = xml_filenames[split_index:]

    # Collect all entity types from all data, applying the mapping
    all_entity_types = collect_entity_types_from_xml(
        {**train_xml_files, **test_xml_files}, entity_type_mapping
    )

    print(f"All entity types after mapping: {all_entity_types}\n")

    # Process each split
    process_xml_files(
        train_files,
        train_xml_files,
        os.path.join(output_base, "train.json"),
        "train",
        consider_discontinuous,
        all_entity_types,
        entity_type_mapping,
    )
    process_xml_files(
        val_files,
        train_xml_files,
        os.path.join(output_base, "val.json"),
        "val",
        consider_discontinuous,
        all_entity_types,
        entity_type_mapping,
    )
    process_xml_files(
        list(test_xml_files.keys()),
        test_xml_files,
        os.path.join(output_base, "test.json"),
        "test",
        consider_discontinuous,
        all_entity_types,
        entity_type_mapping,
    )


if __name__ == "__main__":
    # Define paths and parameters
    train_folder_path = "./raw_data/tac/train_xml"
    test_folder_path = "./raw_data/tac/gold_xml"
    output_base = "./data/tac"
    seed = 42
    consider_discontinuous = {"train": False, "val": False, "test": False}
    entity_type_mapping = {
        'AdverseReaction': 'Adverse drug event',
        'Severity': 'Adverse drug event severity',
        'Factor': 'Adverse drug event contextual modifier',
        'DrugClass': 'Drug class',
        'Negation': 'Adverse drug event negation cue',
        'Animal': 'Animal model'
    }

    # Split data and process
    split_and_process_data(
        train_folder_path,
        test_folder_path,
        output_base,
        seed,
        consider_discontinuous,
        entity_type_mapping,
    )