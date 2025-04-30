import os
import random
import json
import spacy
from tqdm.auto import tqdm


def collect_entity_types(path_to_ann, entity_type_mapping):
    """
    Collect all unique entity types from the annotation files in the dataset.
    """
    all_entity_types = set()

    # Iterate over all annotation files
    for file_name in os.listdir(path_to_ann):
        if file_name.endswith(".ann"):
            ann_file_path = os.path.join(path_to_ann, file_name)

            with open(ann_file_path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("T"):
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            entity_info = parts[1]
                            info_parts = entity_info.split()
                            entity_type = info_parts[0]

                            # Rename entity type if in the mapping
                            entity_type = entity_type_mapping.get(
                                entity_type, entity_type
                            )

                            # Add entity type to the set of all entity types
                            all_entity_types.add(entity_type)

    print(f"Collected {len(all_entity_types)} unique entity types.")
    return all_entity_types


def process_files(
    files,
    path_to_ann,
    path_to_txt,
    output_file,
    split_type,
    consider_discontinuous,
    entity_type_mapping,
    all_entity_types,
):
    """
    Process annotation and text files to prepare data for NER tasks. This function allows renaming entity types
    and prints statistics on entity inclusion/exclusion based on discontinuity handling.
    """
    nlp = spacy.load("en_core_web_sm")

    data = []
    excluded_discontinuous_count = 0  # Counter for excluded entities
    total_entities = 0  # Counter for total entities processed

    for file_name in tqdm(files, desc=f"Processing {split_type} set"):
        ann_file_path = os.path.join(path_to_ann, file_name + ".ann")
        txt_file_path = os.path.join(path_to_txt, file_name + ".txt")

        with open(txt_file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Read annotations
        entities = []
        file_entity_types = set()  # Track entity types in this file
        with open(ann_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("T"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        entity_id = parts[0]
                        entity_info = parts[1]
                        entity_text = parts[2]
                        info_parts = entity_info.split()
                        entity_type = info_parts[0]

                        # Rename entity type if in the mapping
                        entity_type = entity_type_mapping.get(entity_type, entity_type)

                        # Add entity type to file-specific set
                        file_entity_types.add(entity_type)

                        # Handle possibly discontinuous spans
                        span_str = " ".join(info_parts[1:])
                        span_parts = span_str.split(";")
                        spans = []
                        for span_part in span_parts:
                            start_end = span_part.strip().split()
                            if len(start_end) != 2:
                                continue
                            start_char, end_char = map(int, start_end)
                            spans.append((start_char, end_char))

                        # Update entity counts
                        total_entities += len(spans)

                        # Exclude or include discontinuous entities based on setting
                        if len(spans) > 1:
                            if not consider_discontinuous[split_type]:
                                # Skip discontinuous spans
                                excluded_discontinuous_count += 1
                                continue
                            else:
                                # Handle discontinuous spans as separate entities
                                for start_char, end_char in spans:
                                    entities.append(
                                        {
                                            "start_char": start_char,
                                            "end_char": end_char,
                                            "type": entity_type,
                                        }
                                    )
                        else:
                            start_char, end_char = spans[0]
                            entities.append(
                                {
                                    "start_char": start_char,
                                    "end_char": end_char,
                                    "type": entity_type,
                                }
                            )

        # Tokenize text with spaCy
        doc = nlp(text)
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
                token_spans.append([start_token_index, end_token_index, entity_type])
            else:
                print(
                    f"Warning: could not create token span for entity in file {file_name} at chars {start_char}-{end_char}"
                )

        # Identify absent entity types in this file
        negative_entity_types = list(all_entity_types - file_entity_types)

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


def split_data(
    path_to_ann,
    path_to_txt,
    output_base,
    train_split,
    val_split,
    test_split,
    seed,
    consider_discontinuous,
    entity_type_mapping,
):
    """
    Split the data into training, validation, and testing sets, and process each set with the option to rename
    entity types and handle discontinuous entities based on settings.
    """
    random.seed(seed)

    # Collect common files from annotation and text folders
    ann_files = {
        os.path.splitext(file)[0]
        for file in os.listdir(path_to_ann)
        if file.endswith(".ann")
    }
    txt_files = {
        os.path.splitext(file)[0]
        for file in os.listdir(path_to_txt)
        if file.endswith(".txt")
    }
    common_files = list(ann_files.intersection(txt_files))
    random.shuffle(common_files)

    # Split files
    total = len(common_files)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)

    train_files = common_files[:train_end]
    val_files = common_files[train_end:val_end]
    test_files = common_files[val_end:]

    # Collect all entity types before processing splits
    all_entity_types = collect_entity_types(path_to_ann, entity_type_mapping)
    print(all_entity_types, "\n")

    # Process each split
    process_files(
        train_files,
        path_to_ann,
        path_to_txt,
        os.path.join(output_base, "train.json"),
        "train",
        consider_discontinuous,
        entity_type_mapping,
        all_entity_types,
    )
    process_files(
        val_files,
        path_to_ann,
        path_to_txt,
        os.path.join(output_base, "val.json"),
        "val",
        consider_discontinuous,
        entity_type_mapping,
        all_entity_types,
    )
    process_files(
        test_files,
        path_to_ann,
        path_to_txt,
        os.path.join(output_base, "test.json"),
        "test",
        consider_discontinuous,
        entity_type_mapping,
        all_entity_types,
    )


if __name__ == "__main__":

    split_data(
        path_to_ann="./raw_data/cadec/original",
        path_to_txt="./raw_data/cadec/text",
        output_base="./data/cadec",
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        seed=42,
        consider_discontinuous={"train": False, "val": False, "test": False},
        entity_type_mapping={
            "Disease": "Disease",
            "ADR": "Adverse drug event",
            "Drug": "Drug",
            "Symptom": "Symptom",
            "Finding": "Finding",
        },
    )