import os
import spacy
import json
import glob
import random
from tqdm.auto import tqdm

random.seed(42)

nlp = spacy.load("en_core_web_sm")


def collect_entity_types(ann_files, entity_type_mapping):
    """
    Collect all unique entity types from annotation files after applying the entity type mapping.
    """
    all_entity_types = set()
    for ann_file in ann_files:
        with open(ann_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("T"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        entity_info = parts[1]
                        info_parts = entity_info.split()
                        entity_type = info_parts[0]
                        # Apply entity type mapping
                        entity_type = entity_type_mapping.get(entity_type, entity_type)
                        all_entity_types.add(entity_type)
    return all_entity_types


def parse_ann(ann_lines, entity_type_mapping):
    """
    Parse annotation lines and extract entities, treating discontinuous entities as continuous
    by taking the minimum start and maximum end offsets.
    """
    entities = []
    for line in ann_lines:
        line = line.strip()
        if line.startswith("T"):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                entity_id = parts[0]
                entity_info = parts[1]
                entity_text = parts[2]
                info_parts = entity_info.split()
                entity_type = info_parts[0]
                # Apply entity type mapping
                entity_type = entity_type_mapping.get(entity_type, entity_type)

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

                if spans:
                    # Take the minimum start and maximum end to represent the entity as continuous
                    start_char = min(start for start, end in spans)
                    end_char = max(end for start, end in spans)
                    entities.append({'start': start_char, 'end': end_char, 'type': entity_type})
    return entities


def process_files_no_save(
    txt_file_path,
    ann_file_path,
    all_entity_types,
    entity_type_mapping,
):
    """
    Process individual text and annotation files, mapping entities to token indices
    and computing negative entity types.
    """
    with open(txt_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(ann_file_path, "r", encoding="utf-8") as f:
        ann_lines = f.readlines()
    # Parse annotations
    entities = parse_ann(ann_lines, entity_type_mapping)

    # Tokenize text
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Map character offsets to token indices using doc.char_span
    token_spans = []
    entity_types_in_doc = set()
    for entity in entities:
        start_char = entity["start"]
        end_char = entity["end"]
        entity_type = entity["type"]
        span = doc.char_span(start_char, end_char, alignment_mode="expand")
        if span is not None:
            start_token_index = span.start
            end_token_index = span.end - 1  # end index is inclusive
            token_spans.append([start_token_index, end_token_index, entity_type])
            # Collect entity types in this document
            entity_types_in_doc.add(entity_type)
        else:
            print(
                f"Warning: could not create token span for entity in file {os.path.basename(txt_file_path)} at chars {start_char}-{end_char}"
            )

    # Compute negatives
    negatives = list(all_entity_types - entity_types_in_doc)

    # Prepare data entry
    assert negatives or token_spans
    data = {"tokenized_text": tokens}
    data["ner"] = token_spans  # Ensure 'ner' key exists (empty list if no entities)
    if negatives:
        data["negatives"] = negatives

    return data


def process_and_save(
    input_path, output_path, entity_type_mapping
):
    """
    Process all files in the dataset, handle splits, and save the processed data.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Collect all .ann files across all splits to collect all entity types
    ann_files = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(input_path, split)
        ann_files.extend(glob.glob(os.path.join(split_path, "*.ann")))

    # Collect all entity types
    all_entity_types = collect_entity_types(ann_files, entity_type_mapping)
    print(
        f"Collected {len(all_entity_types)} unique entity types: {all_entity_types}\n"
    )

    for split in ["train", "val", "test"]:
        all_instances = []
        split_path = os.path.join(input_path, split)
        txt_files = glob.glob(os.path.join(split_path, "*.txt"))

        # Process each file
        for txt_file in tqdm(txt_files, desc=f"Processing {split} files"):
            ann_file = txt_file.replace(".txt", ".ann")
            data_entry = process_files_no_save(
                txt_file,
                ann_file,
                all_entity_types,
                entity_type_mapping,
            )
            all_instances.append(data_entry)

        # Save to JSON
        output_file = os.path.join(output_path, f"{split}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                all_instances, f, ensure_ascii=False, indent=2
            )

        print(f"Saved {split} data to {output_file}")


if __name__ == "__main__":
    input_path = "./raw_data/n2c2"
    output_path = "./data/n2c2"
    entity_type_mapping = {
        'Form': 'Drug form',
        'ADE': 'Adverse drug event',
        'Duration': 'Treatment duration',
        'Dosage': 'Drug dosage',
        'Frequency': 'Frequency of drug administration',
        'Drug': 'Drug',
        'Route': 'Drug administration route',
        'Reason': 'Reason for drug prescription',
        'Strength': 'Drug strength'
    }
    process_and_save(
        input_path, output_path, entity_type_mapping
    )