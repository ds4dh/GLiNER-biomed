import os
import json
import spacy
from datasets import load_dataset
from tqdm.auto import tqdm
import random

# Set seed for reproducibility
random.seed(42)
nlp = spacy.load("en_core_web_sm")


def collect_chia_entities(dataset):
    """Collect all entity types across all splits with original casing"""
    entity_types = set()
    for split in dataset.keys():
        for example in dataset[split]:
            for ent in example["entities"]:
                entity_types.add(ent["type"])
    return entity_types


def process_chia_example(example, entity_type_mapping, all_entity_types):
    # Concatenate text from all passages
    text = " ".join([p["text"][0] for p in example["passages"]])

    # SpaCy tokenization
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Process entities
    entities = []
    present_types = set()
    for ent in example["entities"]:
        ent_type = entity_type_mapping.get(ent["type"], ent["type"])
        for offset in ent["offsets"]:
            span = doc.char_span(
                offset[0], offset[1], alignment_mode="expand", label=ent_type
            )
            if span is not None:
                entities.append(
                    [span.start, span.end - 1, ent_type]  # Convert to inclusive end
                )
                present_types.add(ent_type)

    # Build data entry
    data_entry = {"tokenized_text": tokens, "ner": entities}

    # Add negatives if any types are missing
    negatives = list(all_entity_types - present_types)
    if negatives:
        data_entry["negatives"] = negatives

    return data_entry


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split data into train, validation, and test sets based on given ratios."""
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def process_chia():
    # Load dataset
    dataset = load_dataset("bigbio/chia", "chia_bigbio_kb")

    # Entity type mapping (customize as needed)
    entity_type_mapping = {
        "Qualifier": "Qualifier",
        "Temporal": "Temporal",
        "Drug": "Drug",
        "Negation": "Negation",
        "Multiplier": "Multiplier",
        "Condition": "Condition",
        "Mood": "Mood",
        "Measurement": "Measurement",
        "Procedure": "Procedure",
        "Device": "Device",
        "Scope": "Scope",
        "Value": "Value",
        "Observation": "Observation",
        "Reference_point": "Reference point",
        "Visit": "Visit",
        "Person": "Person",
    }

    # Collect all entity types
    all_entity_types = collect_chia_entities(dataset)
    all_entity_types = {entity_type_mapping.get(et, et) for et in all_entity_types}

    # Process dataset
    processed_data = []
    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue

        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            entry = process_chia_example(example, entity_type_mapping, all_entity_types)
            processed_data.append(entry)

    # Split into train/val/test (80/10/10)
    train_data, val_data, test_data = split_dataset(processed_data, 0.8, 0.1, seed=42)

    # Output directory
    output_dir = "./data/chia"
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    for name, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_chia()