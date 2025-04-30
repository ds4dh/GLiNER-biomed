import os
import json
import spacy
from datasets import load_dataset
from tqdm.auto import tqdm
import random

random.seed(42)
nlp = spacy.load("en_core_web_sm")


def collect_biored_entities(dataset):
    """Collect all entity types across all splits"""
    entity_types = set()
    for split in dataset.keys():
        for example in dataset[split]:
            for ent in example["entities"]:
                entity_types.add(ent["type"])
    return entity_types


def process_biored_example(example, entity_type_mapping, all_entity_types):
    text = " ".join([p["text"][0] for p in example["passages"]])

    # SpaCy tokenization
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Process entities
    entities = []
    present_types = set()
    for ent in example["entities"]:
        ent_type = entity_type_mapping[ent["type"]]
        span = doc.char_span(
            ent["offsets"][0][0],
            ent["offsets"][0][1],
            alignment_mode="expand",
            label=ent_type,
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


def process_biored():
    # Load dataset
    dataset = load_dataset("bigbio/biored", "biored_bigbio_kb")

    # Official entity type mapping
    entity_type_mapping = {
        "SequenceVariant": "Sequence variant",
        "GeneOrGeneProduct": "Gene or gene product",
        "CellLine": "Cell line",
        "ChemicalEntity": "Chemical entity",
        "DiseaseOrPhenotypicFeature": "Disease or phenotype",
        "OrganismTaxon": "Organism",
    }

    # Collect all entity types
    all_entity_types = collect_biored_entities(dataset)

    # Process splits
    output_dir = "./data/biored"
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        processed = []
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            entry = process_biored_example(
                example, entity_type_mapping, all_entity_types
            )
            processed.append(entry)

        # Save split
        if split == "validation":
            save_name = f"val.json"
        else:
            save_name = f"{split}.json"
        output_path = os.path.join(output_dir, save_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_biored()