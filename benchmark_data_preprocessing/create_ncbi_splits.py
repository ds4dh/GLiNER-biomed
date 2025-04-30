import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

def convert_tags_to_spans(ner_tags):
    """Convert BIO NER tags to entity spans in target format."""
    spans = []
    current_start = None
    for idx, tag in enumerate(ner_tags):
        if tag == 1:  # B-Disease
            if current_start is not None:
                spans.append((current_start, idx-1))
            current_start = idx
        elif tag == 2:  # I-Disease
            if current_start is None:
                current_start = idx  # Handle edge case (invalid but possible)
        else:  # O
            if current_start is not None:
                spans.append((current_start, idx-1))
                current_start = None
    # Handle entity at end of sequence
    if current_start is not None:
        spans.append((current_start, len(ner_tags)-1))
    # Format as [start, end, type]
    return [[start, end, "Disease"] for (start, end) in spans]

def process_split(dataset_split, split_name, output_dir):
    """Process a single split and save to JSON."""
    processed_data = []
    all_entity_types = {"Disease"}  # Only entity type in NCBI Disease
    
    for example in tqdm(dataset_split, desc=f"Processing {split_name}"):
        # Convert tags to spans
        ner_spans = convert_tags_to_spans(example["ner_tags"])
        
        # Build data entry
        data_entry = {
            "tokenized_text": example["tokens"],
            "ner": ner_spans
        }
        
        # Add negatives if no entities
        if not ner_spans:
            data_entry["negatives"] = list(all_entity_types)
        
        processed_data.append(data_entry)
    
    # Save to JSON
    output_path = os.path.join(output_dir, f"{split_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

def main():
    # Configuration
    output_dir = "./data/ncbi_disease"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset("ncbi_disease")

    # Process splits
    split_mapping = {"validation": "val", "test": "test"}  # Rename validation→val
    for orig_split in ["train", "validation", "test"]:
        target_split = split_mapping.get(orig_split, orig_split)
        process_split(dataset[orig_split], target_split, output_dir)

if __name__ == "__main__":
    main()