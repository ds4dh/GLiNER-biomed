import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

def convert_tags_to_spans(ner_tags):
    """Convert BIO tags to entity spans with type awareness"""
    spans = []
    current_start = None
    current_type = None
    
    for idx, tag in enumerate(ner_tags):
        # Determine entity type from tag
        if tag in [1, 4]:  # B/I-Chemical
            entity_type = "Chemical"
        elif tag in [2, 3]:  # B/I-Disease
            entity_type = "Disease"
        else:
            entity_type = None

        if entity_type:
            if tag in [1, 2]:  # Start of new entity
                if current_start is not None:
                    spans.append((current_start, idx-1, current_type))
                current_start = idx
                current_type = entity_type
            else:  # Continuation of entity
                if current_type != entity_type:  # Type mismatch
                    if current_start is not None:
                        spans.append((current_start, idx-1, current_type))
                    current_start = idx
                    current_type = entity_type
        else:  # O tag
            if current_start is not None:
                spans.append((current_start, idx-1, current_type))
                current_start = None
                current_type = None
    
    # Add final entity if exists
    if current_start is not None:
        spans.append((current_start, len(ner_tags)-1, current_type))
    
    return [[start, end, ent_type] for (start, end, ent_type) in spans]

def process_split(dataset_split, split_name, output_dir):
    """Process dataset split and save to JSON format"""
    processed_data = []
    all_entity_types = {"Chemical", "Disease"}
    
    for example in tqdm(dataset_split, desc=f"Processing {split_name}"):
        # Convert tags to spans
        ner_spans = convert_tags_to_spans(example["tags"])
        
        # Determine present entity types
        present_types = {span[2] for span in ner_spans}
        negatives = list(all_entity_types - present_types)
        
        # Build data entry
        data_entry = {
            "tokenized_text": example["tokens"],
            "ner": ner_spans
        }
        
        # Add negatives if applicable
        if negatives:
            data_entry["negatives"] = negatives
        
        # Validate entry has either entities or negatives
        assert ner_spans or negatives, "Empty document with no negatives"
        
        processed_data.append(data_entry)
    
    # Save to JSON
    output_path = os.path.join(output_dir, f"{split_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

def main():
    # Configuration
    output_dir = "./data/bc5cdr"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset("tner/bc5cdr")

    # Process splits with proper naming
    split_mapping = {"validation": "val", "test": "test"}
    for orig_split in ["train", "validation", "test"]:
        target_split = split_mapping.get(orig_split, orig_split)
        process_split(dataset[orig_split], target_split, output_dir)

if __name__ == "__main__":
    main()