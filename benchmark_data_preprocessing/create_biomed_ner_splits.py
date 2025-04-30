import os
import json
import spacy
from datasets import load_dataset
from tqdm.auto import tqdm
import random

random.seed(42)
nlp = spacy.load("en_core_web_sm")

ENTITY_MAPPING =  {
    # BioMed_general_NER Classes (24 keys as defined in the dataset card)
    "CHEMICALS": "Chemical",
    "CLINICAL DRUG": "Drug",
    "BODY SUBSTANCE": "Body substance",
    "ANATOMICAL STRUCTURE": "Anatomical structure",
    "CELLS AND THEIR COMPONENTS": "Cell or cell component",
    "GENE AND GENE PRODUCTS": "Gene or gene product",
    "INTELLECTUAL PROPERTY": "Intellectual property",
    "LANGUAGE": "Language",
    "REGULATION OR LAW": "Regulation or law",
    "GEOGRAPHICAL AREAS": "Geographical area",
    "ORGANISM": "Organism",
    "GROUP": "Group",
    "PERSON": "Person",
    "ORGANIZATION": "Organization",
    "PRODUCT": "Product",
    "LOCATION": "Location",
    "PHENOTYPE": "Phenotype",
    "DISORDER": "Disorder",
    "SIGNALING MOLECULES": "Signaling molecule",
    "EVENT": "Event",
    "MEDICAL PROCEDURE": "Medical procedure",
    "ACTIVITY": "Activity",
    "FUNCTION": "Function",
    "MONEY": "Money",
    # Extra variants (not defined in the dataset card but included in the dataset)
    "PRODUCTS": "Product",             # variant of PRODUCT
    "GENES": "Gene or gene product",                   # variant of GENE AND GENE PRODUCTS
    "Unlabelled": "Unlabelled",        # not part of the YAML classes
    "ORGANIZATIONS": "Organization",   # plural variant of ORGANIZATION
    "EVENTS": "Event",                 # plural variant of EVENT
    "ORGANISMS": "Organism",           # plural variant of ORGANISM
    "ORGANISMS ": "Organism",          # same as above (with trailing space)
    "INTELLECTUAL": "Intellectual property",    # partial form (vs. INTELLECTUAL PROPERTY)
    "LOCATIONS": "Location",           # plural variant of LOCATION
    "DISORDERS": "Disorder",           # plural variant of DISORDER
    "FINDINGS/PHENOTYPES": "Finding"  # slash-delimited variant
}


def collect_biomed_entities(dataset):
    """Collect all entity types across all splits, excluding 'Unlabelled'."""
    entity_types = set()
    for split in dataset.keys():
        for example in dataset[split]:
            for ent in example["entities"]:
                entity_class = ent["class"]
                normalized = ENTITY_MAPPING.get(entity_class, entity_class)
                if normalized != "Unlabelled":  # Ignore "Unlabelled" entities
                    entity_types.add(normalized)
    return entity_types

def process_biomed_example(example, all_entity_types):
    text = example["text"]
    doc = nlp(text)
    tokens = [token.text for token in doc]
    
    entities = []
    present_types = set()
    
    # Iterate over entities directly; no nested "spans" key
    for ent in example["entities"]:
        orig_type = ent["class"]
        entity_type = ENTITY_MAPPING.get(orig_type, orig_type)
        if entity_type == "Unlabelled":  # Ignore "Unlabelled" entities
            continue
        
        # Directly use "start" and "end" from the entity dictionary
        start_char = ent["start"]
        end_char = ent["end"]
        
        span = doc.char_span(
            start_char,
            end_char,
            label=entity_type,
            alignment_mode="expand"
        )
        
        if span is not None:
            entities.append([
                span.start,
                span.end - 1,  # Inclusive end
                entity_type
            ])
            present_types.add(entity_type)
        else:
            #import ipdb; ipdb.set_trace()
            #print(f"Could not align entity: {ent}")
            pass
    
    # Build data entry
    data_entry = {
        "tokenized_text": tokens,
        "ner": entities
    }
    
    # Add negatives if any types are missing
    negatives = list(all_entity_types - present_types)
    if negatives:
        data_entry["negatives"] = negatives
        
    return data_entry

def process_biomed_ner():
    dataset = load_dataset("knowledgator/biomed_NER")
    all_entity_types = collect_biomed_entities(dataset)
    
    processed = []
    for example in tqdm(dataset["train"], desc="Processing train"):
        try:
            entry = process_biomed_example(example, all_entity_types)
            processed.append(entry)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    random.shuffle(processed)
    
    total_samples = len(processed)
    train_end = int(0.8 * total_samples)
    val_end = train_end + int(0.1 * total_samples)
    
    train_data = processed[:train_end]
    val_data = processed[train_end:val_end]
    test_data = processed[val_end:]

    output_dir = "./data/biomed_ner"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_biomed_ner()