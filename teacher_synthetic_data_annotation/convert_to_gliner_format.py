import spacy
import json
import re
import os
import argparse
import logging

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def tokenize(text):
    tokens = [tok.text for tok in nlp.make_doc(text)]
    return tokens

def tokenize_and_match(text):
    doc = nlp.make_doc(text)
    tokens = []
    s2t = {}
    e2t = {}
    for t, tok in enumerate(doc):
        tokens.append(tok.text)
        s = tok.idx
        s2t[s] = t
        e = tok.idx + len(tok.text)
        e2t[e] = t
    return tokens, s2t, e2t

def search_span(text, label, s2t, e2t):
    sts = []
    ets = []
    try:
        searches = re.finditer(re.escape(label), text, re.IGNORECASE)
    except Exception as e:
        logger.error(f"Error while searching span: {e}")
        return sts, ets
    for search in searches:
        s, e = search.span()
        if s in s2t and e in e2t:
            st = s2t[s]
            et = e2t[e]
            sts.append(st)
            ets.append(et)
    return sts, ets

def collect_labels(text, span, label, s2t, e2t):
    labels = []
    sts, ets = search_span(text, span, s2t, e2t)
    for st, et in zip(sts, ets):
        labels.append([st, et, label])
    return labels

def process_for_entity_recognition(text, item):
    tokens, s2t, e2t = tokenize_and_match(text)
    ner = []

    entities = item['generated_json']['entities']
    for ent in entities:
        ent_text = ent['text']
        ent_type = ent['type']

        ent_labels = collect_labels(text, ent_text, ent_type, s2t, e2t)
        ner.extend(ent_labels)
    return {"tokenized_text": tokens, 'ner': ner}

def process_for_relation_extraction(text, item):
    tokens, s2t, e2t = tokenize_and_match(text)
    ner = []

    entities = item['generated_json']['entities']
    relations = item['generated_json']['relations']

    positive_pairs = set()
    unique_relations = set()
    for rel in relations:
        head_id = rel['head']
        tail_id = rel['tail']
        
        rel_type = rel['type']
        unique_relations.add(rel_type)
        
        if head_id < len(entities):
            head = entities[head_id]['text']
        else:
            continue
        if tail_id < len(entities):
            tail = entities[tail_id]['text']
        else:
            continue

        label = f"{head} > {rel_type}"
        positive_pairs.add(label)

        ent_labels = collect_labels(text, tail, label, s2t, e2t)
        ner.extend(ent_labels)

    negatives = []
    for ent in entities:
        ent_text = ent['text']
        for rel in unique_relations:
            label = f"{ent_text} > {rel}"
            if label not in positive_pairs:
                negatives.append(label)
                
    return {"tokenized_text": tokens, 'ner': ner, "negatives": negatives}

def process_batches(data_path, save_result, processing_function):
    logger.info(f"Processing data from: {data_path}")

    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        return

    batches = [os.path.join(data_path, batch) for batch in os.listdir(data_path) if batch.endswith('.json')]

    dataset = []
    for batch in batches:
        logger.info(f"Processing batch: {batch}")
        with open(batch, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item['input_text']
                result = processing_function(text, item)
                dataset.append(result)

    logger.info(f"Saving results to: {save_result}")
    with open(save_result, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=1)

    logger.info("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Tokenize text and collect NER labels or relation extraction data from batches.")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to the directory containing batch files in JSON format."
    )
    parser.add_argument(
        '--save_result',
        type=str,
        required=True,
        help="Path to save the output JSON file."
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['ner', 'relation_extraction'],
        required=True,
        help="Task to perform: 'ner' for named entity recognition or 'relation_extraction' for relation extraction."
    )

    args = parser.parse_args()

    if args.task == 'ner':
        processing_function = process_for_entity_recognition
    elif args.task == 'relation_extraction':
        processing_function = process_for_relation_extraction

    process_batches(args.data_path, args.save_result, processing_function)

if __name__ == "__main__":
    main()
