import os
import json
from transformers import DebertaV2TokenizerFast
from tqdm import tqdm

def collect_all_entity_types(input_folder):
    """
    Collect all unique entity types from the 'ner' key across all splits.
    """
    entity_types = set()
    for split in ['train', 'val', 'test']:
        input_file = os.path.join(input_folder, f'{split}.json')
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for instance in data:
            ner_annotations = instance.get('ner', [])
            for annotation in ner_annotations:
                _, _, entity_type = annotation
                entity_types.add(entity_type)
    return entity_types

def process_instance(instance, tokenizer, max_length=2000, all_entity_types=None):
    """
    Process a single instance by tokenizing the tokens, splitting into chunks without splitting entities,
    adjusting word-level annotations, and recomputing negatives for each chunk.
    """
    tokens = instance['tokenized_text']  # List of words
    ner_annotations = instance.get('ner', [])
    total_words = len(tokens)

    # Tokenize the entire list of words
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    word_ids = encoding.word_ids()  # List of word indices per token

    # Build mapping from words to number of tokens
    word_to_num_tokens = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            word_to_num_tokens[word_idx] = word_to_num_tokens.get(word_idx, 0) + 1

    # Build a mapping from word index to entities starting or ending at that word
    word_to_entities = {}
    for idx, (start_idx, end_idx, entity_type) in enumerate(ner_annotations):
        for word_idx in range(start_idx, end_idx + 1):
            word_to_entities.setdefault(word_idx, set()).add(idx)

    # Now, create chunks without splitting entities
    chunks = []
    current_chunk_words = []
    current_chunk_word_indices = []
    current_chunk_token_count = 0
    current_word_idx = 0

    while current_word_idx < total_words:
        num_tokens_in_word = word_to_num_tokens.get(current_word_idx, 0)
        entity_indices_in_word = word_to_entities.get(current_word_idx, set())

        # Check if adding this word would exceed max_length
        if current_chunk_token_count + num_tokens_in_word > max_length:
            if entity_indices_in_word:
                # If the word is part of an entity, we need to handle it carefully
                # Check if any entity starts at this word
                entities_starting_here = [
                    idx for idx in entity_indices_in_word
                    if ner_annotations[idx][0] == current_word_idx
                ]
                if entities_starting_here:
                    # Start a new chunk to avoid splitting the entity
                    if current_chunk_words:
                        # Process current chunk
                        chunk = create_chunk(
                            current_chunk_words,
                            ner_annotations,
                            current_chunk_word_indices,
                            all_entity_types
                        )
                        chunks.append(chunk)
                        # Start a new chunk
                        current_chunk_words = []
                        current_chunk_word_indices = []
                        current_chunk_token_count = 0
                    else:
                        # The entity is too long to fit in one chunk
                        # Decide how to handle this case
                        pass
                else:
                    # Continue adding to current chunk to finish the entity
                    pass
            else:
                # Word is not part of an entity; start a new chunk
                if current_chunk_words:
                    # Process current chunk
                    chunk = create_chunk(
                        current_chunk_words,
                        ner_annotations,
                        current_chunk_word_indices,
                        all_entity_types
                    )
                    chunks.append(chunk)
                    # Start a new chunk
                    current_chunk_words = []
                    current_chunk_word_indices = []
                    current_chunk_token_count = 0

        # Add word to current chunk
        current_chunk_words.append(tokens[current_word_idx])
        current_chunk_word_indices.append(current_word_idx)
        current_chunk_token_count += num_tokens_in_word
        current_word_idx += 1

    # Process any remaining words in the last chunk
    if current_chunk_words:
        chunk = create_chunk(
            current_chunk_words,
            ner_annotations,
            current_chunk_word_indices,
            all_entity_types
        )
        chunks.append(chunk)

    return chunks

def create_chunk(words, ner_annotations, chunk_word_indices, all_entity_types):
    """
    Create a chunk with adjusted 'ner' annotations and recomputed 'negatives'.
    """
    # Adjust 'ner' annotations for this chunk
    chunk_ner = []
    entity_types_in_chunk = set()
    chunk_start_word_idx = min(chunk_word_indices)
    chunk_end_word_idx = max(chunk_word_indices)
    for annotation in ner_annotations:
        start_idx, end_idx, entity_type = annotation
        # Check if the entity is fully within the chunk
        if start_idx >= chunk_start_word_idx and end_idx <= chunk_end_word_idx:
            adjusted_start = start_idx - chunk_start_word_idx
            adjusted_end = end_idx - chunk_start_word_idx
            chunk_ner.append([adjusted_start, adjusted_end, entity_type])
            #print(entity_type, " : ", words[adjusted_start:adjusted_end+1])
            entity_types_in_chunk.add(entity_type)
        else:
            # Entity is not fully within the chunk; exclude it
            pass
    # Recompute negatives
    negatives_in_chunk = list(all_entity_types - entity_types_in_chunk)
    # Create the chunk data
    chunk_data = {
        'tokenized_text': words.copy(),  # List of words
        'ner': chunk_ner,
        'negatives': negatives_in_chunk,
    }
    if not negatives_in_chunk:
        del chunk_data["negatives"]

    return chunk_data

def process_data(input_folder, output_folder, max_length=2000):
    """
    Process JSON files in the input_folder, split instances into chunks with a maximum token length,
    adjust annotations, and save to output_folder.
    """
    tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v3-large")

    # Collect all unique entity types from 'ner' annotations in all splits
    all_entity_types = collect_all_entity_types(input_folder)
    print(f"Collected {len(all_entity_types)} unique entity types: {all_entity_types}\n")

    for split in ['train', 'val', 'test']:
        input_file = os.path.join(input_folder, f'{split}.json')
        output_file = os.path.join(output_folder, f'{split}.json')

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for instance in tqdm(data, desc=f'Processing {split} data'):
            chunks = process_instance(instance, tokenizer, max_length, all_entity_types=all_entity_types)
            processed_data.extend(chunks)

        # Save the processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {split} data to {output_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process JSON files and split instances into chunks.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing the JSON files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save the processed JSON files.')
    parser.add_argument('--max_length', type=int, default=2000, help='Maximum token length for each chunk.')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_data(args.input_folder, args.output_folder, max_length=args.max_length)