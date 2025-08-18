import argparse
import json
import os
import re
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Set

from OblConfig import OblConfig
from utils import extract_noun_chunks
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor,
    build_vllm_token_enforcer_tokenizer_data,
)
from pydantic import BaseModel, Field
import copy

import spacy

nlp = spacy.load('en_core_web_sm')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the JSON Schema using Pydantic
class EntitySchema(BaseModel):
    id: int
    text: str
    type: str


class RelationSchema(BaseModel):
    head: int
    tail: int
    type: str


class JSONSchema(BaseModel):
    entities: List[EntitySchema]
    relations: List[RelationSchema]


def create_generation_prompt_base(
    text: str, one_shot_messages: List[Dict[str, str]], tokenizer
) -> str:
    """
    Creates a prompt for the LLM by appending the user text to the one-shot messages.

    Args:
        text (str): The input text to process.
        one_shot_messages (List[Dict[str, str]]): A list of messages for the prompt.
        tokenizer: The tokenizer used to apply the chat template.

    Returns:
        str: The generated prompt.
    """
    print('Text tokens:', len(text)//4)
    messages = copy.deepcopy(one_shot_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                f'Here is a text input: "{text}" '
                "Analyze this text, identify the entities, and extract their relationships as per your instructions."
            ),
        }
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def create_generation_prompt_spacy(
    text: str, one_shot_messages: List[Dict[str, str]], tokenizer
) -> str:
    """
    Creates a prompt for the LLM by appending the user text to the one-shot messages.

    Args:
        text (str): The input text to process.
        one_shot_messages (List[Dict[str, str]]): A list of messages for the prompt.
        tokenizer: The tokenizer used to apply the chat template.

    Returns:
        str: The generated prompt.
    """
    noun_chunks = extract_noun_chunks(nlp, text)
    messages = copy.deepcopy(one_shot_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                f'Here is a text input: "{text}" '
                f"""Here is the list of input entities: {noun_chunks}\n"""
                "Analyze this text, select and classify the entities, and extract their relationships as per your instructions."
            ),
        }
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

def load_texts(data_path: str, limit: int = None) -> List[str]:
    """
    Loads texts from a JSON file.

    Args:
        data_path (str): The path to the JSON file containing texts.
        limit (int, optional): The maximum number of texts to load.

    Returns:
        List[str]: A list of texts.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return []

    if '.csv' in data_path:
        texts = pd.read_csv(data_path)['text'].fillna('')
        texts = [text for text in texts if text]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
    if limit:
        texts = texts[:limit]
    return texts


def save_batch_results(
    results: List[Dict[str, Any]], batch_start: int, batch_end: int, output_dir: str
):
    """
    Saves the batch results to a JSON file.

    Args:
        results (List[Dict[str, Any]]): The results to save.
        batch_start (int): The starting index of the batch.
        batch_end (int): The ending index of the batch.
        output_dir (str): The directory where the output files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"batch_{batch_start}_{batch_end}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Batch results saved to {output_path}")


def initialize_model_and_tokenizer(obl_config: OblConfig) -> Tuple[LLM, Any]:
    """
    Initializes the LLM model and tokenizer.

    Args:
        obl_config (OblConfig): The configuration object.

    Returns:
        Tuple[LLM, Any]: The initialized LLM and tokenizer.
    """
    model_name = obl_config.model_name
    tokenizer_name = obl_config.tokenizer_name
    max_model_len = obl_config.max_model_len
    tensor_parallel_size = obl_config.tensor_parallel_size
    chat_template = obl_config.chat_template
    dtype = obl_config.dtype
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.chat_template = chat_template

    # Initialize the vllm LLM
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype
    )

    return llm, tokenizer


def prepare_logits_processor(llm: LLM) -> Any:
    """
    Prepares the logits processor for the LLM.

    Args:
        llm (LLM): The initialized LLM.

    Returns:
        Any: The logits processor.
    """
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
    json_parser = JsonSchemaParser(JSONSchema.schema())
    logits_processor = build_vllm_logits_processor(tokenizer_data, json_parser)
    return logits_processor


def define_sampling_params(
    obl_config: OblConfig, logits_processor: Any
) -> SamplingParams:
    """
    Defines the sampling parameters for the LLM generation.

    Args:
        obl_config (OblConfig): The configuration object.
        logits_processor (Any): The logits processor.

    Returns:
        SamplingParams: The sampling parameters.
    """
    return SamplingParams(
        n=1,
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=42,
        max_tokens=obl_config.max_gen_tokens,
        logits_processors=[logits_processor],
    )


def generate_batch_prompts(
    batch_texts: List[str], one_shot_messages: List[Dict[str, str]], tokenizer, task_type: str
) -> List[str]:
    """
    Generates prompts for a batch of texts.

    Args:
        batch_texts (List[str]): The texts in the current batch.
        one_shot_messages (List[Dict[str, str]]): One-shot messages for prompt creation.
        tokenizer: The tokenizer used to apply the chat template.

    Returns:
        List[str]: The list of generated prompts.
    """
    batch_prompts = []
    for text in batch_texts:
        if task_type=='base':
            prompt = create_generation_prompt_base(text, one_shot_messages, tokenizer)
        else:
            prompt = create_generation_prompt_spacy(text, one_shot_messages, tokenizer)
        batch_prompts.append(prompt)
    return batch_prompts


def process_batch_outputs(
    outputs, batch_texts: List[str], batch_indices: List[int]
) -> List[Dict[str, Any]]:
    """
    Processes the outputs from the LLM for a batch of texts.

    Args:
        outputs: The outputs from the LLM generation.
        batch_texts (List[str]): The texts in the current batch.
        batch_indices (List[int]): The indices of the texts in the current batch.

    Returns:
        List[Dict[str, Any]]: The processed results for the batch.
    """
    batch_results = []
    for idx, (output, text) in zip(batch_indices, zip(outputs, batch_texts)):
        try:
            result_text = output.outputs[0].text.strip()
            print(result_text)
            parsed_json = json.loads(result_text)
            batch_results.append(
                {
                    "index": idx,
                    "input_text": text,
                    "generated_json": parsed_json,
                }
            )
            # logger.info(f"Processed text at index {idx}")
        except Exception as e:
            logger.error(f"Error processing output at index {idx}: {e}")
            continue
    return batch_results


def get_processed_indices(output_dir: str) -> Set[int]:
    """
    Retrieves the set of indices that have already been processed.

    Args:
        output_dir (str): The directory where the output files are saved.

    Returns:
        Set[int]: A set of processed indices.
    """
    processed_indices = set()
    if not os.path.exists(output_dir):
        return processed_indices

    batch_files = [
        f for f in os.listdir(output_dir) if re.match(r"batch_\d+_\d+\.json", f)
    ]
    for batch_file in batch_files:
        match = re.match(r"batch_(\d+)_(\d+)\.json", batch_file)
        if match:
            start_idx = int(match.group(1))
            end_idx = int(match.group(2))
            indices = range(start_idx, end_idx + 1)
            processed_indices.update(indices)
    return processed_indices


def main():
    """
    Main function to execute the batch processing.
    """
    # Set up argparse to accept batch_size, data_path, output_dir, and limit
    parser = argparse.ArgumentParser(description="Batch processing script")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing (overrides config value if provided)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/bio_texts_balanced.json",
        help="Path to the data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/batches",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of texts to process",
    )
    args = parser.parse_args()

    # Load configuration
    obl_config = OblConfig()

    # Use batch_size from argparse
    batch_size = args.batch_size

    # Use data_path, output_dir, and limit from argparse
    data_path = args.data_path
    output_dir = args.output_dir
    limit = args.limit

    # Load texts with limit
    texts = load_texts(data_path, limit=limit)
    total_texts = len(texts)

    if not texts:
        logger.warning("No texts to process.")
        return

    # Get processed indices
    processed_indices = get_processed_indices(output_dir)
    logger.info(f"Found {len(processed_indices)} processed indices.")

    # Build list of unprocessed indices
    all_indices = set(range(total_texts))
    unprocessed_indices = sorted(all_indices - processed_indices)

    if not unprocessed_indices:
        logger.info("All texts have been processed.")
        return

    # Initialize model and tokenizer
    llm, tokenizer = initialize_model_and_tokenizer(obl_config)

    # Prepare logits processor and sampling parameters
    logits_processor = prepare_logits_processor(llm)
    generation_args = define_sampling_params(obl_config, logits_processor)

    one_shot_messages = obl_config.one_shot_messages

    task_type = obl_config.task_type

    # Process texts in batches
    num_unprocessed = len(unprocessed_indices)
    logger.info(f"Processing {num_unprocessed} unprocessed texts.")

    for batch_start_idx in range(0, num_unprocessed, batch_size):
        batch_indices = unprocessed_indices[
            batch_start_idx : batch_start_idx + batch_size
        ]
        batch_texts = [texts[idx] for idx in batch_indices]

        # Generate prompts for the current batch
        try:
            batch_prompts = generate_batch_prompts(
                batch_texts, one_shot_messages, tokenizer, task_type
            )
        except Exception as e:
            logger.error(
                f"Error generating prompts for batch starting at index {batch_indices[0]}: {e}"
            )
            continue

        # Batch generation
        logger.info(
            f"Generating outputs for batch indices {batch_indices[0]} to {batch_indices[-1]}..."
        )
        try:
            outputs = llm.generate(batch_prompts, generation_args)
            # print(outputs)
        except Exception as e:
            logger.error(
                f"Error during generation for batch indices {batch_indices[0]} to {batch_indices[-1]}: {e}"
            )
            continue

        # Process outputs for the current batch
        batch_results = process_batch_outputs(outputs, batch_texts, batch_indices)

        # Save batch results
        batch_start = batch_indices[0]
        batch_end = batch_indices[-1]
        save_batch_results(batch_results, batch_start, batch_end, output_dir)

    logger.info("Processing completed.")


if __name__ == "__main__":
    main()