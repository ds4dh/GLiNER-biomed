import re
import os
import json
import torch
import random
import argparse
from tqdm import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def create_chat(prompt, text):
    return prompt.format(text)

def generate_response(llm, chats, sampling_params):
    responses = llm.generate(chats, sampling_params, use_tqdm=False)
    return responses

def generate_dataset(texts, prompt, llm, sampling_params, process_response = None, batch_size=8, max_lines=None):
    batch_texts = []
    batch_chats = []
    text = True
    final_results = []
    for i in tqdm(range(0, max_lines)):
        text = texts[i]

        batch_texts.append(text)

        chat = prompt.format(text)
        batch_chats.append(chat)

        if len(batch_texts)==batch_size:
            try:
                responses = generate_response(llm, batch_chats, sampling_params)
            except Exception as err:
                print(err)
                continue

            if process_response is not None:
                try:
                    batch_results = process_response(responses, batch_texts)
                except Exception as err:
                    print(err)
                    continue
            else:
                batch_results = [response.outputs[0].text for response in responses]

            final_results.extend(batch_results)

            batch_texts = []
            batch_chats = []

    return final_results

def get_ner_prompt(tokenizer):
    ner_prompt = """Extract entities and their types from the given text and format the output as a list of strings.

    Extracted entities should be precisely spelt as in the text.

    Each string should contain the entity followed by its type, separated by " <> ".

    For example, for a publisher entity "State University of New York Press", the output should be "State University of New York Press <> Publisher".

    You need to extract only named entities.

    Below are more examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Output: ['Harvard University <> Institution']

    Input: Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen.

    Output: ['Microsoft Corporation <> Company', 'Bill Gates <> founder name', 'Paul Allen <> founder name']

    Input: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

    Output: ['Eiffel Tower <> landmark', 'Champ de Mars <> location', 'Paris <> City', 'France <> country']

    Input: According to the 2015 census, it has a population of 70,757 people.

    Output: ['2015 census <> Time', '70,757 people <> Quantity']

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise entity recognition model that extract named entities from arbitrary text given a prompt.",
        },
        {"role": "user", "content": ner_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def process_ner_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        result = response.outputs[0].text

        inner_list_string_ = re.search(pattern, result)

        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = [item for item in items if len(item.split('<>'))==2]
            if len(filtered_items)==len(items):
                batch_results.append({'text': text, 'labels': items})
    return batch_results

def get_hard_ner_prompt(tokenizer):
    ner_prompt = """You need to perform two actions:
    * Generate a prompt for NER model to extract specific entity types;
    * Extract entities and their types from the given text and format the output as a list of strings.

    Extracted entities should be precisely spelt as in the text.

    Each string should contain the entity followed by its type, separated by " <> ".

    For example, for a publisher entity "State University of New York Press", the output should be "State University of New York Press <> Publisher".

    You need to extract only named entities.

    Below are more examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Prompt: Extract organizations that were founded in 1636.

    Output: ['Harvard University <> organization']

    Input: Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen. Apple is a tech company founded by Steve Jobs and Steve Wozniak.

    Prompt: Please, identify the founders of Apple Corporation in the following text:

    Output: ['Steve Jobs <> founder', 'Steve Wozniak <> founder']

    Input: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

    Prompt: Recognize all cities located in France; don't include cities from other countries.

    Output: ['Paris <> City']

    Input: According to the 2015 census, it has a population of 70,757 people.

    Prompt: Extract all years mentioned in the text:

    Output: ['2015 census <> year']

    Please, return only one prompt and one output.

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise named entity recognition model that extract named entities from arbitrary text.",
        },
        {"role": "user", "content": ner_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def process_hard_ner_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        result = response.outputs[0].text
        inner_list_string_ = re.search(pattern, result)

        text = texts[i]
        
        prompt = re.search("Prompt: (.*?)\n", result)

        if not prompt:
            continue
        else:
            prompt = prompt.group(1)
            prompt = re.sub(',"', '"', prompt)

        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = [item for item in items if len(item.split('<>'))==2]
            if len(filtered_items)==len(items):
                batch_results.append({'text': text, 'prompt': prompt, 'labels': items})
    return batch_results


def get_rex_prompt(tokenizer):
    rex_prompt = """Extract relations between named entities from the given text and format the output as a list of strings.

    Extraced entities should be precisely spelled as in the text.

    Each string should contains source entity, relation type and target entity, separated by " <> ".

    You need to extract only relations between named entities.

    Below are examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Output: ['Harvard University <> founded in <> 1636', 'Harvard University <> located in <> United States']

    Input: Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen.

    Output: ['Microsoft Corporation <> was founded by <> Bill Gates', 'Microsoft Corporation <> was founded by <> Paul Allen', 'Paul Allen <> business partner <> Bill Gates']

    Input: The Eiffel Tower is a wrought-iron lattice tower Paris, France.

    Output: ['Eiffel Tower <> location <> Paris, France', 'Eiffel Tower <> city <> Paris', 'Eiffel Tower <> country <> Paris']

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise relation extraction model that extract relations between named entities from arbitrary text.",
        },
        {"role": "user", "content": rex_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def process_rex_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        result = response.outputs[0].text
        inner_list_string_ = re.search(pattern, result)
        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = [item for item in items if len(item.split('<>'))==3]
            batch_results.append({'text': text, 'labels': filtered_items})
    return batch_results


def get_qa_prompt(tokenizer):
    qa_prompt = """You need to perform two actions:
    * Generate a question, answer on which is exactly mentioned in the text;
    * Extract answers to the question and format them in the right way.

    Extracted answers should be precisely spelt as in the text, it can be some entity or sentence or even a whole paragraph;

    Each string should contain a list of answers separated by a comma.

    Below are more examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Question: When was Hardvar University founded?

    Output: ['1636']

    Input: Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen. Apple is a tech company founded by Steve Jobs and Steve Wozniak.

    Question: Who are the founders of Apple Corporation?:

    Output: ['Steve Jobs', 'Steve Wozniak']

    Input: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

    Question: In which city the Eiffel Tower is located?

    Output: ['Paris']

    Please return only one question and one output; don't forget to put square brackets.

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise question-answering model.",
        },
        {"role": "user", "content": qa_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def process_qa_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        result = response.outputs[0].text
        inner_list_string_ = re.search(pattern, result)
        gen_prompt = re.search("Question: (.*?)\n", result)

        if not gen_prompt:
            continue
        else:
            gen_prompt = gen_prompt.group(1)

        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = []
            for item in items:
                if re.search(item, text, re.IGNORECASE):
                    filtered_items.append(item)
            if len(filtered_items)==len(items):
                batch_results.append({'text': text, 'prompt': gen_prompt, 'labels': items})

    return batch_results

def get_open_matching_prompt(tokenizer):
    matching_prompt = """You need to perform two actions:
    * Generate a prompt with ask to find some piece of text that contains required information;
    * Extract answers to the prompt and format them in the right way.

    Extracted answers should be precisely spelt as in the text, it can be some entity or sentence or even a whole paragraph;

    Each string should contain a list of answers separated by a comma.

    Below are more examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Prompt: Please match the description of Harvard University from the following text:

    Output: ['is the oldest institution of higher education in the United States']

    Input: Microsoft Corporation is a technology company founded by Bill Gates and Paul Allen. Apple is a tech company founded by Steve Jobs and Steve Wozniak.

    Prompt: Find a first sentence that describe Apple:

    Output: ['Apple is a tech company founded by Steve Jobs and Steve Wozniak.']

    Input: The Eiffel Tower is a wrought-iron lattice tower colored in blue and yellow on the Champ de Mars in Paris, France.

    Prompt: Please, extract a characteristic of tower mentioned in the text:

    Output: ['wrought-iron lattice tower', 'colored in blue and yellow']

    Please return only one prompt and one output; don't forget to put square brackets.

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise information matching model.",
        },
        {"role": "user", "content": matching_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def process_open_matching_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        result = response.outputs[0].text
        inner_list_string_ = re.search(pattern, result)
        gen_prompt = re.search("Prompt: (.*?)\n", result)
        if not gen_prompt:
            continue
        else:
            gen_prompt = gen_prompt.group(1)
        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = []
            for item in items:
                if re.search(re.escape(item), text, re.IGNORECASE):
                    filtered_items.append(item)
            batch_results.append({'text': text, 'prompt': gen_prompt, 'labels': filtered_items})

    return batch_results


def get_summarization_prompt(tokenizer):
    summarization_prompt = """You need to perform two actions:
    * Generate a prompt with ask to extract sentences that describe and summarize a given text;
    * Extract few sentences that describe text in the best way.

    Extracted sentences must precisely match the text. You can't paraphrase sentences in anyway;

    Each string should contain a list of sentences or parts of sentences separated by a comma.

    Below are some examples:
    Input: Investment arm of world’s most valuable company leads new $35 million investment in AI drug discovery firm.

    Longevity AI drug discovery company Insilico Medicine has completed a $35 million Series D2 round, bringing the total raised in its Series D financing to $95 million. The new round was led by Prosperity7 Ventures, the diversified growth fund of Aramco Ventures, a subsidiary of Aramco, the world’s leading integrated energy and chemicals company.

    Longevity.Technology: We brought you the news of Insilico’s initial $60 million Series D raise in June, and this latest funding injection is further evidence of the huge investor interest now being shown in the longevity sector. The new capital will support the continued advancement of Insilico’s pipeline, including its lead program which is currently in a Phase 1 study in New Zealand and in China, as well as several pipeline programs in the IND-enabling stage, and much more besides.

    Prompt: Please summarize the following text:

    Output: ['Longevity AI drug discovery company Insilico Medicine has completed a $35 million Series D2 round',  'The new round was led by Prosperity7 Ventures, the diversified growth fund of Aramco Ventures', 'The new capital will support the continued advancement of Insilico’s pipeline']

    Input: Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect, while also being the largest individual shareholder until May 2014. Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. Apple went public in 1980 to instant financial success. The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement called "1984". By 1985, the high cost of its products, and power struggles between executives, caused problems. Wozniak stepped back from Apple and pursued other ventures, while Jobs resigned and founded NeXT, taking some Apple employees with him.

    Prompt: Summarize the following text extracting information about Microsoft:

    Output: ['Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800.']

    Please return only one prompt and one output; don't forget to put square brackets.

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise information summarization model.",
        },
        {"role": "user", "content": summarization_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


def jaccard_coefficient_words(str1, str2):
    # Tokenize the strings into words
    set1 = set(word_tokenize(str1))
    set2 = set(word_tokenize(str2))

    # Calculate the intersection and union of the word sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard coefficient
    if not union:  # To handle the case when both sets are empty
        return 1.0 if not intersection else 0.0
    return len(intersection) / len(union)

def match_real_string(sentences, str1, threshold=0.7):
    sentence = None
    for sentence in sentences:
        jc = jaccard_coefficient_words(sentence, str1)
        if jc>=threshold:
            break
    return sentence

def process_summarization_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        sentences = sent_tokenize(text)

        result = response.outputs[0].text
        pattern = r"\['(.*?)'\]"
        inner_list_string_ = re.search(pattern, result)
        gen_prompt = re.search("Prompt: (.*?)\n", result)
        if not gen_prompt:
            continue
        else:
            gen_prompt = gen_prompt.group(1)
        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = []
            for item in items:
                if not re.search(re.escape(item), text):
                    real_item = match_real_string(sentences, item, threshold=0.5)
                else:
                    real_item = item
                if real_item:
                    filtered_items.append(real_item)
            if len(filtered_items):
                batch_results.append({'text': text, 'prompt': gen_prompt, 'labels': filtered_items})

    return batch_results


def get_open_task_prompt(tokenizer):
    matching_prompt = """You need to perform two actions:
    * Generate a prompt that describes an extractive task that token classification model can perform given a text;
    * Extract answers to the prompt and format them in the right way.

    Extracted answers should be precisely spelt as in the text, it can be some entity, key-words, sentence or even a whole paragraph;

    Each string should contain a list of answers separated by a comma.

    Below are more examples:
    Input: Harvard University, founded in 1636, is the oldest institution of higher education in the United States.

    Prompt: Extract the most imporant key-words from the text:

    Output: ['Harvard University', 'institution', 'higher education', 'the United States']

    Input: Bill Gates and Paul Allen founded Microsoft Corporation, a leading technology company.

    Prompt: Find all verb phrases in the text:

    Output: ['founded']

    Input: The Eiffel Tower is a wrought-iron lattice tower colored in blue and yellow on the Champ de Mars in Paris, France.

    Prompt: Please, extract a characteristic of tower mentioned in the text:

    Output: ['wrought-iron lattice tower', 'colored in blue and yellow']

    Input: The Eiffel Tower is a wrought-iron lattice tower colored in blue and yellow on the Champ de Mars in Paris, France that attracts many tourists.

    Prompt: Classify a text, given the following classes: traveling, business, sport, politics, France:

    Output: ['traveling', 'France']

    Please return only one prompt and one output; don't forget to put square brackets.

    Try to generate diverse set of tasks and prompts, including classification and extraction tasks,

    You need to generate only output based on the input text below:
    {}
    """

    messages = [
        {
            "role": "system",
            "content": "You are accurate and precise information matching model.",
        },
        {"role": "user", "content": matching_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt

def process_open_task_responses(responses, texts):
    pattern = re.compile(r"\['(.*?)'\]")
    batch_results = []

    for i, response in enumerate(responses):
        text = texts[i]
        result = response.outputs[0].text
        inner_list_string_ = re.search(pattern, result)
        gen_prompt = re.search("Prompt: (.*?)\n", result)
        if not gen_prompt:
            continue
        else:
            gen_prompt = gen_prompt.group(1)
        if inner_list_string_:
            inner_list_string = inner_list_string_.group(1)
            items = list(map(lambda x: x.strip("'") , inner_list_string.split("', ")))
            filtered_items = []
            for item in items:
                if re.search(re.escape(item), text, re.IGNORECASE):
                    filtered_items.append(item)
            batch_results.append({'text': gen_prompt+text, 'labels': filtered_items})

    return batch_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "data/raw_texts.json")
    parser.add_argument('--save_path', type=str, default= "data/ner")
    parser.add_argument('--model', type=str, default= "NousResearch/Hermes-2-Pro-Llama-3-8B")
    parser.add_argument('--quantization', type=str, default= "fp8")
    parser.add_argument('--max_examples', type=int, default= 100000)
    parser.add_argument('--batch_size', type=int, default= 16)
    parser.add_argument('--temperature', type=float, default= 0.25)
    parser.add_argument('--task', type=str, default= "ner")
    args = parser.parse_args()

    with open(args.data_path, 'r') as f:
        texts = json.load(f)
        random.shuffle(texts)
        print('Texts count: ', len(texts))

     
    llm = LLM(model=args.model,
                    max_model_len = 12000, 
                    tensor_parallel_size=1, dtype="half", #kv_cache_dtype="fp8", 
                                        gpu_memory_utilization = 0.9, quantization = args.quantization)

    sampling_params = SamplingParams(temperature = args.temperature, repetition_penalty = 1.1, top_k=100, max_tokens=512, top_p=0.8, stop="<end>")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.task == 'ner':
        ner_prompt = get_ner_prompt(tokenizer)

        results = generate_dataset(texts, ner_prompt, llm, sampling_params,
                                process_response = process_ner_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)
    elif args.task == 'hard_ner':
        ner_prompt = get_hard_ner_prompt(tokenizer)

        results = generate_dataset(texts, ner_prompt, llm, sampling_params,
                                process_response = process_hard_ner_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)

    elif args.task == 'rex':
        rex_prompt = get_rex_prompt(tokenizer)

        results = generate_dataset(texts, rex_prompt, llm, sampling_params,
                                process_response = process_rex_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)

    elif args.task == 'qa':
        qa_prompt = get_qa_prompt(tokenizer)

        results = generate_dataset(texts, qa_prompt, llm, sampling_params,
                                process_response = process_qa_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)

    elif args.task == 'open_matching':
        open_matching_prompt = get_open_matching_prompt(tokenizer)

        results = generate_dataset(texts, open_matching_prompt, llm, sampling_params,
                                process_response = process_open_matching_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)

    elif args.task == 'summarization':
        summarization_prompt = get_open_matching_prompt(tokenizer)

        results = generate_dataset(texts, summarization_prompt, llm, sampling_params,
                                        process_response = process_summarization_responses,
                                                    batch_size=args.batch_size, max_lines=args.max_examples)

    elif args.task == 'open_task':
        open_task_prompt = get_open_task_prompt(tokenizer)

        results = generate_dataset(texts, open_task_prompt, llm, sampling_params,
                                process_response = process_open_task_responses,
                                  batch_size=args.batch_size, max_lines=args.max_examples)

    save_path = os.path.join(args.save_path, f"raw_{args.task}.json")
    
    with open(save_path, 'w') as f:
        json.dump(results, f)