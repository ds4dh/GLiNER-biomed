from messages import (FEW_SHOT_WITH_MULTI_MESSAGES, FEW_SHOT_WITH_SINGLE_SYSTEM_MESSAGE, FEW_SHOT_WITH_MULTI_MESSAGES_AND_ENTITIES)

class OblConfig:
    model_name = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer_name = "aaditya/Llama3-OpenBioLLM-8B"
    max_model_len = 4096
    tensor_parallel_size = 1
    max_gen_tokens = 2048

    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|end_of_text|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    one_shot_messages = FEW_SHOT_WITH_MULTI_MESSAGES_AND_ENTITIES
    
    task_type = "spacy" #spacy

    dtype = "half"

    