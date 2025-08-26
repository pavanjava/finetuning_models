import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Llama-3.2-3B-Instruct
llama_base_model = "meta-llama/Llama-3.2-3B-Instruct"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )

base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=llama_base_model,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    # quantization_config=quantization_config,
    token="hf_"
).to(device)

base_tokenizer = AutoTokenizer.from_pretrained(base_model, token="hf_")

"""
We now use the Llama-3.1 format for conversation style fine tuning. 
But we convert it to HuggingFace's normal multiturn format ("role", "content"). 
Llama-3 renders multi turn conversations like below:

<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hey there! How Can i assist you?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'm great thanks!<|eot_id|>
"""


def formatting_prompts_func(examples):
    """
    Formats each chat conversation in examples["messages"] into a prompt string
    using the tokenizer's chat template. Returns a dict with the formatted texts.
    """
    messages = examples["messages"]
    texts = [
        base_tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        for message in messages
    ]
    return {
        "text": texts,
    }

dataset_raw = load_dataset(
    "json", data_files="data/blackwell_architecture.jsonl", split="train"
)

dataset_llama_format = dataset_raw.map(
    formatting_prompts_func,
    batched=True,
)
