import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

HF_TOKEN = "hf_"
local_base_meta_merged_lora = "models/Llama-3.2-3B-lora-pubmed-qa"

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
    token=HF_TOKEN,
).to(device)

base_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=llama_base_model,
    token=HF_TOKEN,
)

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
    "json", data_files="../data/ft_pubmedqa.jsonl", split="train"
)

dataset_llama_format = dataset_raw.map(
    formatting_prompts_func,
    batched=True,
)

messages = dataset_llama_format[49]["messages"]
print(json.dumps(messages, indent=2))

# training args for model fine tuning
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=3,
    output_dir="./output",
    save_strategy="epoch",
    learning_rate=2e-5,
    bf16=True, # according to your GPU config
    logging_steps=1,
    optim="adam",
    weight_decay=0.01,
    lr_scheduler_type="linear"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    use_rslora=True,
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset_llama_format,
    args=training_args,
    peft_config=peft_config,
)

# show current GPU stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# trigger finetuning
trainer_stats = trainer.train()

# push lora adapter to huggingface
trainer.push_to_hub(
    token=HF_TOKEN,
    commit_message="Pushing LoRA adapter to Hugging Face Hub",
    blocking=True,
)

# Merge LoRA with base model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
#
# model = AutoModelForCausalLM.from_pretrained(llama_base_model)
# model = PeftModel.from_pretrained(model, hf_base_meta_lora)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(local_base_meta_merged_lora)
#
# merged_model.push_to_hub(
#     hf_base_meta_merged_lora,
#     token=HF_TOKEN,
#     commit_message="Pushing merged model to Hugging Face Hub",
#     blocking=True,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(llama_base_model)
#
# tokenizer.push_to_hub(
#     hf_base_meta_merged_lora,
#     token=HF_TOKEN,
#     commit_message="Pushing tokenizer to Hugging Face Hub",
#     blocking=True,
# )

# load LoRA adapter with base model
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
#
# base_model = AutoModelForCausalLM.from_pretrained(llama_base_model)
# peft_model_id = hf_base_meta_lora
# model = PeftModel.from_pretrained(base_model, peft_model_id).to(device)
#
# tokenizer = AutoTokenizer.from_pretrained(llama_base_model)
#
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
# if model.config.pad_token_id is None:
#     model.config.pad_token_id = model.config.eos_token_id
