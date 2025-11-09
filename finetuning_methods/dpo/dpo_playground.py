from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load Dataset and Model
dataset = load_dataset('inclusionAI/Ling-Coder-DPO')
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # set padding token

# generate sample code with base model
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


def format_chat_prompt(user_input):
    """
    Formats user input into the chat template format with <|im_start|> and <|im_end|> tags.

    Args:
        user_input (str): The input text from the user.

    Returns:
        str: Formatted prompt for the model.
    """

    # Format user message
    user_prompt = f"<|im_start|>user:\n{user_input}<|im_end|>\n"

    # Start assistant's turn
    assistant_prompt = "<|im_start|>assistant:\n"

    # Combine prompts
    formatted_prompt = user_prompt + assistant_prompt

    return formatted_prompt


prompt = format_chat_prompt(str(dataset['train']['prompt'][0]))
print(prompt)

output = generator(prompt, max_length=512, max_new_tokens=512, truncation=True,
                   num_return_sequences=1, temperature=0.8)
print(output)

ft_model_name = model_name.split('/')[1].replace('Instruct', 'DPO')

training_args = DPOConfig(
    num_train_epochs=3,
    output_dir=f"{ft_model_name}",
    logging_steps=25,
    per_device_train_batch_size=8,
    # per_device_eval_batch_size=8, # enable if your dataset has eval split
    # load_best_model_at_end=True, # requires the save and eval strategy to match
    # metric_for_best_model='eval_loss',
    save_strategy='epoch',
    # eval_strategy='epoch', # enable if your dataset has eval split
    # eval_steps=1,
    report_to="none",  # Add this line not log into any infra
)

print(f'training args set: {training_args}')

trainer = DPOTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset['train'],
)

trainer.train()
