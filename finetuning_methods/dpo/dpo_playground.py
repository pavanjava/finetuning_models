import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

# Monkey-patch Accelerate to disable the buggy conversion
import accelerate.utils.operations as ops

def patched_convert(tensor, *args, **kwargs):
    return tensor  # Don't convert, return as-is

ops.convert_to_fp16 = patched_convert
ops._convert_to_fp16 = patched_convert

# NOW import everything else
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dataset = load_dataset('inclusionAI/Ling-Coder-DPO')
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
)

ft_model_name = model_name.split('/')[1].replace('Instruct', 'DPO')

training_args = DPOConfig(
    num_train_epochs=3,
    output_dir=f"{ft_model_name}",
    logging_steps=25,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    optim="adamw_torch_fused",
    save_strategy='epoch',
    report_to="none",
    fp16=True
)

trainer = DPOTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset['train'],
    ref_model=None,
)

print(f"Memory before training: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

trainer.train()