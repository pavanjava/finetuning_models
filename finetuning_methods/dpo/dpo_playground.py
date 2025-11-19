import os

# ✓ Let Accelerate handle mixed precision (bf16 on Ampere+ is stable)
os.environ.pop("ACCELERATE_MIXED_PRECISION", None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dataset = load_dataset("damerajee/hindi-dpo")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# causal LM prefers left padding for batched generation-style loss
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Policy model: bf16 (or fp16 if your GPU lacks bf16)
policy = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
policy.config.use_cache = False
policy.gradient_checkpointing_enable()  # big memory saver

# Reference model: frozen & 8-bit to avoid doubling VRAM
# (requires bitsandbytes installed)
ref = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)

ft_model_name = model_name.split("/")[1].replace("Instruct", "DPO")

training_args = DPOConfig(
    output_dir=f"{ft_model_name}",
    num_train_epochs=100,
    logging_steps=25,
    # ↓ keep per-device batch tiny; use accumulation for effective batch size
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective batch ≈16
    learning_rate=1e-4,
    optim="adamw_torch_fused",
    save_strategy="epoch",
    report_to="none",
    # use bf16 when available; otherwise set fp16=True
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    # TRL/DPO-specific helpful caps
    max_length=1024, # was 1024 → 2x speedup
    max_prompt_length=512, # was 512 → 2x speedup
    # truncation_mode="keep_end",
    # precompute_ref_log_probs=True,
    # precompute_ref_batch_size=4,
    # reference_free=True
)

trainer = DPOTrainer(
    model=policy,
    # ref_model=ref,                   # <— crucial: prevent hidden deep copy [if uncommented]
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    args=training_args,
)

print(f"Memory before training: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
trainer.train()
