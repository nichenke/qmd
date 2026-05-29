# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "datasets",
#     "bitsandbytes",
#     "torch",
# ]
# ///
"""
Granite-3.3-2B SFT — best-checkpoint variant (cell B of the 4-way eval).

Identical recipe to jobs/sft_granite.py EXCEPT load_best_model_at_end=True +
metric_for_best_model=eval_loss, so the pushed adapter is the lowest-eval_loss
checkpoint rather than the (overfit) final epoch. The epoch-5 run's eval_loss
rose after ~epoch 1.3 (0.434 -> 0.535); this fixes that.

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft_granite_best.py
"""

import os
from huggingface_hub import login, whoami

BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

_user = os.environ.get("HF_NAMESPACE") or whoami()["name"]
DATASET = os.environ.get("QMD_TRAIN_DATASET", f"{_user}/qmd-query-expansion-granite-train")
OUTPUT_MODEL = os.environ.get("QMD_OUTPUT_MODEL", f"{_user}/qmd-query-expansion-granite-2b-sft-best")
print(f"Base: {BASE_MODEL}\nDataset: {DATASET}\nOutput: {OUTPUT_MODEL}")

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

print(f"Loading dataset: {DATASET}...")
dataset = load_dataset(DATASET, split="train")
print(f"Dataset loaded: {len(dataset)} examples")

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

config = SFTConfig(
    output_dir="qmd-query-expansion-granite-2b-sft-best",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_strategy="every_save",

    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,

    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,

    # Checkpoint selection — push the best-generalizing checkpoint, not the last.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,

    report_to="none",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print("Initializing SFT trainer...")
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
)

print("Starting SFT training...")
trainer.train()

print("Pushing best checkpoint to Hub...")
trainer.push_to_hub()
print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")
