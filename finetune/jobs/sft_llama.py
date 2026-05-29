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
SFT training for QMD query expansion — Llama-3.2-3B-Instruct.

Llama variant of jobs/sft_granite.py (pai-source qmd-expansion-eval, #144 bake-off).
Differences from the Granite job:
  - BASE_MODEL = unsloth/Llama-3.2-3B-Instruct. The canonical
    meta-llama/Llama-3.2-3B-Instruct is GATED (403 for this HF token); the unsloth
    mirror is ungated with identical weights/architecture. License: Llama 3.2
    Community License (naming clause + "Built with Llama") — recorded as info.
  - SELF-CONTAINED: the dead eval_common.py download (tobil URL now 404s) and the
    post-train run_eval block are dropped — the real eval is expand_eval.ts on the
    converted GGUF (the production serving path). Nothing is fetched at runtime.

Dataset reuse is intentional and verified safe: the shared
<user>/qmd-query-expansion-granite-train carries a clean `messages` column. TRL's
SFTTrainer detects the conversational format and tokenizes from `messages`, applying
*Llama's own* chat template — the pre-rendered Granite `text` field is ignored. So no
Granite tokens leak in, and no Llama-specific data prep is needed. Llama has its own
chat template, so there is NO custom-template plumbing (unlike Granite base).

LoRA target_modules and hyperparameters are byte-identical to the Qwen/Granite recipe
(Llama-arch module names), per the issue's "zero LoC recipe transfer cost" finding.

Run on HF Jobs (A10G has 24 GB — runs the stock plain-LoRA recipe unchanged):
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h --detach jobs/sft_llama.py
"""

import os
from huggingface_hub import login, whoami

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Resolve repos: explicit env override, else <user>/<name> from the token identity.
# DATASET defaults to the SHARED granite-train set (schema-identical; SFTTrainer applies
# Llama's template to the clean `messages` column — see module docstring).
_user = os.environ.get("HF_NAMESPACE") or whoami()["name"]
DATASET = os.environ.get("QMD_TRAIN_DATASET", f"{_user}/qmd-query-expansion-granite-train")
OUTPUT_MODEL = os.environ.get("QMD_OUTPUT_MODEL", f"{_user}/qmd-query-expansion-llama-3b-sft")
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
    output_dir="qmd-query-expansion-llama-3b-sft",
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

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,

    report_to="none",
)

# LoRA: rank 16, all projection layers (Llama-arch module names — identical to Qwen/Granite).
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

print("Pushing to Hub...")
trainer.push_to_hub()
print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")
print("Next: convert_gguf.py --base", BASE_MODEL, "--sft", OUTPUT_MODEL,
      "--output", f"{_user}/qmd-query-expansion-llama-3b-gguf")
