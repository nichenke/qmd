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
SFT training for QMD query expansion — Granite-3.3-2B-Instruct.

Granite variant of jobs/sft.py (pai-source qmd-expansion-eval, #135). Differences:
  - BASE_MODEL = ibm-granite/granite-3.3-2b-instruct (Apache-2.0 adoption candidate).
  - DATASET = the Granite-templated set from dataset/prepare_data_granite.py (NO
    /no_think); defaults to <your-hf-namespace>/qmd-query-expansion-granite-train.
  - OUTPUT_MODEL / DATASET resolve from env (QMD_OUTPUT_MODEL / QMD_TRAIN_DATASET)
    or fall back to the logged-in user's namespace via whoami().
Granite-3.3 is Llama-architecture, so the LoRA target_modules are unchanged.

Run on HF Jobs (A10G has 24 GB — runs the stock plain-LoRA recipe unchanged):
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft_granite.py
"""

import os
import sys
from huggingface_hub import login, whoami

BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Resolve repos: explicit env override, else <user>/<name> from the token identity.
_user = os.environ.get("HF_NAMESPACE") or whoami()["name"]
DATASET = os.environ.get("QMD_TRAIN_DATASET", f"{_user}/qmd-query-expansion-granite-train")
OUTPUT_MODEL = os.environ.get("QMD_OUTPUT_MODEL", f"{_user}/qmd-query-expansion-granite-2b-sft")
print(f"Base: {BASE_MODEL}\nDataset: {DATASET}\nOutput: {OUTPUT_MODEL}")

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

print(f"Loading dataset: {DATASET}...")
dataset = load_dataset(DATASET, split="train")
print(f"Dataset loaded: {len(dataset)} examples")

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

config = SFTConfig(
    output_dir="qmd-query-expansion-granite-2b-sft",
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

# LoRA: rank 16, all projection layers (Granite-3.3 = Llama-arch module names).
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

# --- Automatic evaluation (qmd's reward.py via eval_common) ---
_eval_common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_common.py")
if not os.path.exists(_eval_common_path):
    import urllib.request
    _url = "https://huggingface.co/datasets/tobil/hf-cli-jobs-uv-run-scripts/resolve/main/eval_common.py"
    _opener = urllib.request.build_opener()
    _token = os.environ.get("HF_TOKEN", "")
    if _token:
        _opener.addheaders = [("Authorization", f"Bearer {_token}")]
    with open(_eval_common_path, "wb") as _f:
        _f.write(_opener.open(_url).read())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import run_eval

print("\nStarting automatic evaluation...")
eval_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if eval_tokenizer.pad_token is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
trainer.model.eval()
run_eval(trainer.model, eval_tokenizer, "sft")
