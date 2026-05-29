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
GRPO training for QMD query expansion — Granite-3.3-2B-Instruct.

Granite variant of experiments/grpo/grpo.py (pai-source qmd-expansion-eval, #135).
Refines the instruct SFT checkpoint with reward.py (via eval_common.QMDRewardFunction)
as the RL reward — targeting the runaway-repetition / Diversity collapse that caps the
instruct-SFT at ~85.9% on normal queries (see docs analysis-granite-cluster-diagnosis).

Differences from the Qwen recipe:
  - BASE_MODEL = ibm-granite/granite-3.3-2b-instruct (has its own chat template, so the
    apply_chat_template prompt path and qmd serving are unchanged — no template plumbing).
  - SFT_MODEL / OUTPUT_MODEL / DATASET resolve from env (QMD_SFT_MODEL / QMD_OUTPUT_MODEL
    / QMD_TRAIN_DATASET) or the logged-in user's namespace.
  - GRPO recipe (LoRA r=4 q/v, beta=0.04 KL, lr=5e-7, 200 steps) is kept IDENTICAL to the
    proven Qwen run for comparability; only model/data/repo change.
Granite-3.3 is Llama-architecture, so the GRPO LoRA target_modules are unchanged.

Run on HF Jobs (GRPO = generate + train, heavier than SFT):
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 3h \
        experiments/grpo/grpo_granite.py
"""

import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login, whoami
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Download eval_common.py if running as a standalone script (e.g. HF Jobs)
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
from eval_common import QMDRewardFunction, run_eval  # noqa: E402

BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    _user = os.environ.get("HF_NAMESPACE") or whoami()["name"]
    sft_model = os.environ.get("QMD_SFT_MODEL", f"{_user}/qmd-query-expansion-granite-2b-sft")
    output_model = os.environ.get("QMD_OUTPUT_MODEL", f"{_user}/qmd-query-expansion-granite-2b-grpo")
    dataset_repo = os.environ.get("QMD_TRAIN_DATASET", f"{_user}/qmd-query-expansion-granite-train")
    print(f"Base: {BASE_MODEL}\nSFT: {sft_model}\nDataset: {dataset_repo}\nOutput: {output_model}")

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset — Granite's own chat template (matches SFT prompt regime).
    print(f"Loading dataset: {dataset_repo}...")
    dataset = load_dataset(dataset_repo, split="train")

    def extract_prompt(example):
        content = example["messages"][0]["content"]
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base, merge the instruct SFT adapter — GRPO refines on top of SFT weights.
    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Merging SFT adapter {sft_model}...")
    model = PeftModel.from_pretrained(base_model, sft_model)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Fresh LoRA for GRPO (small: rank 4, q/v only) — identical to the Qwen recipe.
    grpo_lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, grpo_lora)
    model.print_trainable_parameters()

    config = GRPOConfig(
        output_dir="qmd-query-expansion-granite-2b-grpo",
        push_to_hub=True,
        hub_model_id=output_model,

        num_generations=4,
        max_completion_length=200,
        beta=0.04,  # KL regularization — prevents drift from the SFT checkpoint

        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        max_grad_norm=0.5,
        max_steps=200,

        logging_steps=10,
        save_strategy="epoch",
        bf16=True,

        report_to="none",
    )

    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[QMDRewardFunction()],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()
    print(f"Done! Model: https://huggingface.co/{output_model}")

    # --- Automatic evaluation (reward.py via eval_common) ---
    print("\nStarting automatic evaluation...")
    trainer.model.eval()
    run_eval(trainer.model, tokenizer, "grpo")


if __name__ == "__main__":
    main()
