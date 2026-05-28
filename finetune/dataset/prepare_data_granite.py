#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "pydantic>=2.0",
#     "jinja2",
#     "datasets",
#     "huggingface_hub>=0.20.0",
# ]
# ///
"""Prepare QMD query-expansion data for Granite-3.3-2B-Instruct SFT.

Mirrors dataset/prepare_data.py (loads all data/*.jsonl through the strict
Pydantic schema, dedups by query, splits, writes a "text"+"messages" dataset)
but with two Granite-specific changes:

  1. NO `/no_think` directive in the user prompt — that is a Qwen3 control token,
     meaningless to Granite (it would leak into the prompt and drive drift). The
     chat template is Granite's own, applied via its tokenizer.
  2. No Qwen `<think>` tag stripping.

Adds a `--push <repo>` step (which prepare_data.py lacks) because the HF Jobs
entrypoint (jobs/sft_granite.py) loads the training set from the Hub, not local
files.

Run from the finetune/ directory:
    uv run dataset/prepare_data_granite.py                       # local only (verify)
    uv run dataset/prepare_data_granite.py --push <user>/qmd-query-expansion-granite-train
"""

import argparse
import glob as globmod
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # make finetune/ importable
from dataset.schema import TrainingExample, load_examples, output_items_to_text

from transformers import AutoTokenizer

GRANITE_BASE = os.environ.get("QMD_BASE_MODEL", "ibm-granite/granite-3.3-2b-instruct")

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(GRANITE_BASE)
    return _tokenizer


def format_for_training(ex: TrainingExample) -> dict:
    """Format one validated example for Granite SFT (no /no_think)."""
    tokenizer = get_tokenizer()
    output_text = output_items_to_text(ex.output)

    user_prompt = f"Expand this search query: {ex.query}"
    if ex.intent:
        user_prompt = (
            f"Expand this search query: {ex.query}\n"
            f"Query intent: {ex.intent.strip()}"
        )

    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {
        "query": ex.query,
        "output": ex.output_as_lists(),
        "text": text,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Granite SFT")
    parser.add_argument("--input", default="data/*.jsonl", help="Input JSONL glob")
    parser.add_argument("--output", default="data/train-granite", help="Output dir")
    parser.add_argument("--split", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push", default=None,
                        help="HF dataset repo to push to (e.g. user/qmd-query-expansion-granite-train)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Push as a private dataset (default)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(globmod.glob(args.input)) if "*" in args.input else [args.input]
    if not input_files:
        print(f"Error: no files match {args.input}")
        raise SystemExit(1)
    print(f"Found {len(input_files)} input files (tokenizer: {GRANITE_BASE})")

    all_examples: list[TrainingExample] = []
    for f in input_files:
        examples = load_examples(f)
        print(f"  {Path(f).name}: {len(examples)} examples")
        all_examples.extend(examples)
    print(f"Loaded {len(all_examples)} examples total")

    # Deduplicate by query (case-insensitive)
    seen: set[str] = set()
    deduped: list[TrainingExample] = []
    for ex in all_examples:
        key = ex.query.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    if len(deduped) < len(all_examples):
        print(f"Deduplicated: {len(all_examples)} -> {len(deduped)}")
    all_examples = deduped

    random.seed(args.seed)
    random.shuffle(all_examples)

    formatted = [format_for_training(ex) for ex in all_examples]
    split_idx = int(len(formatted) * (1 - args.split))
    train_data, val_data = formatted[:split_idx], formatted[split_idx:]

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with open(output_dir / name, "w") as fh:
            for item in data:
                fh.write(json.dumps(item) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total: {len(all_examples)}  Train: {len(train_data)}  Val: {len(val_data)}")
    print(f"Output: {output_dir}")
    print(f"\nSample formatted text (verify Granite template, NO /no_think):\n")
    print(train_data[0]["text"][:600])

    if args.push:
        from datasets import Dataset, DatasetDict
        cols = ["query", "output", "text", "messages"]
        dd = DatasetDict({
            "train": Dataset.from_list([{k: r[k] for k in cols} for r in train_data]),
            "validation": Dataset.from_list([{k: r[k] for k in cols} for r in val_data]),
        })
        print(f"\nPushing to hub: {args.push} (private={args.private})")
        dd.push_to_hub(args.push, private=args.private)
        print(f"Pushed. Set jobs/sft_granite.py DATASET (or QMD_TRAIN_DATASET) to {args.push}")


if __name__ == "__main__":
    main()
