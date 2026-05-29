# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "peft>=0.7.0",
#     "torch",
#     "huggingface_hub>=0.20.0",
#     "accelerate",
# ]
# ///
"""
Evaluate QMD query expansion models on HuggingFace Jobs.

Self-contained script — inlines the reward function and test queries.

    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 30m jobs/eval.py
    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 30m jobs/eval.py -- --sft-only
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
GRPO_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"

# --- Test queries (inlined from evals/queries.txt) ---
QUERIES = [
    # Technical documentation
    "how to configure authentication",
    "typescript async await",
    "docker compose networking",
    "git rebase vs merge",
    "react useEffect cleanup",
    # Short/ambiguous
    "auth",
    "config",
    "setup",
    "api",
    # Named entities
    "who is TDS motorsports",
    "React hooks tutorial",
    "Docker container networking",
    "Kubernetes pod deployment",
    "AWS Lambda functions",
    # Personal notes / journals
    "meeting notes project kickoff",
    "ideas for new feature",
    "todo list app architecture",
    # Research / learning
    "what is dependency injection",
    "difference between sql and nosql",
    "kubernetes vs docker swarm",
    # Error/debugging
    "connection timeout error",
    "memory leak debugging",
    "cors error fix",
    # Temporal / recency
    "recent news about Shopify",
    "latest AI developments",
    "best laptops right now",
    "what changed in kubernetes latest version",
    # Complex
    "how to implement caching with redis in nodejs",
    "best practices for api rate limiting",
    "setting up ci cd pipeline with github actions",
]

# =============================================================================
# Reward function (inlined from reward.py)
# =============================================================================

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in',
    'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
})

KEY_TERM_STOPWORDS = frozenset({
    'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
    'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
    'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell',
})

GENERIC_LEX_PHRASES = frozenset({
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with',
})

CHAT_TEMPLATE_TOKENS = frozenset({
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\nassistant\n', '\nuser\n',
})


def parse_expansion(text):
    result = {"lex": [], "vec": [], "hyde": [], "invalid": []}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())
        else:
            result["invalid"].append(line)
    return result


def clean_model_output(text):
    text = text.replace('<|im_end|>', '').strip()
    used_thinking = '<think>' in text and '</think>' in text
    if used_thinking:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return text, used_thinking


def extract_named_entities(query):
    entities = set()
    words = query.split()
    prev_was_entity = False
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue
        is_entity = False
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower()); is_entity = True
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower()); is_entity = True
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower()); is_entity = True
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower()); is_entity = True
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower()); is_entity = True
        prev_was_entity = is_entity
    return entities


def get_key_terms(query):
    return set(query.lower().split()) - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line, query):
    key_terms = get_key_terms(query)
    return not key_terms or bool(key_terms & set(lex_line.lower().split()))


def lex_preserves_entities(line, entities):
    if not entities: return True
    return any(e in line.lower() for e in entities)


def lex_is_generic(lex_line):
    lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lower or lower.startswith(phrase.split()[0]):
            remaining = lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


def word_set_distance(a, b):
    return len(set(a.lower().split()) ^ set(b.lower().split()))


def is_diverse(a, b, min_distance=2):
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a: return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion, query):
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text):
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


def score_expansion_detailed(query, expansion):
    text, used_thinking = clean_model_output(expansion.strip())
    deductions = []

    def _fail(reason):
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "think_bonus": 0, "total": 0, "max_possible": 100,
            "percentage": 0.0, "rating": "Failed", "deductions": [reason],
        }

    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return _fail("CHAT TEMPLATE LEAKAGE")
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return _fail(f"INVALID LINE: {line[:50]}")

    parsed = parse_expansion(text)

    format_score = 10
    if parsed["lex"]: format_score += 10
    else: deductions.append("missing lex:")
    if parsed["vec"]: format_score += 10
    else: deductions.append("missing vec:")

    diversity_score = 0
    types_present = sum(1 for t in ("lex", "vec") if parsed[t])
    if types_present >= 2: diversity_score += 10
    if len(parsed["lex"]) + len(parsed["vec"]) >= 2: diversity_score += 5
    lex_div = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, 2): lex_div -= 2
    diversity_score += max(0, lex_div)
    vec_div = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, 3): vec_div -= 2
    diversity_score += max(0, vec_div)
    echo = 5
    for exp in parsed["lex"] + parsed["vec"]:
        if echoes_query(exp, query): echo -= 3
    diversity_score += max(0, echo)

    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200: hyde_score += 5
        elif hyde_len < 50: hyde_score += 2
        if "\n" not in hyde_text: hyde_score += 5
        hyde_score += max(0, 5 - word_repetition_penalty(hyde_text))

    quality_score = 5
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec: quality_score += 5
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        quality_score += 5 if natural == len(parsed["vec"]) else 2
    if parsed["lex"]:
        with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if with_terms == len(parsed["lex"]): quality_score += 5
        elif with_terms > 0: quality_score += 2

    entity_score = 0
    entities = extract_named_entities(query)
    if entities and parsed["lex"]:
        with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if with_entities == len(parsed["lex"]): entity_score += 15
        elif with_entities > 0: entity_score += 5
        else: entity_score -= 30
        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count: entity_score -= generic_count * 15
        if parsed["vec"]:
            vec_with = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with > 0: entity_score += 5
    elif not entities:
        entity_score = 10

    think_bonus = 0 if used_thinking else 20
    total = format_score + diversity_score + hyde_score + quality_score + entity_score + think_bonus
    max_possible = 140 if parsed["hyde"] else 120
    percentage = max(0.0, min(100.0, total / max_possible * 100))

    if percentage >= 80: rating = "Excellent"
    elif percentage >= 60: rating = "Good"
    elif percentage >= 40: rating = "Acceptable"
    elif percentage >= 20: rating = "Poor"
    else: rating = "Failed"

    return {
        "format": format_score, "diversity": diversity_score, "hyde": hyde_score,
        "quality": quality_score, "entity": max(0, entity_score),
        "think_bonus": think_bonus, "total": max(0, total),
        "max_possible": max_possible, "percentage": round(percentage, 1),
        "rating": rating, "deductions": deductions,
        "entities_detected": list(entities) if entities else [],
    }




# =============================================================================
# 4-way model loading + generation (cells A/B/C/D, greedy + sampled decodes)
# =============================================================================
from transformers import set_seed

GRANITE_BASE = "ibm-granite/granite-3.3-2b-instruct"
QWEN_BASE = "Qwen/Qwen3-1.7B"

# Each cell: base + ordered adapter list (all but the last are merged, so a later
# adapter trains on the merged weights of earlier ones — required for SFT->GRPO).
# family drives the prompt: Granite was trained WITHOUT /no_think; Qwen WITH it.
MODELS = [
    {"label": "A:granite-sft-ep5",   "base": GRANITE_BASE, "adapters": ["nichenke/qmd-query-expansion-granite-2b-sft"],                                              "family": "granite"},
    {"label": "B:granite-sft-best",  "base": GRANITE_BASE, "adapters": ["nichenke/qmd-query-expansion-granite-2b-sft-best"],                                         "family": "granite"},
    {"label": "C:qwen-sft-current",  "base": QWEN_BASE,    "adapters": ["nichenke/qmd-query-expansion-qwen-1.7b-sft-current"],                                       "family": "qwen"},
    {"label": "D:qwen-deployed-grpo","base": QWEN_BASE,    "adapters": ["tobil/qmd-query-expansion-1.7B-sft", "tobil/qmd-query-expansion-1.7B-grpo"],                "family": "qwen"},
]

DECODES = ["greedy", "sampled"]
SEED = 42


def load_model_chain(base, adapters):
    print(f"Loading base {base}...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # NOTE: no tie_word_embeddings override — both families tie; forcing untied
    # randomly inits lm_head and corrupts generation.
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    for i, adapter in enumerate(adapters):
        last = (i == len(adapters) - 1)
        print(f"  {'loading' if last else 'merging'} adapter {adapter}...")
        model = PeftModel.from_pretrained(model, adapter)
        if not last:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def gen_one(model, tokenizer, query, family, decode, max_new_tokens=400):
    prefix = "/no_think " if family == "qwen" else ""
    messages = [{"role": "user", "content": f"{prefix}Expand this search query: {query}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id, use_cache=True)
    if decode == "sampled":
        kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    else:
        kwargs.update(do_sample=False, num_beams=1)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
    return gen


def run_cell(model, tokenizer, label, family, decode):
    # Fixed seed so the sampled run is reproducible across invocations.
    set_seed(SEED)
    scores = []
    for q in QUERIES:
        exp = gen_one(model, tokenizer, q, family, decode)
        scores.append(score_expansion_detailed(q, exp)["percentage"])
    n = len(scores)
    avg = sum(scores) / n
    zeros = sum(1 for s in scores if s == 0)
    strong = sum(1 for s in scores if s >= 80)
    print(f"  [{label} / {decode}] avg={avg:.1f}%  zeros={zeros}  >=80%={strong}/{n}")
    return {"label": label, "decode": decode, "avg": avg, "zeros": zeros, "strong": strong, "n": n}


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    rows = []
    for cell in MODELS:
        print(f"\n{'='*70}\nLOADING {cell['label']}\n{'='*70}")
        try:
            model, tokenizer = load_model_chain(cell["base"], cell["adapters"])
        except Exception as e:
            print(f"  FAILED to load {cell['label']}: {e}")
            for d in DECODES:
                rows.append({"label": cell["label"], "decode": d, "avg": None, "zeros": None, "strong": None, "n": len(QUERIES)})
            continue
        for d in DECODES:
            rows.append(run_cell(model, tokenizer, cell["label"], cell["family"], d))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n\n{'='*70}\n4-WAY RESULTS (reward.py %, {len(QUERIES)} queries)\n{'='*70}")
    print(f"{'cell':<26}{'greedy':>16}{'sampled':>16}")
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["decode"]] = r
    for label in [m["label"] for m in MODELS]:
        cells = by_label.get(label, {})
        def fmt(d):
            r = cells.get(d)
            if not r or r["avg"] is None:
                return "n/a"
            return f"{r['avg']:.1f}% (z{r['zeros']},s{r['strong']})"
        print(f"{label:<26}{fmt('greedy'):>16}{fmt('sampled'):>16}")
    print("\n(z = #zeros, s = #scores>=80%)")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
