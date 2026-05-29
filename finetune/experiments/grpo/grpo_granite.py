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
Refines the instruct SFT checkpoint with reward.py as the RL reward — targeting the
runaway-repetition / Diversity collapse that caps the instruct-SFT at ~85.9% on normal
queries (see docs analysis-granite-cluster-diagnosis).

  - BASE_MODEL = ibm-granite/granite-3.3-2b-instruct (has its own chat template, so the
    apply_chat_template prompt path and qmd serving are unchanged — no template plumbing).
  - SFT_MODEL / OUTPUT_MODEL / DATASET resolve from env or the logged-in user's namespace.
  - GRPO recipe (LoRA r=4 q/v, beta=0.04 KL, lr=5e-7, 200 steps) is kept IDENTICAL to the
    proven Qwen run for comparability; only model/data/repo change.

SELF-CONTAINED: reward.py (the single source of truth) is embedded below. The original
eval_common.py download fallback was removed — its tobil URL now 404s — and the run_eval
auto-eval (which used a Qwen-only /no_think prompt) is dropped; the real eval is
expand_eval.ts on the converted GGUF. Reward = the embedded QMDRewardFunction.

Run on HF Jobs (GRPO = generate + train, heavier than SFT):
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h \
        experiments/grpo/grpo_granite.py
"""

import os

import torch
from datasets import load_dataset
from huggingface_hub import login, whoami
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

BASE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

# ==========================================================================
# Embedded reward.py (single source of truth — scoring + QMDRewardFunction)
# ==========================================================================
import re
from collections import Counter

# =============================================================================
# Constants
# =============================================================================

# "only:" mode patterns - when query ends with these, expect only that type
# Format: "query /only:lex" (slash prefix, no space after colon)
ONLY_MODE_PATTERN = re.compile(r'\s+/only:(lex|vec|hyde)\s*$', re.IGNORECASE)

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in',
    'and', 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
})

KEY_TERM_STOPWORDS = frozenset({
    'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
    'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
    'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell',
    'about', 'from', 'into', 'between', 'through', 'during', 'after',
    'before', 'like', 'than', 'then', 'that', 'this', 'their', 'its',
    'was', 'were', 'has', 'had', 'been', 'being', 'have', 'not', 'but',
    'just', 'also', 'very', 'so', 'if', 'at', 'by', 'up', 'out', 'all',
    'some', 'any', 'no', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'only', 'same', 'such', 'here', 'there', 'asked', 'said',
    'notes', 'meeting', 'email', 'discussion', 'conversation', 'call',
})

# Words that commonly start queries but aren't named entities.
# Used for position-0 entity detection to avoid false positives.
QUERY_VERB_STOPWORDS = frozenset({
    'configure', 'setup', 'install', 'build', 'create', 'make', 'run',
    'start', 'stop', 'check', 'test', 'debug', 'fix', 'update', 'change',
    'add', 'remove', 'delete', 'use', 'using', 'need', 'want', 'should',
    'would', 'could', 'help', 'please', 'best', 'good', 'new', 'old',
    'latest', 'recent', 'setting', 'settings', 'compare', 'comparing',
    'implement', 'implementing', 'deploy', 'deploying', 'migrate',
    'migrating', 'optimize', 'optimizing', 'understand', 'understanding',
    'explain', 'list', 'describe', 'define', 'convert', 'connecting',
    'performance', 'overview', 'introduction', 'tutorial', 'example',
    'difference', 'between', 'about', 'review', 'resolve', 'resolving',
    'troubleshoot', 'troubleshooting', 'monitor', 'monitoring', 'manage',
    'managing', 'enable', 'disable', 'set', 'write', 'read', 'search',
    'possible', 'common', 'typical', 'recommended', 'alternative',
})

GENERIC_LEX_PHRASES = frozenset({
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with',
})

# Words commonly injected as filler/noise into lex lines by template generators
# (e.g. "ancient overview rome timeline"). Penalized when absent from the query.
INTERIOR_FILLER_WORDS = frozenset({'overview', 'basics'})

# Chat template tokens that indicate a broken output
CHAT_TEMPLATE_TOKENS = frozenset({
    '<|im_start|>', '<|im_end|>', '<|endoftext|>',
    '\nassistant\n', '\nuser\n',
})


# =============================================================================
# Parsing
# =============================================================================

def parse_expansion(text: str) -> dict:
    """Parse a multi-line expansion into {lex, vec, hyde, invalid} lists."""
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


def detect_only_mode(query: str) -> tuple[str | None, str]:
    """Detect if query ends with 'only: lex/vec/hyde'.
    
    Returns (only_type, base_query) where only_type is None for normal queries.
    """
    match = ONLY_MODE_PATTERN.search(query)
    if match:
        only_type = match.group(1).lower()
        base_query = query[:match.start()].strip()
        return only_type, base_query
    return None, query


def clean_model_output(text: str) -> tuple[str, bool]:
    """Strip chat template artifacts from model output.

    Returns (cleaned_text, used_thinking) where used_thinking is True
    if the model emitted <think>...</think> blocks.
    """
    text = text.replace('<|im_end|>', '').strip()

    used_thinking = '<think>' in text and '</think>' in text
    if used_thinking:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    return text, used_thinking


# =============================================================================
# Helpers
# =============================================================================

def extract_named_entities(query: str) -> set:
    """Extract named entities using heuristics.

    Detects: ALL-CAPS acronyms (TDS, API), capitalized proper nouns (React, Bob),
    technical terms with special chars (node.js, C++), CamelCase (JavaScript),
    and compound names (TDS motorsports -> both words).

    Position-0 words are also detected as entities if they are capitalized and
    not common query-starting verbs (e.g. "Bob asked about deploy" -> "bob").

    Compound chaining extends one level from a directly-detected entity:
    "TDS motorsports" -> {tds, motorsports}; "TDS motorsports team" -> {tds, motorsports}.
    """
    entities = set()
    words = query.split()
    prev_was_base_entity = False

    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_base_entity = False
            continue

        is_base_entity = False

        # ALL-CAPS acronyms: TDS, API, GPU, AWS
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower())
            is_base_entity = True
        # Capitalized proper nouns (any position, including first word)
        elif clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            if i > 0:
                # Non-first words: always treat as entity
                entities.add(clean.lower())
                is_base_entity = True
            elif clean.lower() not in QUERY_VERB_STOPWORDS:
                # First word: also entity if not a common query verb
                entities.add(clean.lower())
                is_base_entity = True
        # Technical terms with special chars: node.js, C++, .NET
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_base_entity = True
        # CamelCase: JavaScript, TypeScript
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_base_entity = True
        # Compound names: word following a BASE entity only (one level deep).
        elif prev_was_base_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())

        prev_was_base_entity = is_base_entity

    return entities


def get_key_terms(query: str) -> set:
    """Get non-stopword terms from a query."""
    return set(query.lower().split()) - KEY_TERM_STOPWORDS


def lex_preserves_key_terms(lex_line: str, query: str) -> bool:
    """Does the lex line contain at least one key term from the query?"""
    key_terms = get_key_terms(query)
    if not key_terms:
        return True
    return bool(key_terms & set(lex_line.lower().split()))


def lex_preserves_entities(line: str, entities: set) -> bool:
    """Does the line contain at least one named entity?"""
    if not entities:
        return True
    lower = line.lower()
    return any(e in lower for e in entities)


def lex_has_filler(lex_line: str, query: str) -> bool:
    """Does the lex line contain an INTERIOR_FILLER_WORDS word absent from the query?"""
    query_words = set(query.lower().split())
    return any(w in INTERIOR_FILLER_WORDS and w not in query_words
               for w in lex_line.lower().split())


def lex_is_generic(lex_line: str) -> bool:
    """Is this lex line a useless generic filler phrase?"""
    lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lower or lower.startswith(phrase.split()[0]):
            remaining = lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:
                return True
    return False


def word_set_distance(a: str, b: str) -> int:
    """Symmetric difference of word sets (how many words are unique to one)."""
    return len(set(a.lower().split()) ^ set(b.lower().split()))


def is_diverse(a: str, b: str, min_distance: int = 2) -> bool:
    """Are two strings sufficiently different?"""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return False
    return word_set_distance(a, b) >= min_distance


def echoes_query(expansion: str, query: str) -> bool:
    """Is this expansion just echoing the original query?"""
    exp, q = expansion.lower().strip(), query.lower().strip()
    return exp == q or (q in exp and len(exp) < len(q) + 10)


def word_repetition_penalty(text: str) -> int:
    """Penalty for words repeated 3+ times (excluding stopwords)."""
    counts = Counter(re.findall(r'\b\w+\b', text.lower()))
    return sum((c - 2) * 2 for w, c in counts.items()
               if c >= 3 and w not in STOPWORDS and len(w) > 2)


# =============================================================================
# Scoring
# =============================================================================

def _score_only_mode(query: str, base_query: str, text: str, used_thinking: bool, only_type: str) -> dict:
    """Score an 'only:' mode expansion. Expects ONLY the requested type."""
    parsed = parse_expansion(text)
    deductions = []
    
    # Expected type must be present
    expected_items = parsed.get(only_type, [])
    if not expected_items:
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "think_bonus": 0, "total": 0, "max_possible": 100,
            "percentage": 0.0, "rating": "Failed",
            "deductions": [f"missing expected {only_type}: output"],
            "parsed": parsed,
            "entities_detected": [],
            "only_mode": only_type,
        }
    
    # Penalize presence of OTHER types
    other_types = {"lex", "vec", "hyde"} - {only_type}
    unwanted_count = sum(len(parsed.get(t, [])) for t in other_types)
    if unwanted_count > 0:
        deductions.append(f"contains unwanted types (expected only {only_type})")
    
    # --- Format (0-30) ---
    format_score = 30 if unwanted_count == 0 else max(0, 30 - unwanted_count * 10)
    
    # --- Diversity (0-30) ---
    diversity_score = 0
    div_threshold = 3 if len(base_query.split()) >= 5 else 2
    if len(expected_items) >= 2:
        diversity_score += 15
        # Check for diversity among items
        div_score = 15
        for i, a in enumerate(expected_items):
            for b in expected_items[i+1:]:
                if not is_diverse(a, b, div_threshold):
                    div_score -= 5
                    deductions.append(f"{only_type} duplicate: {a[:20]}...")
        diversity_score += max(0, div_score)
    elif len(expected_items) == 1:
        diversity_score = 15  # One item is fine for single-type output
    
    # Check for echoes
    for exp in expected_items:
        if echoes_query(exp, base_query):
            diversity_score -= 5
            deductions.append(f"echoes query: {exp[:20]}...")
    diversity_score = max(0, diversity_score)
    
    # --- Type-specific quality (0-20) ---
    quality_score = 10  # base
    entities = extract_named_entities(base_query)
    
    if only_type == "lex":
        # Lex should be short keyword phrases with key terms
        with_terms = sum(1 for l in expected_items if lex_preserves_key_terms(l, base_query))
        if with_terms == len(expected_items):
            quality_score += 5
        # Check for generic phrases
        generic = sum(1 for l in expected_items if lex_is_generic(l))
        if generic == 0:
            quality_score += 5
        else:
            deductions.append(f"{generic} generic lex phrases")
        # Penalty: lex lines containing filler words absent from the query
        filler_count = sum(1 for l in expected_items if lex_has_filler(l, base_query))
        if filler_count > 0:
            quality_score -= filler_count * 3
            deductions.append(f"{filler_count} lex line(s) with filler words")
    
    elif only_type == "vec":
        # Vec should be natural language sentences
        natural = sum(1 for v in expected_items if " " in v and len(v) > 15)
        if natural == len(expected_items):
            quality_score += 10
        else:
            quality_score += 5
            deductions.append("vec not all natural language")
    
    elif only_type == "hyde":
        # Hyde should be a document snippet (50-200 chars)
        hyde_text = expected_items[0]
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            quality_score += 10
        elif 30 <= hyde_len <= 300:
            quality_score += 5
            deductions.append(f"hyde length {hyde_len} (ideal: 50-200)")
        else:
            deductions.append(f"hyde length {hyde_len} out of range")
    
    # --- Entity preservation (0-20) ---
    entity_score = 10  # base
    if entities:
        with_entities = sum(1 for item in expected_items if lex_preserves_entities(item, entities))
        if with_entities == len(expected_items):
            entity_score += 10
        elif with_entities > 0:
            entity_score += 5
        else:
            entity_score = 0
            deductions.append(f"missing entities: {entities}")
    
    # --- Think bonus (0-20) ---
    think_bonus = 0 if used_thinking else 20
    
    # --- Total ---
    total = format_score + diversity_score + quality_score + entity_score + think_bonus
    max_possible = 120
    percentage = max(0.0, min(100.0, total / max_possible * 100))
    
    if percentage >= 80:
        rating = "Excellent"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Acceptable"
    elif percentage >= 20:
        rating = "Poor"
    else:
        rating = "Failed"
    
    return {
        "format": format_score,
        "diversity": diversity_score,
        "hyde": 0,  # not used in only mode (quality covers it)
        "quality": quality_score,
        "entity": entity_score,
        "think_bonus": think_bonus,
        "total": total,
        "max_possible": max_possible,
        "percentage": round(percentage, 1),
        "rating": rating,
        "deductions": deductions,
        "parsed": parsed,
        "entities_detected": list(entities) if entities else [],
        "only_mode": only_type,
    }


def score_expansion_detailed(query: str, expansion: str) -> dict:
    """Score an expansion with full breakdown. Returns dict with all dimensions."""
    text, used_thinking = clean_model_output(expansion.strip())
    deductions = []

    # Detect "only:" mode
    only_type, base_query = detect_only_mode(query)

    def _fail(reason):
        return {
            "format": 0, "diversity": 0, "hyde": 0, "quality": 0, "entity": 0,
            "think_bonus": 0, "total": 0, "max_possible": 100,
            "percentage": 0.0, "rating": "Failed",
            "deductions": [reason],
            "parsed": parse_expansion(expansion),
            "entities_detected": [],
            "only_mode": only_type,
        }

    # Hard fail: remaining chat template tokens
    if any(tok in text for tok in CHAT_TEMPLATE_TOKENS):
        return _fail("CHAT TEMPLATE LEAKAGE")

    # Hard fail: every non-empty line must have a valid prefix
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("lex:", "vec:", "hyde:")):
            return _fail(f"INVALID LINE: {line[:50]}")

    # --- Handle "only:" mode separately ---
    if only_type:
        return _score_only_mode(query, base_query, text, used_thinking, only_type)

    parsed = parse_expansion(text)

    # --- Format (0-30) ---
    format_score = 10  # no invalid lines (guaranteed by hard fail)
    if parsed["lex"]:
        format_score += 10
    else:
        deductions.append("missing lex:")
    if parsed["vec"]:
        format_score += 10
    else:
        deductions.append("missing vec:")

    # --- Diversity (0-30) ---
    diversity_score = 0

    types_present = sum(1 for t in ("lex", "vec") if parsed[t])
    if types_present >= 2:
        diversity_score += 10
    else:
        deductions.append("only one type")

    if len(parsed["lex"]) + len(parsed["vec"]) >= 2:
        diversity_score += 5

    div_threshold = 3 if len(query.split()) >= 5 else 2
    lex_div = 5
    for i, a in enumerate(parsed["lex"]):
        for b in parsed["lex"][i+1:]:
            if not is_diverse(a, b, div_threshold):
                lex_div -= 2
                deductions.append(f"lex duplicate: {a[:20]}...")
    diversity_score += max(0, lex_div)

    vec_div = 5
    for i, a in enumerate(parsed["vec"]):
        for b in parsed["vec"][i+1:]:
            if not is_diverse(a, b, div_threshold):
                vec_div -= 2
                deductions.append(f"vec duplicate: {a[:20]}...")
    diversity_score += max(0, vec_div)

    echo = 5
    lex_echo_count = 0
    for exp in parsed["lex"]:
        if echoes_query(exp, query):
            lex_echo_count += 1
            deductions.append(f"lex echoes query: {exp[:20]}...")
    # Harsh penalty for lex echoes - they're useless
    if lex_echo_count > 0:
        echo -= lex_echo_count * 10  # -10 per echo
    
    for exp in parsed["vec"]:
        if echoes_query(exp, query):
            echo -= 3  # vec echoes less severe (natural language overlap ok)
            deductions.append(f"vec echoes query: {exp[:20]}...")
    diversity_score += max(-10, echo)  # can go negative

    # --- HyDE (0-20, optional bonus) ---
    hyde_score = 0
    if parsed["hyde"]:
        hyde_text = parsed["hyde"][0]
        hyde_score += 5
        hyde_len = len(hyde_text)
        if 50 <= hyde_len <= 200:
            hyde_score += 5
        elif hyde_len < 50:
            hyde_score += 2
            deductions.append(f"hyde too short ({hyde_len})")
        else:
            deductions.append(f"hyde too long ({hyde_len})")
        if "\n" not in hyde_text:
            hyde_score += 5
        hyde_score += max(0, 5 - word_repetition_penalty(hyde_text))

    # --- Extract entities (used by both quality and entity sections) ---
    entities = extract_named_entities(query)

    # --- Quality (0-20) ---
    quality_score = 5  # base relevance
    if parsed["lex"] and parsed["vec"]:
        avg_lex = sum(len(l) for l in parsed["lex"]) / len(parsed["lex"])
        avg_vec = sum(len(v) for v in parsed["vec"]) / len(parsed["vec"])
        if avg_lex <= avg_vec:
            quality_score += 5
        else:
            deductions.append("lex longer than vec")
    if parsed["vec"]:
        natural = sum(1 for v in parsed["vec"] if " " in v and len(v) > 15)
        quality_score += 5 if natural == len(parsed["vec"]) else 2
    if parsed["lex"]:
        with_terms = sum(1 for l in parsed["lex"] if lex_preserves_key_terms(l, query))
        if with_terms == len(parsed["lex"]):
            quality_score += 5
        elif with_terms > 0:
            quality_score += 2
        else:
            deductions.append("lex missing key terms")

    # Penalty: lex lines containing filler words absent from the query
    if parsed["lex"]:
        filler_count = sum(1 for l in parsed["lex"] if lex_has_filler(l, query))
        if filler_count > 0:
            quality_score -= filler_count * 3
            deductions.append(f"{filler_count} lex line(s) with filler words")

    # Bonus: lex uses quoted phrases for multi-word queries (+3)
    if parsed["lex"] and len(query.split()) >= 2:
        lex_joined = " ".join(parsed["lex"])
        if '"' in lex_joined:
            quality_score += 3

    # --- Entity Preservation (-45 to +20) ---
    entity_score = 0
    if entities and parsed["lex"]:
        # Per-line check: do lex lines contain entities?
        with_entities = sum(1 for l in parsed["lex"] if lex_preserves_entities(l, entities))
        if with_entities == len(parsed["lex"]):
            entity_score += 15
        elif with_entities > 0:
            entity_score += 5
        else:
            entity_score -= 30
            deductions.append(f"lex missing entities: {entities}")

        # Per-entity coverage: is each entity mentioned somewhere in lex+vec?
        all_output = " ".join(parsed["lex"] + parsed["vec"]).lower()
        missing_entities = {e for e in entities if e not in all_output}
        if missing_entities:
            penalty = len(missing_entities) * 20
            entity_score -= penalty
            deductions.append(f"entities dropped: {missing_entities}")

        generic_count = sum(1 for l in parsed["lex"] if lex_is_generic(l))
        if generic_count:
            entity_score -= generic_count * 15
            deductions.append(f"{generic_count} generic lex phrases")

        if parsed["vec"]:
            vec_with = sum(1 for v in parsed["vec"] if lex_preserves_entities(v, entities))
            if vec_with > 0:
                entity_score += 5
    elif not entities:
        entity_score = 10

    # --- Think bonus (0-20): reward NOT using thinking mode ---
    think_bonus = 0 if used_thinking else 20

    # --- Total ---
    total = format_score + diversity_score + hyde_score + quality_score + entity_score + think_bonus
    max_possible = 140 if parsed["hyde"] else 120
    percentage = max(0.0, min(100.0, total / max_possible * 100))

    # Hard cap: lex echoes are unacceptable - cap at 50%
    if lex_echo_count > 0:
        percentage = min(percentage, 50.0)
        deductions.insert(0, f"CAPPED: {lex_echo_count} lex echo(es)")

    if percentage >= 80:
        rating = "Excellent"
    elif percentage >= 60:
        rating = "Good"
    elif percentage >= 40:
        rating = "Acceptable"
    elif percentage >= 20:
        rating = "Poor"
    else:
        rating = "Failed"

    return {
        "format": format_score,
        "diversity": diversity_score,
        "hyde": hyde_score,
        "quality": quality_score,
        "entity": max(0, entity_score),
        "think_bonus": think_bonus,
        "total": max(0, total),
        "max_possible": max_possible,
        "percentage": round(percentage, 1),
        "rating": rating,
        "deductions": deductions,
        "parsed": parsed,
        "entities_detected": list(entities) if entities else [],
        "only_mode": None,
    }


def score_expansion(query: str, expansion: str) -> float:
    """Score expansion as a float in [0.0, 1.0] for use as RL reward."""
    result = score_expansion_detailed(query, expansion)
    return max(0.0, min(1.0, result["total"] / result["max_possible"]))


def extract_query_from_prompt(prompt: str) -> str:
    """Extract the query string from a chat-formatted prompt."""
    if "Expand this search query:" in prompt:
        query = prompt.split("Expand this search query:")[-1].strip()
        if "<|im_end|>" in query:
            query = query.split("<|im_end|>")[0].strip()
        return query
    return prompt.strip()


# =============================================================================
# TRL-compatible reward class
# =============================================================================

class QMDRewardFunction:
    """Reward function compatible with TRL's GRPOTrainer."""
    __name__ = "qmd_scoring_reward"

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            query = ""
            if prompts and i < len(prompts):
                query = extract_query_from_prompt(prompts[i])
            rewards.append(score_expansion(query, completion))
        return rewards


# =============================================================================
# CLI: run standalone to test the reward function
# =============================================================================



# ==========================================================================
# GRPO training entrypoint
# ==========================================================================


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

    # Format with Granite's own chat template (matches the SFT prompt regime).
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
    print("Next: convert_gguf.py --base", BASE_MODEL, "--sft", sft_model,
          "--grpo", output_model, "--output", output_model + "-gguf")


if __name__ == "__main__":
    main()
