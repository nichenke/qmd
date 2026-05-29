# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Score a production-truth expansion JSONL (from expand_eval.ts) with reward.py.

    uv run finetune/score_expansions.py finetune/outputs/expand_baseline.jsonl

Reads {"query","expansion"} rows, scores each via reward.score_expansion_detailed,
and prints a per-category summary plus the same headline metrics as eval_4way.py
(avg %, #zeros, #strong>=80%).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import score_expansion_detailed  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: score_expansions.py <expansions.jsonl>", file=sys.stderr)
        return 2
    # Robust load: tsx/node-llama-cpp can leak a download spinner onto stdout on a
    # first-time model fetch, so skip any line that is not a valid {query,expansion}
    # JSON object, and dedupe by query (keep first) in case a query was re-emitted.
    rows = []
    seen = set()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "query" in obj and "expansion" in obj and obj["query"] not in seen:
            seen.add(obj["query"])
            rows.append(obj)

    scores = []
    print(f"{'query':<50}{'score%':>8}  breakdown")
    print("-" * 90)
    for row in rows:
        detail = score_expansion_detailed(row["query"], row["expansion"])
        pct = detail["percentage"]
        scores.append(pct)
        bd = f"F{detail['format']} D{detail['diversity']} H{detail['hyde']} Q{detail['quality']} E{detail['entity']}"
        print(f"{row['query'][:48]:<50}{pct:>7.1f}  {bd}")

    n = len(scores)
    avg = sum(scores) / n if n else 0.0
    zeros = sum(1 for s in scores if s == 0)
    strong = sum(1 for s in scores if s >= 80)
    print("-" * 90)
    print(f"n={n}  avg={avg:.1f}%  zeros={zeros}  strong(>=80%)={strong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
