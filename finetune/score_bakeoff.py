# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fair-comparison scorer for the query-expander bake-off (#144, #132).

Companion to score_expansions.py. Adds the slices the bake-off decision actually
needs and that score_expansions.py does not report:

  - the 45 NORMAL queries (excludes the 6 `/only:` queries — only-mode is separately
    broken via a train/eval syntax mismatch and a structured grammar wrongly forces
    all-3 on them; it is routed out of the headline per the seed playbook),
  - HyDE coverage (count of rows with >=1 hyde line) co-reported with every reward
    number, because reward.py treats HyDE as an OPTIONAL 20pt bonus — a HyDE-sparse
    config scores deceptively high, so model-vs-model is only fair at equal coverage,
  - HyDE median length (the stated Llama risk is hyde-length discipline, target <=180),
  - per-component means (format/diversity/hyde/quality/entity) to read WHERE a deficit is,
  - average raw lines/query by type (the runaway-repetition tell).

Re-scores an existing {query,expansion} JSONL — it does NOT regenerate. So running it
on each model's OWN output JSONL (each produced under that model's OWN decode profile)
keeps the comparison apples-to-apples without any cross-model profile bleed.

    uv run finetune/score_bakeoff.py finetune/outputs/expand_llama_structured_greedy.jsonl
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import detect_only_mode, parse_expansion, score_expansion_detailed  # noqa: E402


def load_rows(path: str) -> list[dict]:
    """Robust load: skip stdout spinner pollution, dedupe by query (keep first)."""
    rows, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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
    return rows


def summarize(label: str, rows: list[dict]) -> dict:
    pcts, comps = [], {"format": [], "diversity": [], "hyde": [], "quality": [], "entity": []}
    hyde_present = 0
    hyde_lens: list[int] = []
    lines_by_type = {"lex": [], "vec": [], "hyde": []}
    for row in rows:
        detail = score_expansion_detailed(row["query"], row["expansion"])
        pcts.append(detail["percentage"])
        for k in comps:
            comps[k].append(detail[k])
        parsed = parse_expansion(row["expansion"])
        for t in lines_by_type:
            lines_by_type[t].append(len(parsed[t]))
        if parsed["hyde"]:
            hyde_present += 1
            hyde_lens.append(len(parsed["hyde"][0]))
    n = len(rows)
    avg = sum(pcts) / n if n else 0.0
    strong = sum(1 for p in pcts if p >= 80)
    zeros = sum(1 for p in pcts if p == 0)
    out = {
        "label": label, "n": n, "avg": avg, "strong": strong, "zeros": zeros,
        "hyde_cov": hyde_present,
        "hyde_median_len": (statistics.median(hyde_lens) if hyde_lens else 0),
        "comp_means": {k: (sum(v) / len(v) if v else 0.0) for k, v in comps.items()},
        "avg_lines": {t: (sum(v) / len(v) if v else 0.0) for t, v in lines_by_type.items()},
    }
    return out


def print_summary(s: dict) -> None:
    print(f"\n=== {s['label']} (n={s['n']}) ===")
    print(f"  avg reward : {s['avg']:.1f}%   strong>=80%: {s['strong']}/{s['n']}   zeros: {s['zeros']}")
    print(f"  HyDE cov   : {s['hyde_cov']}/{s['n']} rows with >=1 hyde   hyde median len: {s['hyde_median_len']:.0f} chars")
    cm = s["comp_means"]
    print(f"  components  : F{cm['format']:.1f} D{cm['diversity']:.1f} "
          f"H{cm['hyde']:.1f} Q{cm['quality']:.1f} E{cm['entity']:.1f}")
    al = s["avg_lines"]
    print(f"  avg lines/q : lex {al['lex']:.1f}  vec {al['vec']:.1f}  hyde {al['hyde']:.1f}  (runaway tell)")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: score_bakeoff.py <expansions.jsonl>", file=sys.stderr)
        return 2
    rows = load_rows(sys.argv[1])
    normal = [r for r in rows if detect_only_mode(r["query"])[0] is None]
    only = [r for r in rows if detect_only_mode(r["query"])[0] is not None]

    print(f"loaded {len(rows)} rows  ({len(normal)} normal, {len(only)} only-mode)")
    print_summary(summarize("FULL corpus", rows))
    print_summary(summarize("NORMAL (headline — 45 expected, excl /only:)", normal))
    if only:
        print_summary(summarize("ONLY-mode (routed out of headline)", only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
