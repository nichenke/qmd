/**
 * Production-truth query-expansion harness.
 *
 * Drives the REAL `LlamaCpp.expandQuery()` serving path (grammar-constrained,
 * node-llama-cpp, sampled decode, query-term post-filter) over the full
 * evals/queries.txt corpus, using the configured default expander GGUF
 * (currently the deployed Qwen SFT+GRPO q4_k_m). Emits one JSONL row per query:
 *   {"query": "...", "expansion": "lex: ...\nvec: ...\nhyde: ..."}
 *
 * The reconstructed text is exactly what qmd searches on — it has already passed
 * the grammar and the hasQueryTerm filter inside expandQuery. Score it with
 * finetune/score_expansions.py (which calls reward.py).
 *
 *   ./node_modules/.bin/tsx finetune/expand_eval.ts > finetune/outputs/expand_baseline.jsonl
 *
 * Optional: QMD_GENERATE_MODEL=hf:... to point at a different expander GGUF.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { LlamaCpp } from "../src/llm.ts";

const here = dirname(fileURLToPath(import.meta.url));
const queryFile = join(here, "evals", "queries.txt");

const queries = readFileSync(queryFile, "utf8")
  .split("\n")
  .map((l) => l.trim())
  .filter((l) => l.length > 0 && !l.startsWith("#"));

process.stderr.write(`[expand_eval] ${queries.length} queries from ${queryFile}\n`);

const llm = new LlamaCpp({});
process.stderr.write(`[expand_eval] expander model: ${llm.generateModelName}\n`);

try {
  let i = 0;
  for (const query of queries) {
    i++;
    let expansion = "";
    try {
      const queryables = await llm.expandQuery(query);
      expansion = queryables.map((q) => `${q.type}: ${q.text}`).join("\n");
    } catch (err) {
      process.stderr.write(`[expand_eval] query ${i} FAILED: ${String(err)}\n`);
    }
    process.stdout.write(JSON.stringify({ query, expansion }) + "\n");
    process.stderr.write(`[expand_eval] ${i}/${queries.length} done: ${query.slice(0, 48)}\n`);
  }
} finally {
  await llm.dispose();
}
