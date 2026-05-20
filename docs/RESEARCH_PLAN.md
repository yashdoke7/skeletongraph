# SkeletonGraph — Research Plan

**Status**: living document. Anti-drift reference. Read this first whenever the
context window is recompacted or a new contributor joins.

**Last updated**: 2026-05-20 (post competitive-landscape review)

---

## 1. The reframed thesis

> **Adaptive multi-strategy code retrieval with summary-tier context, driven by
> a query-classifying curator, produces higher-precision retrieval than any
> single strategy and translates that precision differential into measurable
> cost reduction on competent models — partially closing the consolidation gap
> identified by ContextBench (arXiv:2602.05892).**

**Not** "SG is the best code retriever." **Not** "graph + MCP wins." Both of
those framings are lost — Codebase-Memory (March 2026, MCP + tree-sitter graph,
66 languages, 900+ stars in 4 weeks) owns that wedge.

What's still ours and is genuinely novel:

1. **The curator** — a query classifier that routes each query to the most
   appropriate sub-strategy (graph traversal for entity queries, embeddings
   for natural-language queries, lexical fallback for module-level constants).
   No published work makes this routing the central contribution.
2. **Summary tiers** — Tier 1 (target body) / Tier 2 (neighbor skeleton +
   summary) / Tier 3 (FQN only). Codebase-Memory has *no* summary layer.
3. **Hybrid signals** — lexical + structural + semantic, with principled
   routing. Codebase-Memory explicitly rejects embeddings; we have the
   experiment that shows when each helps.

---

## 2. The three contributions, restated for the paper

| # | Contribution | Measurable in eval |
|---|---|---|
| C1 | Query-classifying curator that routes to the right retrieval sub-strategy | Oracle-router upper bound vs SG curator vs each single strategy |
| C2 | Summary-tier context reduces the consolidation gap on competent models | ContextBench usage-drop metric; SG vs baselines on usage % |
| C3 | Hybrid (lexical + structural + semantic) outperforms any single signal on Pareto front of precision-vs-cost | Precision/recall/cost curves per arm |

These are the three claims we run experiments to defend. Every figure and
table in the paper supports one of them.

---

## 3. What we are NOT claiming

To stay honest and avoid reviewer destruction:

- ❌ "SG beats every retrieval system on every task"
- ❌ "Graph-based code retrieval is novel" (RepoGraph, RANGER, Aider, Codebase-Memory)
- ❌ "MCP integration is a contribution" (table stakes)
- ❌ "Retrieval is the dominant bottleneck" (consolidation gap research says no)
- ❌ "Multi-language support is best-in-class" (Codebase-Memory: 66 languages)

The honest tone: "We study *when* each retrieval strategy wins, *which*
SG components carry the gain, and *how much* the consolidation gap shrinks with
better context structure. The contribution is the experimental decomposition,
not a new headline number."

---

## 4. Competitive landscape (as of May 2026)

| System | What it is | Our relation |
|---|---|---|
| **Codebase-Memory** (Mar 2026) | MCP + tree-sitter graph + 14 typed query tools, 66 languages, SQLite, no embeddings. 10× fewer tokens vs file-exploration. | **The strongest direct competitor.** We must compare. Their pure-graph framing is what we differentiate against (we add hybrid + summaries). |
| **Aider repo-map** | Tree-sitter symbol graph + PageRank. Embedded in Aider CLI. Mature. | Primary baseline. Open source, easy to wrap. |
| **RepoGraph** (ICLR 2025) | Repository-level code graphs. +32.8% relative SWE-bench improvement. | Established academic baseline. Optional comparison (heavy infra). |
| **RANGER** (Sept 2025) | Graph + Cypher queries + MCTS exploration. | Academic peer; mention in related work. |
| **Augment** (commercial) | Hybrid BM25 + embedding + code-graph. | Same recipe, productized. Position as "the open-source curated version." |
| **Hybrid RAG** (deployed industry default) | BM25 + dense embedding + cross-encoder rerank. | Required strong baseline. |

**ContextBench** (Feb 2026, arXiv:2602.05892): the new standard benchmark for
this exact problem. 1,136 tasks, 66 repos, 8 languages, file/block/line F1.
**We must report on it.** It is no longer optional.

---

## 5. Evaluation plan — what we run, in what order

### Tier 0 — Local 7B (free, validates harness)

- ✅ Harness works (after the embedding + isolation + tools fixes)
- ✅ 4-arm differentiation visible (SG precision 0.32 vs BM25 0.09 vs grep 0.08)
- Next: add Aider + Hybrid-RAG arms, run on 30 tasks, sanity-check ordering

### Tier 1 — AMD 32B, the headline (~$30 of $100)

- Model: Qwen2.5-Coder-32B-Instruct (vLLM)
- Tasks: 150 SWE-bench Verified, stratified by difficulty + repo
- Arms: SG, Aider, Hybrid-RAG, none
- Reports: pass@1, retrieval F1/precision/recall, tokens-per-resolved-task,
  $-per-resolved-task, McNemar's pairwise vs SG, bootstrap 95% CI

### Tier 2 — Ablation, the paper's intellectual core (~$20)

- Same 150 tasks, same model
- Arms: SG-full, SG-nograph, SG-norerank, SG-nosummary, SG-noembedding
- Reports: which component carries the gain; expected loss per component

### Tier 3 — Scaling, the consolidation-gap story (~$15)

- Same 150 tasks, two models: Qwen2.5-Coder-7B + 32B
- Arms: SG, Hybrid-RAG (strongest baseline)
- Reports: does SG's precision advantage GROW with model strength? (it should)
  → consolidation gap shrinks because better model uses better context

### Tier 4 — Variance + significance (~$10)

- 60-task stratified subset, 3 repeats
- Arms: SG, Hybrid-RAG
- Reports: mean ± std, McNemar's exact, paired bootstrap

### Tier 5 — Second benchmark (~$10, if budget permits)

- ContextBench Lite (500 tasks)
- Arms: SG, Hybrid-RAG, Aider
- Reports: file-F1, block-F1, line-F1, usage-drop

**Total budget**: ~$85 of $100. Leaves $15 buffer for re-runs.

**Strict rule**: DO NOT iterate on AMD. Every parameter is locked here. Run
once, analyze, write.

---

## 6. Metrics — what every run captures

The granularity below is to support any figure/table we might need later
without re-running. Captured per `(task, arm, model, repeat)` run:

### Run-level
- `run_id`, `task_id`, `arm`, `model`, `repeat`, `repo`, `base_commit`,
  `gold_files`, `model_patch`, `resolved` (post-verify)
- `stopped` ∈ {submit, max_turns, error, no_tool}, `error`, `n_turns`, `wall_s`

### Cost / tokens
- `billed_input`, `billed_output`, `cached_input`, `peak_context`
- `imputed_cost` (reference price sheet)
- Per-turn usage in `turns[].usage`

### Retrieval quality (file granularity)
- `retrieval_hit` (binary recall — gameable; keep for backwards compat)
- `retrieval_precision` (gold / returned)
- `retrieval_rank` (1-indexed rank of first gold file, 0 = absent)
- `first_search_hits` (file list, ordered)
- `search_calls[]` — every search with `{turn_index, query, hits, gold_in_hits}`
- `unique_files_retrieved_total`
- `cumulative_recall_per_search` (Pareto curve building block)

### Context utilization (consolidation gap)
- `files_read[]` — every read with `{turn, path, was_gold, lines_read}`
- `gold_files_read_count`, `gold_files_read_first_turn`
- `files_retrieved_but_never_read` (count + list)
- `usage_drop` — ContextBench-style: bytes/lines retrieved that didn't appear
  in the final patch

### Patch quality
- `patch_lines_added`, `patch_lines_removed`, `patch_files_touched`,
  `patch_hunks`
- `patch_syntactic_valid` (Python AST parse of changed files)
- `edited_gold_file` (touches any gold file)

### Trajectory shape
- `tool_counts` (per tool name)
- `time_to_first_edit_turn` (turn of first successful `edit_file`)
- `time_to_first_gold_read_turn`
- `edits_attempted` / `edits_successful` (model tried but `old_str` not found)
- `empty_submit_blocked` (did the guard fire)

### Embedding state (SG only)
- `embeddings_used` (None=N/A, True/False) — flags silent degradation

---

## 7. Figures and tables (what we produce from the run dumps)

| # | Artifact | Backed by |
|---|---|---|
| Table 1 | Retrieval quality (per arm): Precision@k, Recall@k, MRR, File-F1, Block-F1 | first_search_hits + search_calls + gold blocks (TBD) |
| Table 2 | Downstream task: pass@1, edited-gold%, tokens-per-resolved, $-per-resolved | resolved + tokens + cost |
| Table 3 | Significance: McNemar's vs SG, 95% CI | paired pass/fail per arm |
| Table 4 | Ablation: SG-full vs each component disabled | tier 2 runs |
| Fig 1 | Consolidation gap: % of retrieved gold actually in patch, per arm | files_read + model_patch |
| Fig 2 | Curator routing: query class → winning sub-strategy (oracle vs SG vs single) | search_calls per query + ground-truth gold |
| Fig 3 | Scaling: SG vs Hybrid-RAG on 7B and 32B, all metrics | tier 3 runs |
| Fig 4 | Pareto: precision vs cost across arms | precision + imputed_cost |
| Fig 5 | Time-to-edit distribution per arm | time_to_first_edit_turn |
| Fig 6 | Failure-mode taxonomy: max_turns/no_tool/error counts per arm | stopped + error |

All figures must be reproducible from the run JSONs alone — no model in the
loop. Aggregate scripts only.

---

## 8. Venue strategy

**Workshop first, conference if numbers are unambiguous:**

- **Workshop targets** (Spring 2026 submission window):
  - ICLR 2027 workshops (e.g. R0-FoMo, Tiny Papers)
  - ACL/EMNLP code workshops
  - NeurIPS LLM4Code workshop

- **Conference targets** (only if results clearly beat strong baselines and
  ablation tells a clean story):
  - EMNLP main 2026
  - ACL main 2027
  - ICLR 2027 main

The default plan is workshop. Upgrade to conference if tier-1 results show:
(a) SG ≥ Hybrid-RAG on precision AND tokens, AND (b) ablation shows summary
contributes ≥5% absolute on pass@1, AND (c) consolidation gap shrinks ≥10% with
SG vs Hybrid-RAG.

---

## 9. PyPI tool strategy (parallel track)

- Stay scoped: SG is "the curated, hybrid, summary-aware MCP server for code
  agents." Not the broadest, the smartest.
- Python-first for the paper; add JS/TS/Go/Java/Rust/C++/C#/Ruby/PHP/Kotlin
  (10 langs) post-paper as a v0.2 release. Each is a tree-sitter adapter +
  FQN convention; ~30-50 LOC.
- Target users: people running Cursor/Cline/Continue/Claude Code who want a
  smarter retrieval backend than the default and don't want vendor lock-in.

---

## 10. Anti-drift checklist (read before any new feature)

Before adding anything to SG, ask:
1. Does it serve C1, C2, or C3? If no → reject or defer.
2. Does it require >2 days of work? If yes → revisit after the AMD run.
3. Does it touch the curator, summary tiers, or hybrid routing? If no →
   probably not paper-relevant.
4. Are we adding it because Codebase-Memory has it? If yes → reject (don't
   chase their lane).

---

## 11. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| ContextBench gold annotations are not in our SWE-bench dataset | Backfill from ContextBench's published annotations on the overlap subset |
| 7B can't solve tasks → no pass@1 differentiation in tier 0 | Lean on retrieval-quality metrics (precision, F1) for tier 0; pass@1 only at 32B |
| AMD vLLM serving instability on 32B | Pre-flight with 10-task probe; lock model + flags before main run |
| Curator routing is hand-tuned, not learned | Acknowledge in paper; future-work the learned router |
| Embedding index build time | Document the one-time ~30s cost; it's amortized across all searches in the run |

---

## 12. Definitions of done

**Tier 0 done when**: 6-arm 30-task local run shows SG precision > BM25,
Aider, Hybrid-RAG; no embedding-degradation warnings; verify.py glue tests
all pass.

**Tier 1 done when**: 150-task 4-arm AMD run completes; SUMMARY.md auto-built;
all figures (1-6) produced from JSONs; no integrity warnings.

**Paper done when**: tier 1–4 complete, ablation tells a coherent story
(one component or combination clearly drives gain), all figures reproducible
from the result dumps.
