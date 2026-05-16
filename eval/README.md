# SkeletonGraph — Research Evaluation Harness

Research-grade evaluation for the paper:
**"Knowledge-Aware Structural Retrieval for Coding Agents: Equal Task Success at a
Fraction of the Resident Context."**

The thesis: structural (graph-aware) retrieval achieves equal agentic task success
as long-context stuffing and flat RAG, while resident context tokens drop ~5-10x —
and that linearly multiplies inference serving density (requests per GPU).

This directory is **not shipped in the PyPI package**. It's the eval harness only.

---

## 0. The six evaluation axes

| Axis | Question | Primary instrument | Hardware |
| --- | --- | --- | --- |
| 1. Extrinsic task quality | Does the agent still solve the task? | SWE-bench Verified, pass@1 | Cloud API + laptop (verify) |
| 2. Intrinsic retrieval | Did we retrieve the right code? | ContextBench + SWE-bench gold | Laptop (CPU) |
| 3. Context efficiency | How many tokens did we spend? | tiktoken per-query log | Laptop (CPU) |
| 4. Systems / serving | What does that cost to serve? | KV-cache calc + vLLM throughput | Laptop + RunPod (6 hr) |
| 5. Cost | $ per *passing* task? | API billing, cached + uncached | derived |
| 6. Ablations | Which component earns its keep? | controlled harness sweeps | Laptop + API |

**Baselines — tiered. SG's credibility depends on beating the STRONG tier,
not the floor. Beating only grep/flat-BM25 is strawmanning and reviewers will
say so.**

| Tier | Backends | Required by |
| --- | --- | --- |
| Floor (reference) | `grep` (keyword grep), `bm25` (flat BM25) | Stage 0 GO/NO-GO only |
| Strong RAG | `hybrid` (BM25+dense fusion + cross-encoder/LLM reranker), `dense` (SOTA code embedder) | **Stage 1 — mandatory** |
| Deployed graph competitor | `aider_map` (Aider repo-map: tree-sitter + PageRank — closest real competitor) | **Stage 1 — mandatory** |
| Graph SOTA | CODEXGRAPH / GraphCoder (if reproducible) | Stage 2 |
| No-retrieval | `longctx` (dump whole files, no retrieval) | all stages — the cost baseline |
| Agentic | `native` (agent greps/reads on its own) | controlled-harness conditions |

Stage 0 may run only {grep, bm25, sg} — it answers "is there a signal." The
workshop paper MUST show SG vs `hybrid` and SG vs `aider_map`.

---

## 1. Hardware plan

Your machine: RTX 4070 8 GB (laptop) + i9-14900HX (24 cores).

| Runs on the laptop | Runs on RunPod (rented) |
| --- | --- |
| SG indexing, retrieval, intrinsic eval (Axis 2) | vLLM throughput sweep (Axis 4 measured) |
| Context-token logging (Axis 3) | — that's it — |
| SWE-bench patch **verification** (Docker, CPU) | |
| KV-cache analytical calc (Axis 4 analytical) | |
| Ollama Tier-0.5 summary generation | |
| All aggregation + figures | |

**Why not the 4070 for serving benchmarks:** 8 GB VRAM can't hold a realistic
model + meaningful KV cache. You'd benchmark a toy. Rent an **A100-80GB** on
RunPod for ~6 hours (~$2/hr → ~$12, budget $30-50 with setup overhead) and run
the throughput sweep once. Everything else is CPU and runs on the laptop.

**Disk:** SWE-bench per-instance Docker images are ~1-2 GB each. For a 150-task
subset budget ~120 GB; prune images as tasks verify (`docker image prune`).

---

## 2. Cost & time budget

| Item | Cost | Wall time |
| --- | --- | --- |
| Agent runs: 150 tasks x 4 backends, Haiku-class model | $150-300 | ~1 week (parallelized) |
| — same with Sonnet-class on a 30-task quality subset | +$100-200 | overlaps |
| RunPod A100, throughput sweep | $30-50 | 1 day |
| SWE-bench verification (laptop Docker) | $0 | 1-2 days |
| Intrinsic retrieval eval (laptop) | $0 | 2-4 days |
| **Total spend** | **~$300-550** | |

**Time to a paper:**
- **Workshop paper** (Axes 2-4 + small SWE-bench + analytical KV): **3-4 weeks**.
- **Top-tier** (all 6 axes, full ablations, ContextBench, measured throughput,
  type/test-edge graph): **~3 months** of focused work.

Recommendation: ship the **workshop paper first** — it timestamps the idea and
gets review feedback — then extend to the top-tier venue. Acting fast matters:
ContextBench (Feb 2026) shows the field is already benchmarking this.

---

## 3. SG changes required BEFORE evaluating

These land first — they are both genuine quality levers and the paper's
differentiation vs CODEXGRAPH / GraphCoder (which are call-graph only).

| # | Change | Why it matters for the paper | Effort |
| --- | --- | --- | --- |
| 1 | **Test↔code edges** — link test functions to the functions they exercise | SWE-bench *is* tests; a retriever that surfaces the covering test is a clean novel signal | ~1 wk |
| 2 | **Type/schema nodes** — index dataclass / TypedDict / Protocol / struct / interface as first-class graph nodes | bug fixes need the data model, not just functions | ~4 d |
| 3 | **Per-query context log** — JSONL: query, retrieved FQNs, context tokens (tiktoken), zone breakdown, latency | this IS Axis 3's raw data | ~2 d |
| 4 | **Deterministic eval mode** — frozen summaries, fixed seeds, no network in the retrieval path | reproducibility; reviewers require it | ~2 d |
| 5 | **Standalone baseline retrievers** — `eval/backends/{bm25_flat,dense,grep_sim}.py` | the harness needs all four conditions behind one interface | ~1 wk |
| 6 | Inheritance/`implements` edges + import/module graph | rounds out the "multi-relational knowledge graph" claim | ~4 d |
| 7 | (P1) Git-churn ranking signal | recently-changed code ranks higher for bug fixes | ~3 d |

Track these as issues; #1-#5 are blocking, #6-#7 strengthen the top-tier version.

---

## 4. Phase-by-phase runbook

### Phase A — Environment (once)

```bash
# Laptop: SG + eval deps
cd skeletongraph
python -m venv .venv && . .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[llm,embeddings,eval]"
pip install datasets swebench                          # SWE-bench harness + HF datasets

# Docker (for SWE-bench verification) — confirm it runs
docker run --rm hello-world
```

Reproducible eval image (optional but recommended for the paper's artifact):

```dockerfile
# eval/Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app/skeletongraph
RUN pip install -e "/app/skeletongraph[llm,embeddings,eval]" && pip install datasets swebench
ENTRYPOINT ["bash"]
```

```bash
docker build -t sg-eval:0.1 -f eval/Dockerfile .
```

### Phase B — Build datasets

```bash
# B1. Pull SWE-bench Verified, pick a stratified subset (small/med/large repos)
python eval/make_dataset.py --split verified --n 150 \
    --out eval/datasets/swebench_subset.jsonl

# B2. Convert gold patches -> retrieval ground truth (gold_fqns / gold_files)
python eval/make_retrieval_dataset.py \
    --tasks eval/datasets/swebench_subset.jsonl \
    --out   eval/datasets/swebench_retrieval.jsonl

# B3. (P1) Pull ContextBench — it ships human-annotated gold contexts already
python eval/make_dataset.py --source contextbench \
    --out eval/datasets/contextbench_retrieval.jsonl
```

`make_dataset.py` and `make_retrieval_dataset.py` are thin scripts to write
(see §6). Each SWE-bench instance carries `base_commit`, `patch`, `test_patch` —
the `patch` diff tells you exactly which files/functions are gold.

### Phase C — Axis 2: intrinsic retrieval (laptop, CPU, no API)

```bash
for B in sg bm25 dense grep; do
  python eval/retrieval_eval.py \
    --dataset eval/datasets/swebench_retrieval.jsonl \
    --backend $B --k 5 10 20 \
    --out eval/results/retrieval_$B.json
done
python eval/aggregate.py --glob "eval/results/retrieval_*.json" \
    --out eval/figures/retrieval_table.md
```

**The money chart:** filter to gold functions whose names do NOT lexically
appear in the issue text — "non-lexical targets". SG (graph) should recall these;
bm25/dense miss them. `retrieval_eval.py` exposes per-task rows for this slice.

### Phase D — Axis 1/3/5: controlled agent harness

The agent is a **fixed minimal ReAct loop** — same model, same prompt, only the
retrieval backend swapped. This isolates SG's contribution (a commercial IDE
would confound it with its own context management).

```bash
# Per backend, per task: agent produces a patch + a per-query context log
for B in native bm25 dense sg; do
  python eval/run_agent.py \
    --dataset eval/datasets/swebench_subset.jsonl \
    --backend $B --model claude-haiku-4-5 \
    --workers 6 \
    --out eval/results/agent_$B/
done
```

`run_agent.py` writes per task: `patch.diff`, `context_log.jsonl`
(tokens per turn), `usage.json` (API tokens + $). Axis 3 and 5 fall out of these.

### Phase E — Axis 1: SWE-bench verification (laptop, Docker, no API)

```bash
for B in native bm25 dense sg; do
  python -m swebench.harness.run_evaluation \
    --predictions_path eval/results/agent_$B/predictions.jsonl \
    --max_workers 8 --split verified \
    --run_id sg_eval_$B
done
docker image prune -f         # reclaim disk between backends
```

Produces pass@1 per backend.

### Phase F — Axis 4: systems / serving

```bash
# F1. Analytical — laptop, instant, no GPU
python eval/kv_cache.py --csv eval/results/kv_cache.csv

# F2. Measured throughput — RunPod A100-80GB, ~6 hr
#   On the pod:
pip install vllm
python eval/vllm_bench.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --context-lengths 8000 16000 35000 64000 128000 \
    --out vllm_throughput.json
#   scp vllm_throughput.json back to eval/results/
```

`vllm_bench.py` sweeps context length, measures tokens/sec, TTFT, and max
concurrent requests — the *measured* counterpart to `kv_cache.py`'s analytical
curve. Put both on one plot: they should agree, which validates the model.

### Phase G — Aggregate + figures

```bash
python eval/aggregate.py --all --out eval/figures/
```

Produces the paper's tables and the **headline figure**: a pass@1-vs-context-tokens
Pareto frontier with SG's curve dominating all baselines.

### Phase H — Ablations (Axis 6)

Re-run Phases C-E with SG variants: `--ablate graph` (no edges), `--ablate
test-edges`, `--ablate summaries`, `--ablate pagerank`, `--ablate zones`, plus a
`--budget` sweep (2k → 64k) for the Pareto curve.

---

## 5. Smoke test (do FIRST — before any full run)

```bash
# 1 easy task, all 4 backends, end to end. ~30 min.
python eval/make_dataset.py --split verified --task-ids pytest-dev__pytest-5103 \
    --out eval/datasets/smoke.jsonl
python eval/make_retrieval_dataset.py --tasks eval/datasets/smoke.jsonl \
    --out eval/datasets/smoke_retrieval.jsonl

python eval/retrieval_eval.py --dataset eval/datasets/smoke_retrieval.jsonl --backend sg --k 5 10
python eval/run_agent.py --dataset eval/datasets/smoke.jsonl --backend sg --model claude-haiku-4-5
python eval/kv_cache.py --model qwen-coder-7b
```

Checklist: SG index builds · retrieval returns non-empty · agent produces a
non-empty diff · context_log.jsonl has token counts · KV calc prints a table.
Fix anything broken here before scaling to 150 tasks.

---

## 6. What's built vs what to build

| File | Status | Notes |
| --- | --- | --- |
| `eval/kv_cache.py` | **done** | analytical KV + serving density; runs now |
| `eval/retrieval_eval.py` | **done** | metrics + SG backend live; baseline backends are imports |
| `eval/backends/bm25_flat.py` | TODO (SG change #5) | reuse SG's BM25, disable graph + centrality |
| `eval/backends/dense.py` | TODO | code-embedding model + cosine; `sentence-transformers` |
| `eval/backends/grep_sim.py` | TODO | keyword grep → ranked files (naive-agent baseline) |
| `eval/make_dataset.py` | TODO | SWE-bench / ContextBench → task jsonl |
| `eval/make_retrieval_dataset.py` | TODO | gold patch → gold_fqns / gold_files |
| `eval/run_agent.py` | TODO | fixed ReAct loop, swappable retrieval backend |
| `eval/vllm_bench.py` | TODO | RunPod throughput sweep |
| `eval/aggregate.py` | TODO | results/*.json → tables + Pareto figure |
| `eval/Dockerfile` | TODO | reproducible artifact (snippet in §4-A) |

`run_agent.py` is the largest piece — keep the ReAct loop minimal: a system
prompt, a `retrieve(query)` tool bound to the chosen backend, an `edit_file`
tool, a `run_tests` tool, capped at N turns. Log every turn's token count.

---

## 7. Automation map

| Fully automated (script + cron-able) | Needs a human / account |
| --- | --- |
| Dataset build, retrieval eval, KV calc | RunPod pod creation (one click) |
| Agent runs (API-driven — no GUI) | API key + budget approval |
| SWE-bench verification, aggregation, figures | reading the final numbers |

Everything except provisioning the RunPod box and approving spend is scriptable.
A single `eval/run_all.sh` can chain Phases B→C→D→E→G once the TODO scripts
exist. The commercial-IDE runs (Claude Code / Cursor in-GUI) are deliberately
**out of the main pipeline** — keep them as a separate 10-15 task "ecological
validity" appendix, run by hand, not part of the reproducible result.

---

## 8. Paper-readiness checklist

- [ ] SG changes #1-#5 landed (test edges, type nodes, context log, deterministic mode, baseline backends)
- [ ] Smoke test green
- [ ] Axis 2 on SWE-bench gold + ContextBench, 4 backends
- [ ] Axis 1 pass@1, 4 backends, verified
- [ ] Axis 3 context tokens logged per task
- [ ] Axis 4 analytical KV + measured vLLM curve agree
- [ ] Axis 5 $/passing-task, with + without prompt caching
- [ ] Axis 6 ablations + budget Pareto frontier
- [ ] Failure analysis: where SG retrieval misses
- [ ] Reproducible: `eval/Dockerfile` + pinned deps + dataset manifest
