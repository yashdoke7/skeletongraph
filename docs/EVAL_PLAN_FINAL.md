# SkeletonGraph — Final Evaluation Plan

_Last updated 2026-06-06. Supersedes prior versions. Companion docs: ARM_FLOWS.md
(per-arm flow/arch + "what helps what"), plan.md (compaction-proof roadmap)._

## 1. Thesis (retrieval-first)

**Claim.** In a fixed agent harness, a lightweight structural index (SkeletonGraph)
used to **rerank** cheap lexical retrieval and **fetch at function granularity**
gives the **best retrieval quality (recall + rank) at ~30% lower token cost** than
lexical, dense, and graph-based retrieval/competitors — at a **comparable solve rate**.

**Framing.** The field reduced code context to a *token-optimization* game via graphs,
validated by token-count math only. We invert it: **the goal is better retrieval —
landing the right function — and lower token cost is a consequence**, visible only
end-to-end inside the agent loop. We do **not** claim a significant pass@1 win (on
contaminated benchmarks solve rate is retrieval-insensitive; `none` ≈ 44%). Headline =
**tokens + retrieval correctness** (deterministic, significant); pass@1 = "comparable".

## 2. Method (locked)

- **`sg`** — structural core (entity match + gated graph + centrality rerank + BM25
  weak fallback) with native tools `search_code` + `read_symbol` (fetch ONE function)
  + `expand` (callers/callees). Precision / token-floor operating point.
- **`sg-rerank`** — BM25 recall pool reordered by SG structure ( = "sg + bm25"). The
  recall/rank operating point and overall winner. Report **both** (Pareto pair).

## 3. Arms (canonical set = 15) grouped by ENV

| group | arms | env | notes |
|---|---|---|---|
| **Headline** (6) | `sg`, `sg-rerank`, `bm25`, `grep`, `hybrid`, `none` | **sg-env** | pure-python; main table |
| **Comparators** (3) | `cbmem`, `aider`, `graphify` | see below | native tools, systems comparison |
| **Ablations** (6) | `sg-chain`, `sg-embed`, `sg-seed`, `summary-dense`, `sg-nograph`, `sg-norerank` | **sg-env** | ablation table only |
| (single-shot) | `sg-noagent` | sg-env | run via `run_singleshot.py`, ablation |

**Comparator envs (the constraint that drives window layout):**
- `cbmem` → a **prebuilt binary** (`CBMEM_BIN`, `.exe`/linux-amd64). No python conflict;
  runs from **sg-env**. Per-repo **async index** build.
- `aider` → **own venv** (`.venv-aider`; its huggingface_hub pin conflicts with ours).
- `graphify` → **own venv** (`graphifyy`); needs an LLM endpoint to extract (see §8).

## 4. Models & compute environments

Two environments, both reported:

| Model | Role | Where | Task counts |
|---|---|---|---|
| **Qwen2.5-Coder-32B** | open workhorse, scaled | **AMD MI300X (local vLLM)** | workshop 100 → conference **300** Verified / **150** Pro |
| **nemotron-3-super-120B** | scale / generality | **NIM API** (rate-limited) | **150** (extended from 100) Verified + Pro |

**NVIDIA deprecated Qwen-32B-Coder on NIM** — so the 32B data point can ONLY come from
local serving (AMD; HuggingFace weights are unaffected). That is the *reason* the AMD run
exists. There is no 32B-via-NIM cross-check (model unavailable). 120B is **NOT served on
AMD** (done via NIM). **Real-agent validation:** Claude Code ± SG (MCP) on 30–50 **Pro**
tasks.

## 5. Benchmarks

- **SWE-bench Verified** — 100/150/300 tasks all span **~12 unique repos** (astropy,
  django, sympy, matplotlib, sklearn, xarray, pytest, sphinx, pylint, requests,
  seaborn, flask). *Crucial for cost:* per-repo builds (graphify/cbmem) happen ~12×,
  not per task.
- **SWE-bench Pro** (ScaleAI) — decontaminated, **multi-language** (SG has 9 tree-sitter
  parsers). More repos (~20–40 est.). Needs Scale's verifier; confirm schema on build.

## 6. Metrics

pass@1 (verified) · patch% · **input tokens (uncached, uniform)** · **cost (uncached +
prefix-cached)** + **graphify/cbmem one-time LLM build cost attributed to the arm** ·
turns · **file recall@1/cum, precision, fRank** · **function recall@10, funcHit, fnRank**
(tightened AST gold) · McNemar (paired pass@1) · bootstrap CI · **per-language breakdown
(Pro)**. Figures: `eval/scripts/make_figures.py` (headline = `--arms` filtered to
headline+comparators; ablations separate).

## 7. Compute plan — in depth

### 7a. Timing model (state the assumptions)

- **MI300X, 32B BF16, vLLM:** sustains ~**16–20 in-flight** agent generations. Total
  concurrency across ALL tmux windows should sum to ~16–20 (not per-window) — vLLM is
  the shared bottleneck; CPU-bound arms interleave to keep the GPU fed.
- **Sustained throughput:** ~**0.18–0.22 tasks/s ≈ 700 tasks/GPU-h** (ReAct ≤40 turns,
  most tasks finish 6–15 turns).

**Per-task wall time (informs per-window workers):**

| arm class | env | workers/window | per-task | one-time / repo (amortized, reused) |
|---|---|---|---|---|
| headline+ablation (sg-env) | sg-env | 8–12 | 60–110 s | SG/bm25/hybrid index 5–30 s |
| `cbmem` | sg-env + .exe | 4 | 90–150 s | async graph index 1–5 min |
| `aider` | aider-venv | 4 | 120–240 s | repo-map build 10–30 s |
| `graphify` | graphify-venv | 4 | 90–130 s | **extract built ONCE (§8), reused** |

**One-time graphify/cbmem build totals** (per-repo, reused across every task count AND
both agent models): graphify extract ≈ ~150 LLM calls/repo avg (astropy 381; smaller
fewer) × (~12 Verified + ~30 Pro) ≈ **~6k calls, ~1.5–2.5 GPU-h on local 32B, ONCE**.
cbmem index ≈ **~0.5 GPU-h** total. Both are paid once and reused.

### 7b. AMD 32B — WORKSHOP tier (both benchmarks, 100 tasks)

| set | arms | runs (×100×2 bench) | ~GPU-h |
|---|---|---|---|
| headline + comparators | 9 | 1800 | ~3.0 |
| ablations | 6 | 1200 | ~1.8 |
| graphify graph prebuild (~42 repos, once) | — | — | ~2.0 |
| cbmem index prebuild | — | — | ~0.5 |
| smoke (10 tasks, the gate) | 6 | ~120 | ~0.2 |
| **Workshop subtotal** | | ~3120 | **~7–9 GPU-h** |

### 7c. AMD 32B — CONFERENCE tier (scale headline+comparators; ablations stay 100; + Claude Code)

| set | runs | ~GPU-h |
|---|---|---|
| headline+comparators extra: (300−100)×9 Verified + (150−100)×9 Pro | 2250 | ~3.3 |
| Claude Code ± SG, 30–50 Pro (external frontier API) | ~80 | ~0 (API) |
| **Conference subtotal** | | **~4–6 GPU-h** |

### 7d. AMD budget rollup

Workshop ~8 + conference ~5 + (model download + setup + smoke + verify-overlap +
restarts/buffer) ~6 = **~19 GPU-h of 50.** **Headroom ≈ 31 GPU-h.**

**Top-tier headroom additions (spend the surplus, step by step):**
1. **7B scaling point** (Qwen2.5-Coder-7B) on the headline arms → a model-size curve
   (7B/32B/120B) — strong "does the win hold across scale" result. ~3 GPU-h.
2. **Verified-500** for the headline arms → tighter recall/token CIs. ~2 GPU-h.
3. **3-seed variance appendix** on 4 arms × 20 tasks → serving-noise floor. ~2 GPU-h.
4. **More Claude Code** (50→100 Pro) — API, ~0 GPU.
5. Deeper **per-language Pro** coverage if Pro has long-tail languages.

### 7e. NIM — 120B (+ small 32B cross-check), 150 tasks

Rate-limited → budget in **$ + wall-clock**, not GPU-h. Use 4-key rotation
(`SG_EVAL_API_KEYS`). graphify graphs are **built on AMD-32B and copied** to the NIM
machine (tar `*/graphify-out/`) — so NIM never pays the 381-calls extraction.

| run | arms | runs | priority |
|---|---|---|---|
| 120B × {Verified, Pro} × 150 | headline 6 + comparators 3 | ~2700 (reuse done Verified-100) | **primary** |
| 32B × Verified × 100 | headline 6 | 600 | cross-check (optional) |

Trim to 120B-Verified-150 + 120B-Pro-150 headline+comparators if budget is tight; the
120B numbers anchor the "scale/generality" column, AMD-32B anchors the scaled column.

## 8. graphify's extraction model — per-environment (NIM deprecated 32B-Coder)

graphify needs an LLM to BUILD its graph (a per-repo `graph.json`, NOT part of the agent
loop). The extraction model is a property of the graphify **index**, held separate from
the agent. Build it **once per repo** and reuse across task counts. There is **no
"120B-built graph"** — 120B only *reads* the graph. State exactly that in the paper.

**Models (best-available competent open instruct model per environment):**
- **AMD run:** the **local 32B-Coder vLLM** (HF weights — NIM's deprecation does not affect
  local serving). Free, fast, no rate limits; build graphs on the box.
- **NIM/laptop run:** **`meta/llama-3.3-70b-instruct`** (best latency of NIM's options;
  Gemma4 too slow; **Qwen-32B-Coder deprecated on NIM**). The agent is nemotron-120B, so
  using a *different, non-agent* model for extraction is actually cleaner — it removes any
  agent↔graph confound.

**Reviewer-defensible because:**
1. Both 32B-Coder and Llama-70B are competent open extractors — neither nerfs graphify
   (a 7B would risk hollow JSON → "you crippled the baseline"). Both are ≥ graphify's own
   native tier (its defaults: gpt-4.1-mini, gemini-flash, deepseek-flash, kimi-k2).
2. Reproducible (open weights), unlike a closed flash model.
3. The per-repo extraction (~150 calls avg; astropy 381) is paid **once per repo** (~12
   Verified + ~30 Pro), not per task, not per agent model. Each environment builds its own
   graphs locally (no cross-machine copy).

**Cost attribution:** the one-time extraction (calls / tokens / $ per repo) is **counted
against graphify** in the cost column — a per-repo LLM build SG (zero-LLM tree-sitter)
never pays. This is a *finding*, not a flaw. Paper line: "graphify graphs were built with
the best-available open instruct model in each environment (Llama-3.3-70B on NIM,
Qwen2.5-Coder-32B on AMD)."

**Routing (no structural change to graphify):** its `ollama` backend slot is a generic
OpenAI-compatible path. On AMD point it at the local vLLM; the wrapper passes
`--backend ollama` explicitly so a stray cloud key can't hijack extraction:
```
OLLAMA_BASE_URL=http://127.0.0.1:8000/v1  OLLAMA_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
OLLAMA_API_KEY=EMPTY  GRAPHIFY_OLLAMA_PARALLEL=1
```
If a target endpoint 422s on the ollama `extra_body` (num_ctx/keep_alive), fall back to a
vendored 6-line `BACKENDS["nim"]` entry + `GRAPHIFY_BACKEND=nim`. Selftest before any run.

## 9. Run order, batching & tmux windows (AMD)

**Batches (paste as separate tmux sessions/windows):**
1. **Setup batch** — pull, deps, model weights, build datasets, install cbmem/graphify/
   aider envs, **prebuild graphify graphs (local 32B)**, prebuild cbmem indexes.
2. **Smoke batch** — 10 tasks (a slice of the workshop set) = THE GATE.
3. **Workshop batch** — baselines+comparators in parallel windows, then ablations.
4. **Conference batch** — same windows, increased tasks (300/150), headline+comparators.
5. **Verify (rolling)** — after each arm-set finishes, kick its Docker verify in a
   background window while the next set runs (CPU/Docker, doesn't touch the GPU).

**Window layout (total in-flight ≈ 16–20):**
| win | content | env | workers |
|---|---|---|---|
| 0 | vLLM 32B server | — | — |
| 1 | headline+method (sg/sg-rerank/bm25/grep/hybrid/none) | sg-env | 8 |
| 2 | `cbmem` | sg-env + .exe | 4 |
| 3 | `graphify` (graphs prebuilt) | graphify-venv | 4 |
| 4 | `aider` | aider-venv | 4 |
| 5 | **then** ablations (6) as one stretch (windows 1–4 freed) | sg-env | 12–16 |
| 6 | rolling verify + monitor (rocm-smi, show_progress) | — | — |

Rationale: baselines (GPU) + comparators (CPU-heavy: cbmem index, aider context,
graphify graph queries) run **concurrently** so CPU work interleaves with GPU
generations and vLLM stays saturated (~16–20 in-flight). Ablations are pure-GPU →
run them **after**, alone, at higher workers. Conference repeats with bigger `--n`.

## 10. Language support (Pro)

SG ships 9 tree-sitter parsers (py, js/ts, java, go, rust, cpp, csharp, ruby, php). Pro
is multi-language and **in scope**; report a **per-language breakdown** as a capability
result. Hold the caveat: no multi-language *generalization* claim beyond tasks run;
selftest SG indexing on one repo per language before trusting that language's recall.

## 11. What to push / set up on AMD

**Push to GitHub (small):** code only. Datasets + repos + graphify graphs are built **on
the box** (gitignored). **Pull on AMD, then:** install deps + vLLM, download 32B weights,
build Verified+Pro datasets (`make_dataset`), install cbmem binary + graphify-venv +
aider-venv, **prebuild graphify graphs (local 32B) + cbmem indexes**, run smoke gate.
Push back **only `eval/results/`** JSONLs (verify locally if AMD lacks Docker).

## 12. Paper-readiness checklist

2 models (32B AMD + 120B NIM) ✓planned · 2 benchmarks (Verified + Pro multi-lang) ⧗ ·
scale 32B→300/150 ⧗ · McNemar ✓ · ablation table (ARM_FLOWS) ✓ · Claude Code ± SG ⧗ ·
graphify/cbmem build-cost attributed ✓planned · per-language Pro ⧗ · clean artifact +
README + MCP ⧗ · honest limitations (contamination, n.s. pass@1) ✓ · figures ✓.
