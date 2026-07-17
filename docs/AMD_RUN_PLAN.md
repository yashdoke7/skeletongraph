# AMD MI300X Run Plan — arms, n, and time budget

_Last updated 2026-07-15. Companion to `eval/scripts/amd_run.sh` (v7) and
`docs/EVAL_PLAN_FINAL.md` (the original per-arm timing model this refines).
Read this before spending any AMD Developer Cloud credit._

## 1. Hardware

**AMD Instinct MI300X** — 192 GB HBM3, ~5.3 TB/s memory bandwidth, ~1.3 PFLOPS
BF16. One GPU is enough (don't request multi-GPU — wastes credit; a 32B model
fits on one with huge headroom).

Compared to the RTX 4070 (12 GB, ~504 GB/s, ~29-46 TFLOPS) this project's dense
prewarm was measured on: **~10x memory bandwidth, ~25-45x compute, ~16x VRAM.**
For a small model (the 161M-param embedder), bandwidth + batch size matter more
than raw FLOPS — expect a real but sub-linear speedup, not a 25x one.

## 2. What runs where (models × environments)

| Model | Role | Environment | Notes |
|---|---|---|---|
| Qwen2.5-Coder-32B-Instruct (bf16) | scaled open workhorse | **AMD MI300X, local vLLM** | the only 32B data point — NVIDIA deprecated Qwen-32B-Coder on NIM |
| nemotron-3-super-120B (MoE, 12B active) | scale/generality | **NIM API** (rate-limited) | keep n LOW — real $ + rate limits, no free-compute headroom |
| Claude Sonnet (Claude Code, real MCP) | frontier agent, real deployment | separate account, `run_claude_code.py` | n=50→100-150; real $/task |

AMD serves ONLY the 32B LLM + the dense embedding model (161M params, tiny next
to the LLM — they share the GPU fine, see §6).

## 3. The three benchmarking axes, arms, and n

| Axis | What | Arms | n — what we can afford | n — top-tier standard (irrespective of our budget) |
|---|---|---|---|---|
| **A. Retrieval-only** | deterministic, no LLM calls, model-independent | sg, sg-rerank, bm25, grep, hybrid, fusion, none | **full 500** (Verified) + full Pro — easily affordable, see §7 | **500** (the full official SWE-bench Verified size — this is what "Verified" means in the literature; a subset needs strong justification) + full Pro |
| **B. React loop, 32B (AMD)** | agentic ReAct loop, our harness offers tools directly | **fusion** (SG product, 3-way dense), **cbmem**, **graphify**, **none** (baseline) | fusion/cbmem/none: probe-determined, likely full 500 affordable (§7). **graphify: capped separately** (its extraction is LLM-bound; now fixed to ~2 GPU-h total regardless of n, see §5 — so it can ALSO go to full 500 once the fix is confirmed cheap in practice) | 100-300 is common for agentic multi-arm comparisons in top papers (full-n agent runs across many arms get expensive fast even off-AMD); 500 if affordable is strictly better |
| **B2. React loop, 120B (NIM)** | same loop, nemotron via API | same 4 arms | **n=50-100** — deliberately capped, real $ + rate limits | 100-150 typical for an API-rate-limited scale/generality cross-check point |
| **C. Claude Code deployment** | real frontier agent, real MCP | **sg-fusion vs native** (headline only — competitors structurally excluded/non-adopting, reported separately, never as peer rows) | **n=100-150** (your target) — real $/task on a separate account, no GPU involved | 100+ is already strong for a real-agent deployment study (these are expensive even at frontier-lab scale; 30-50 is common in industry blog posts, 100+ is unusually rigorous) |

**Why full-500 is realistic for Axis A/B despite earlier caution:** the
dominant risk (graphify's LLM-extraction cost) is now FIXED to scale with
**unique repo count (~12 for Verified), not task count** — so it costs the same
whether n=99 or n=500 (§5). The other big cost (LLM serving for the react loop)
is throughput-bound, not repo-count-bound, and the existing timing model (§7)
shows it's comfortably inside budget even at n=500.

## 4. Grounded time-per-unit numbers (what's real vs estimated)

| Cost | Number | Source | Confidence |
|---|--:|---|---|
| Dense embedding encode | **~170-180s/checkout** | measured on user's RTX 4070 (real hardware, this project) | **real, but GPU-specific** — MI300X should be faster (see §6), re-measure in Block 3's probe |
| graphify extraction | **~150 LLM calls/repo avg (astropy up to 381)**, ~1.5-2.5 GPU-h total for ~12 Verified + ~30 Pro repos | `docs/EVAL_PLAN_FINAL.md` §8 (an earlier session's derivation) | previously real, was BROKEN by the worktree-per-task architecture change, **now fixed** (§5) to restore this exact economics |
| React-loop sustained throughput | **~0.18-0.22 tasks/s ≈ 600-700 (task,arm) runs per GPU-hour**, ~16-20 concurrent in-flight generations | `docs/EVAL_PLAN_FINAL.md` §7a | derived from the harness's own turn-count/timeout model, not yet measured on THIS exact stack — Block 3's probe confirms |
| Per-arm-class wall time | headline/fusion 60-110s/task (8-12 workers); cbmem 90-150s/task (4 workers); graphify 90-130s/task-in-loop (4 workers, extraction separately amortized) | same source | same caveat |
| cbmem indexing | ~0.5 GPU-h total (static, non-LLM) | same source | high confidence — tree-sitter based, not LLM-bound |

## 5. The graphify fix (real code change, done this session)

**Problem found:** `eval/scripts/graphify_prebuild.py`'s own docstring claimed
"12 unique repos = 12 builds, reused across every task count" — but its dedup
logic keyed on `repo_path`, which under the current `make_dataset.py` is
**unique per task checkout** (10 django tasks = 10 different directories, since
SWE-bench worktrees each task at its own base_commit). Without a fix, this
would have silently run **one full LLM extraction per task** (up to n=500) —
an 8-10x cost blowup that could have consumed most of the 50-hour budget.

**Fix applied:** `graphify_prebuild.py` now groups tasks by the dataset's
`repo` field (e.g. "django/django"), extracts ONCE on one representative
checkout per repo name, then copies `graphify-out/graph.json` into every
sibling checkout of that repo. Restores the original ~150-calls/repo, ~2-GPU-h
total economics — and because SWE-bench Verified's ~500 instances span only
~12-17 unique repos total, **this cost is now roughly CONSTANT regardless of
whether n=99 or n=500** (only the copy step scales, and that's a cheap
filesystem copy, not an LLM call).

_Tradeoff being made: a graph built at one task's commit is reused for other
tasks of the same repo at different (nearby) commits. Justified because
SWE-bench issues are small, targeted bug fixes — a repo's overall
structure/call-graph doesn't meaningfully change between two of its task
commits. This is the same tradeoff the original per-repo design already
accepted before the worktree change broke it._

## 6. GPU/VRAM utilization plan — what runs parallel vs serial

**Serial (must happen in order):**
1. Weights + embedder download (network-bound, ~20-40 min)
2. vLLM server up (~90s + warmup)
3. Dense prewarm + graphify prebuild for the current dataset (real GPU/LLM work)
4. **Mandatory probe** (Block 3) — 5-task/4-arm timed react run + VRAM check
5. Decision point — compute affordable n from the probe's real numbers, re-freeze dataset bigger if supported

**Parallel candidates (full GPU utilization without contention):**
- `--gpu-memory-utilization 0.85` (dropped from a naive 0.90) deliberately
  leaves ~15-20 GB VRAM headroom so the embedding model (161M params, tiny
  activations even at batch_size=64) can run **concurrently** with vLLM's
  32B serving — confirm this empirically in Block 3(c) via `rocm-smi` before
  relying on it for the full run.
- Once confirmed: dense prewarm / graphify prebuild for the NEXT dataset slice
  (e.g. Pro) can run **while** the react loop is mid-flight on Verified.
- The retrieval-only axis (Axis A) reuses Block 4's exact checkouts/caches —
  no separate clone or encode pass, safe to run concurrently once its own CPU
  load doesn't starve the react loop's worker processes.
- `SG_DENSE_BATCH_SIZE` (new env var, was hardcoded to 8) set to 64 as a
  starting point — bump to 128/256 if Block 3 shows no OOM; this is close to
  free throughput on a 192GB card.

**What must NOT run concurrently:** don't run the model-scale ablation (7B/72B,
Block 7) while the 32B serves the main react loop — swapping the served model
requires killing and restarting vLLM.

## 7. Total time-budget estimate (n=500 Verified + Pro, full axes)

Using §4's grounded numbers (aggregate throughput model + the graphify fix +
the 4070-anchored dense estimate, corrected for MI300X's bandwidth/compute
advantage — all to be CONFIRMED by Block 3's probe before committing):

| Phase | Estimate |
|---|--:|
| Setup (weights, deps, cbmem/graphify Linux install) | ~1-1.5h |
| Dense prewarm, Verified(500)+Pro, parallel-sharded | ~1-1.5h |
| graphify extraction (FIXED — ~12-17 unique repos, one-time, roughly n-independent) | ~2-2.5h |
| cbmem indexing | ~0.5h |
| React loop, 32B, 4 arms × n=500 Verified (+ proportional Pro) | ~3-4h |
| Retrieval-only scoring, full Verified+Pro (reuses warm caches) | ~0.5-1h |
| Rolling verify (Docker/CPU, overlaps with the above) | ~1-2h (mostly free, concurrent) |
| Buffer (~20%) | ~2-3h |
| **Subtotal** | **~11-16 GPU-h** |
| **Headroom in the 50h budget** | **~34-39 GPU-h** |

**What to spend the headroom on, in priority order** (see `project_master_plan`
memory for the full reasoning):
1. Model-scale ablation — 7B + 72B alongside 32B on a 60-task subset
   (fusion+cbmem only) — "SG helps across scale" is a real reviewer-defense
   point, ~3-4h.
2. Push react-loop n even further if the ablation isn't needed.
3. Variance appendix — repeat a 30-task subset ×3 seeds for a documented
   noise-floor CI.

**This is an ESTIMATE, not a promise** — Block 3 of `amd_run.sh` is a
mandatory, non-skippable probe specifically because every number above still
needs real-hardware confirmation before committing the bulk of the 50 hours.

## 8. Dataset state

`eval/datasets/sg_final_100.jsonl` — 99/100 tasks (1 skipped, not chased),
seed 42, **12 unique repos**: astropy, django, matplotlib, xarray, pylint,
scikit-learn, sphinx, sympy, pytest, requests, seaborn, flask. First-50 slice
is the Claude Code matched subset. Re-freeze to n=500 after Block 3's probe
confirms it's affordable (`eval/make_dataset.py --n 500 --seed 42 --out ...`).

**Local dev note:** `make_dataset.py` clones into `eval/datasets/repos/`, a
different location from the pre-existing `swebench-data/repos` cache built up
over earlier react-loop sessions. Set `SG_EVAL_DATA_ROOT` to point at that
existing cache for any future LOCAL regeneration, to reuse the bare git clone
instead of re-fetching from GitHub. Does not apply to the AMD cloud instance
itself (fresh disk regardless).

## 9. $100 AMD Developer Cloud credit — re-request

Was granted once, went unused because the eval pipeline was still being
rebuilt (a follow-up email was already sent about this). Re-request framing:
pipeline is now finalized and ready to consume the full 50 hours productively.
Affiliation type: Independent Developer / Open-Source Contributor. Profile 1
(compulsory): GitHub (`github.com/yashdoke7/skeletongraph`). Profile 2
(optional): PyPI package and/or Google Scholar profile, whichever is live —
both are credibility signals for research-compute requests.
