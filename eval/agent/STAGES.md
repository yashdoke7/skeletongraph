# Staged Evaluation Plan

Budget: AMD Developer Cloud credit — **$100 ≈ 50 hours** on one MI300X. The
T&C wording ("roughly equivalent to 50 hours") confirms billing is **time-based**,
not compute-based: every wall-clock hour the box is up costs the same.

## Principle — breadth over task count

A first instinct is to stage by *quantity*: 150 SWE-bench tasks for a workshop,
500 for top-tier. That is the **worst** use of marginal compute. 150 stratified
tasks already give enough statistical power for paired significance tests
against strong baselines — going to 500 mostly buys leaderboard-number
comparability, which reviewers do not reject a paper for lacking.

So the plan is inverted:

- **SWE-bench task count is FIXED at 150** (`config.SWEBENCH_N`) for every stage.
- Stages add **different kinds of evidence** — strong baselines, ablations,
  significance, a second benchmark, model scaling — ordered by **paper value**.
- Each stage is a clean stop point: credits die after stage 1 → workshop paper;
  after stage 2 → conference; after stage 3 → top-tier.

## Evaluation dimensions, ranked by paper value

| # | Dimension | Why it matters | Stage |
|---|---|---|---|
| 1 | Core controlled result (SG vs floor baselines) | no paper without it | 1 |
| 2 | Strong baselines (hybrid RAG, Aider repo-map) | else "beats strawmen" → reject | 2 |
| 3 | Ablations (which SG component carries the gain) | reviewers always ask | 2 |
| 4 | Variance + significance (repeats, McNemar/CIs) | one run is not a result | 2 |
| 5 | Measured serving density (Axis 4, real vLLM sweep) | the systems differentiator | 2 |
| 6 | Second benchmark (generalisation) | breadth — top-tier expectation | 3 |
| 7 | Model scalability (gain holds on a 7B too) | defends "not model-specific" | 3 |

Scaling SWE-bench 150→500 would rank *below* all of these — hence it is demoted
to optional (only if compute is left over; see the budget table).

## The three stages

| Stage | Tier unlocked | Dimensions added | config keys |
|---|---|---|---|
| **1** | Workshop | 1 | `1-core` |
| **2** | Conference | 2, 3, 4, 5 | `2-strong`, `2-ablation`, `2-variance` + `vllm_bench.py` |
| **3** | Top-tier | 6, 7 | `3-benchmark`, `3-scale` |

## Compute budget

**Sweep unit** = 150 SWE-bench tasks × 1 arm × 1 model, **GPU inference only**.
Patch verification (running test suites) is CPU/Docker — do it **off-box** on a
laptop or a cheap CPU instance so the 50 GPU-hours are spent purely on the model.

Per-sweep cost is unknown until the 5-task probe. Planning band:
**optimistic ≈ 1.5 h/sweep · conservative ≈ 3.5 h/sweep.**

| Phase | Sweeps (≈) | Optimistic | Conservative | Cumulative (cons.) |
|---|---|---|---|---|
| Setup + 5-task probe | — | 2 h | 3 h | 3 h |
| **Stage 1** — 4 arms | 4.0 | 6 h | 14 h | 17 h |
| **Stage 2a** strong — 2 arms | 2.0 | 3 h | 7 h | 24 h |
| **Stage 2b** ablation — 3 arms | 3.0 | 4.5 h | 10.5 h | 34.5 h |
| **Stage 2c** variance — 3 arms × 3, 60-task | 3.6 | 5.5 h | 12.5 h | 47 h |
| **Stage 2** Axis-4 vLLM sweep | flat | 2.5 h | 3 h | 50 h |
| **Stage 3a** 2nd benchmark — 3 arms | 3.0 | 4.5 h | 10.5 h | 60.5 h |
| **Stage 3b** model-scale — 2 arms, 60-task | 0.8 | 1.2 h | 3 h | 63.5 h |
| **Total** | ~16.4 | **~29 h** | **~63 h** | |

**Verdict:**
- **Stages 1 + 2 fit in either case** (~21 h optimistic, ~50 h conservative).
- **Stage 3 fits only if throughput is good.** If the probe shows conservative
  numbers, Stage 3 is trimmed (see cut order).
- The probe (first ~1 h on the box) tells you which world you are in.

**If optimistic** (~29 h used) → ~20 h spare: add the full-500 SWE-bench run for
leaderboard comparability, or a 3rd model-scale point, or more variance repeats.

## Cut order (if the box is slow or the credit shrinks)

Cut from the bottom — each cut still leaves a coherent paper:

1. Stage 3b (model-scale) — drop first; least load-bearing.
2. Stage 3a (2nd benchmark) — drop → you keep a strong conference paper.
3. Stage 2c variance → shrink 60→30 tasks or 3→2 repeats.
4. Stage 2b ablation → 3 ablations → 1 (graph only).

**Never cut:** the 150-task count, any floor baseline, or the two strong
baselines. Those are what make it credible at all.

## Parallelism

Billing is time-based, so the goal is to keep the GPU saturated every paid hour.

- **Within one model:** vLLM continuous batching already runs ~8–12 agent
  workers concurrently against one server. The sweep estimates above *assume*
  this. There is no further "run two sweeps at once" win on a single GPU — the
  GPU is the bottleneck; two sweeps just take 2× as long each.
- **The one real extra-parallel win:** a 7B model uses only ~15–20 % of an
  MI300X. **Co-host the 7B alongside the main 32B on the same GPU** and run
  Stage 3b (model-scale) concurrently with Stages 1–2 — it costs almost no
  extra wall-clock. This effectively makes Stage 3b free.
- **If AMD ever grants a multi-GPU instance:** then one model per GPU is true
  parallelism and halves wall-clock — but plan for a single MI300X; treat
  multi-GPU as a bonus.

## Execution order on the box

```
0. setup + 5-task probe                         (decides optimistic vs conservative)
1. Stage 1  (run to completion — the must-have)
2. Stage 2a strong baselines
   Stage 2b ablations          ← co-host 7B here, run Stage 3b in parallel
   Stage 2c variance
   Axis-4 vLLM throughput sweep
3. Stage 3a second benchmark
   Stage 3b model-scale        ← already done if co-hosted in step 2
```

Run each stage to completion before the next — a crash then leaves a *complete*
stage, never two half-stages.

## Running a stage

```bash
bash eval/agent/serve_model.sh                          # start vLLM
python -m eval.agent.run_stage --stage 1-core --probe   # 5-task timing probe
python -m eval.agent.run_stage --stage 1-core --workers 10
python -m eval.agent.verify    --stage 1-core           # off-box if possible
python -m eval.agent.aggregate --stage 1-core
```
