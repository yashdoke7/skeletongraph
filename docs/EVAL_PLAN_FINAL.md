# SkeletonGraph — Final Evaluation Plan (AMD-anchored, scale-on-results)

**Version**: v3.2  ·  **Date**: 2026-05-20  ·  **Status**: ratified for execution

> **v3.2 changes**: stronger "why agent" framing with RAG/MemGPT/HierMem refs;
> added Stage IDE (deployed-IDE testing as secondary eval stream); detailed
> "what makes conference acceptance hard" analysis; ablation coherence
> explanation; smoke-script spec for Sonnet to implement.

> Self-contained planning doc. Replaces no other doc; sits next to `RESEARCH_PLAN.md`
> (thesis) and `IMPLEMENTATION_PLAN.md` (engineering work). This doc is about
> **what we run, on what, with what budget, expecting what numbers, targeting
> what venue.** Brutal honest assessment — no optimism baked in.

---

## 0. Executive summary (the brutal version)

- **Target venue**: workshop first (NeurIPS CodeML / MATH-AI, EMNLP findings, or ICSE
  tool track). **Push to conference only if Tier A produces specific result thresholds.**
- **Realistic acceptance probability** with Tier A only:
  - Workshop: **50–65%**
  - Conference main track: **15–20%** (best case)
- **Realistic acceptance probability** with Tier A + Tier B (conditional):
  - Workshop: **70–80%**
  - Conference main track: **25–35%** (and that assumes results land cleanly)
- **Primary infrastructure**: AMD MI300X at \$1.99/hr (Hot Aisle reference rate, parity with
  AMD Developer Cloud credit value). Effective budget after setup + overhead: **~\$90 usable for runs out of \$100 credits**.
- **NIM**: appendix-only. 30 showcase tasks on Llama-3.3-70B for the scalability table.
  Not in the critical path.
- **Local 7B**: smoke, scale-down ablation, debugging — free, slow.

The plan below is **AMD-only for the headline numbers** so every reported number
shares the same model server, the same tokenizer, the same KV-cache behavior,
the same seed. **Reproducibility > model scale.**

---

## 1. The paper's actual contributions (re-stated, post-drift-check)

### What we claim
1. **C1 — Adaptive curator routing**: A query-classifying curator selects between
   retrieval strategies (graph-heavy for refactors, summary-heavy for explains,
   BM25 for lookups). Outperforms any fixed pipeline.
2. **C2 — Three-tier summaries reduce the consolidation gap**: Tier-1 (full body),
   Tier-2 (skeleton + 1-line summary), Tier-3 (FQN-only) assembly leaves agents
   with *useful* context, not noise. Measured directly by the ContextBench-style
   consolidation-gap metric.
3. **C3 — Hybrid signals beat single signals**: BM25 + graph expansion +
   embeddings + centrality reranking, fused, outperforms any one of them.

### What is NOT novel (be honest)
- "Hybrid retrieval" itself — Augment, Voyage, Cohere all do BM25 + dense + rerank.
- "PageRank over symbol graph" — Aider has done this since 2023.
- "Query classification + routing" — exists in many systems (RouteLLM, etc.).
- "Three-tier hierarchical summaries" — borrowed wholesale from HierMem.

### What IS novel (the thin slice we own)
- **The specific combination** of all four signals into one curator-routed pipeline.
- **The consolidation-gap metric** applied to retrieval+agent pipelines as a
  primary evaluation axis. ContextBench introduced this idea (Dec 2025); we apply
  it to retrieval comparison rather than to a single system.
- **Empirical evidence** that summary tiers (not just file selection) drive the
  agent's task-success gain, not just retrieval recall.

### What kind of paper is this, really?
- It is a **systems / empirical** paper, not an algorithmic / theoretical one.
- The contribution is **engineering+measurement**, not a new model or theorem.
- That's fine for: workshops (CodeML, MATH-AI, FMDM), SE venues (ICSE, FSE, ASE),
  applied tracks (EMNLP findings, NAACL findings).
- That's a poor fit for: NeurIPS/ICML/ICLR main tracks (they want new
  algorithms or surprising results).

---

## 2. Target venue analysis (brutal)

### Path A — ML workshop (recommended)
- **Venues**: NeurIPS CodeML, NeurIPS MATH-AI, NeurIPS FMDM, EMNLP findings,
  NAACL findings, ICLR Code/Tool workshops.
- **What they accept**: clean empirical work with a focused contribution, even
  if engineering-heavy. 4–8 pages.
- **What they expect**: 1 benchmark min (we have SWE-bench), reasonable
  ablations, statistical care. ContextBench as second benchmark is gravy, not
  required.
- **Realistic accept rate for our work with Tier A only**: 55–65%.
- **With Tier A + Tier B**: 70–80%.
- **Why this is the honest target**: our contributions are real but incremental;
  workshops are designed exactly for this kind of work.

### Path B — SE conference (alternative)
- **Venues**: ICSE (tool track), FSE/ESEC, ASE.
- **What they accept**: tools that demonstrably help developers, with practical
  ablations and a real artifact.
- **What they expect**: multi-language support (a real tool isn't Python-only),
  reproducible artifact, sometimes a small user study.
- **Realistic accept rate**: 20–25% for full tracks; tool tracks ~30%.
- **Why this is plausible**: SG is shippable. MCP integration is real. The
  PyPI plan is real.
- **Why this is harder**: the multi-language work isn't ablated empirically (we
  said it's "tool viability, not eval"). SE reviewers will push on this.

### Path C — ML main track (don't do this)
- **Venues**: NeurIPS / ICML / ICLR main.
- **What they want**: novel algorithm or theoretical insight or 10× empirical
  jump.
- **What we have**: 3.5× retrieval precision over BM25 in a 5-task smoke.
- **Realistic accept rate**: **<10%** for our work as-is.
- **Recommendation**: don't try. The reframing time would be better spent on
  a stronger workshop submission.

### The honest target: **NeurIPS CodeML workshop or EMNLP findings**.

---

## 3. Acceptance-rate reality, conditioned on results

The acceptance probability is not fixed — it depends on what Tier A actually
produces. Here's the conditional probability table:

| Tier A produces | Workshop accept | Conference accept |
|-----------------|-----------------|-------------------|
| pass@1 gap (SG − BM25) ≥ +8pp **AND** clean ablation separation | 80% | 35% |
| pass@1 gap +5–8pp, ablations mostly clean | 65% | 25% |
| pass@1 gap +3–5pp, mixed ablations | 50% | 15% |
| pass@1 gap +0–3pp, **but** precision/rank improvements are striking | 40% | <10% |
| pass@1 gap 0 or negative (precision improvement doesn't translate) | 20% | <5% |

**The "kill shot" scenario**: retrieval precision/rank improve massively but
pass@1 doesn't. Reviewer says: "you've shown better retrieval doesn't actually
help." This is the **most likely failure mode**, because:
- 7B/32B models are not strong enough to capitalize on perfect retrieval
- An agent that gets the right file may still fail to write the right fix
- Token efficiency arguments don't compensate for "didn't fix the bug"

Mitigation: if pass@1 differential is small, **reframe** as a retrieval-quality
+ consolidation-gap paper, drop the pass@1 claim, lean on the ContextBench
metric. This salvages a workshop submission.

---

## 4. Setup overhead — what \$100 actually buys

### Setup costs (one-time, billed)
| Step | Time | Cost |
|------|------|------|
| Provision MI300X VM + ROCm verification | 30 min | \$1 |
| Install vLLM + flash-attn for ROCm | 1 hr | \$2 |
| Download Qwen2.5-Coder-32B (64 GB BF16) | 30 min | \$1 |
| First vLLM serve + health-check | 30 min | \$1 |
| Run smoke (5 tasks × 4 arms) for end-to-end validation | 1.5 hr | \$3 |
| verify.py setup + SWE-bench harness dry-run | 1 hr | \$2 |
| Inevitable first-try debugging | 1 hr | \$2 |
| **Setup subtotal** | **~6 hr** | **~\$12** |

### Runtime overhead (recurring, ~15–20% of model-runtime)
- vLLM occasional restart on OOM (~once per 50 runs at 32B with 32K context)
- Network blips, retry logic firing
- Workspace cleanup between runs
- Verifier dry-runs

**Effective compute available**: 50 hr total − 6 hr setup = 44 hr.
With 18% overhead on the 44 hr → **~36 hr of pure model runtime** at \$100.

### Run throughput at 32B on MI300X
- Retrieval arms (sg, bm25, grep, hybrid, aider, sg-*): ~4 min/run avg.
- `none` arm: ~8 min/run avg (18 turns × 25s/turn).
- **Mixed throughput (75% retrieval, 25% none)**: ~5 min/run.

**Total runs feasible on \$100**: 36 hr × 60 min / 5 min = **~430 runs**.

---

## 5. Tier A — the workshop-strength baseline (~\$90 of the \$100)

**Goal**: produce the headline table + minimum ablations that get a workshop
paper across the line, with budget headroom for one re-run if something breaks.

### Run grid

| Stage | Arms | Tasks | Repeats | Runs | Mix | Hours @ 32B |
|-------|------|-------|---------|------|-----|-------------|
| **A0** Smoke | 9 | 5 | 1 | 45 | 7B local | 0 (free, laptop) |
| **A1** Headline | sg, bm25, grep, none | 60 | 1 | 240 | 3 retr + 1 none | 60×4×3 + 60×8 = 720+480 = 20 hr |
| **A2** Strong baselines | hybrid, aider | 60 | 1 | 120 | retr | 60×4×2 = 8 hr |
| **A3** Core ablation | sg-nosummary, sg-nograph | 40 (subset of A1's 60) | 1 | 80 | retr | 40×4×2 = 5.3 hr |
| **A4** verify.py (no inference) | — | — | — | — | — | ~1 hr |
| **Total** | | | | **440** | | **~34.3 hr** |

### Budget math
- Setup: 6 hr × \$1.99 = **\$12**
- Runs: 34.3 hr × \$1.99 × 1.18 (overhead) = **\$80.6**
- **Tier A subtotal**: **~\$93** (under \$100 with \$7 buffer)

### What Tier A produces
- **Table 1**: pass@1 + retrieval-precision + rank + cost + turn-count per arm
  across the 6 main arms (sg, bm25, grep, none, hybrid, aider). N=60 each.
- **Table 2**: ablation impact (sg vs sg-nosummary, sg-nograph). N=40 each.
- **Figure 1**: consolidation-gap distribution per arm (the C2 visual).
- **Figure 2**: cumulative-recall curve over search calls per arm.
- **Figure 3**: token efficiency (input tokens vs pass@1) Pareto.
- Significance tests: McNemar paired exact, 95% bootstrap CI on retrieval-hit.

### What Tier A does NOT give us
- Second benchmark (no ContextBench)
- C1 (curator) ablation evidence
- Variance bars (single repeat)
- Scaling evidence (only one model size in main eval)

### Why N=60, not 150 (the brutal answer)
- 150 was wishful at \$100 budget.
- N=60 stratified across the 11 SWE-bench repos = 5–6 tasks per repo, enough
  for paired McNemar to detect a 10pp pass@1 difference with α=0.05 power ≈0.7.
- Smaller N is honest about budget; larger N would require dropping arms.

---

## 6. Tier B — the conference-push expansion (+\$30–40)

**Trigger condition**: Tier A produces results meeting the "go for conference"
thresholds in §7. Otherwise, do NOT spend Tier B credits — write up the
workshop submission with Tier A and submit.

### Run grid

| Stage | Arms | Tasks | Repeats | Runs | Hours @ 32B |
|-------|------|-------|---------|------|-------------|
| **B1** ContextBench | sg, bm25, hybrid | 50 | 1 | 150 | 50×4×3 = 10 hr |
| **B2** C1 ablation | sg-nocurator | 60 (matches A1 set) | 1 | 60 | 4 hr |
| **B3** Mini-variance | sg, bm25, hybrid | 30 (subset of A1) | 2 extra | 180 | 12 hr |
| **B4** NIM scale-up (optional) | sg, bm25 | 20 | 1 | 40 | 0 hr AMD |
| **Total AMD** | | | | **~390 add'l** | **~26 hr** |

### Budget math
- 26 hr × \$1.99 × 1.18 = **~\$61**

That's more than \$30–40. To hit \$30–40 exactly, **trim Tier B to**:
- B1 (ContextBench, 50 tasks × 3 arms = 150 runs, ~10 hr) → ~\$24
- B2 (sg-nocurator, 60 tasks, ~4 hr) → ~\$10
- Skip B3 mini-variance unless extra credits arrive
- B4 NIM is free (appendix only)

**Tier B trimmed**: ~\$34 → fits \$30–40 envelope.

### What Tier B adds
- ContextBench generalization (Table 3, Figure 4 with block-level
  consolidation gap)
- C1 ablation completing the contribution-isolation story (Table 4 row)
- NIM 70B scaling line in the discussion (Table 5)

### What Tier B still doesn't give
- Variance bars (skipped in trimmed version)
- Multi-language eval (out of scope)

---

## 7. Decision rules — what Tier A numbers justify Tier B spend

Run Tier A. Look at these five conditions. **Spend Tier B only if 3 or more
hold** — anything fewer means the conference push is unlikely to land regardless
of how much more compute we burn.

| # | Condition | Why this threshold |
|---|-----------|---------------------|
| **R1** | SG pass@1 − BM25 pass@1 ≥ +5pp (absolute) | Below this, reviewers say "retrieval gain doesn't translate" |
| **R2** | SG retrieval-precision ≥ 3× BM25 retrieval-precision | Confirms smoke holds at N=60; this is our strongest signal |
| **R3** | SG vs sg-nosummary pass@1 gap ≥ +3pp | C2 (the headline claim) has measurable effect |
| **R4** | SG vs sg-nograph pass@1 gap ≥ +2pp OR consolidation-gap difference ≥ 0.15 | C3 has some isolated signal |
| **R5** | SG mean consolidation_gap_files ≤ 0.50; BM25 ≥ 0.70 | Metric is discriminating (else C2 looks like noise) |

### Decision matrix

| Conditions met | Action |
|----------------|--------|
| 4–5 | Spend full Tier B (+\$34). Aim conference (~25% accept). |
| 3 | Spend partial Tier B (B1 ContextBench + B2 nocurator only, +\$34). Aim workshop with conference reach. |
| 2 | Stop. Workshop submission with Tier A only. ~55% accept. |
| 0–1 | Stop. Workshop submission with reframed thesis (drop pass@1 claim, lean on retrieval-quality + consolidation-gap framing). ~35% accept. |

This is the "no optimism" position. The numbers do not lie. If pass@1 doesn't
move, more compute won't help — the agent isn't strong enough to use SG's
better context, and we'd be papering over that with more tasks.

---

## 8. Methodology — defending every choice (for paper §3)

These are the design choices we'll need to defend in the paper. Each has a
one-line answer.

### Q: Why agent-based eval instead of offline retrieval (recall@k, precision@k)?
**A**: Three reasons, with prior-work support:

1. **Static RAG underperforms on code localization.** Pure embedding-based
   retrieval surfaces *topically similar* code (similar names, comments, types)
   but not necessarily *causally relevant* code (the function that's broken,
   the caller that triggers the bug). This is documented in CodeRAG-Bench
   (2024) and "Grep Is All You Need" (2024/2025) — both showed that flat
   embedding RAG underperforms lexical baselines on code retrieval. SG must be
   evaluated in a setting that lets retrieval recover from these failures.
2. **Agents add a feedback loop.** Multi-turn agents issue follow-up queries
   informed by what they read. After viewing file X, the agent can search for
   callers of `X.parse()`. Static retrieval can't. We need the agentic setting
   to test whether SG's per-call retrieval improves over time
   (`cumulative_recall` in our metrics).
3. **Only agents produce patches**, and patches are how we measure the
   consolidation gap. Without a patch we cannot compute
   `files_retrieved_but_unused`. The metric requires the agent loop to run to
   completion.

Reframed positively, this connects to MemGPT (Packer et al., 2023) and HierMem
(prior work by the SG author): both established that **hierarchical context is
necessary for long agentic tasks**. SG transfers this insight from agent
*memory* to agent *retrieval-time context* — the agent's first peek at each
file is a Tier-2 summary, deepening to Tier-1 (full body) on demand. This is
the direct mechanism for C2 and a key reason agentic eval is the right setting.

SWE-bench is also the field standard: SWE-agent, AutoCodeRover, RepoGraph,
CodeR all evaluate agentically. A pure recall@k paper would be off-trend and
weaker as a result.

### Q: Why context assembly (Tier-1/2/3) instead of just returning file paths?
**A**: A retrieval system that returns the right files but assembles them
poorly (raw concatenation) wastes the agent's context window and increases the
consolidation gap. SG's three-tier assembly is the *contribution* — file
selection alone is solved by Aider/BM25. This is C2.

### Q: Why a curator over a single fixed pipeline?
**A**: Different query types want different retrieval strategies. A refactor
query benefits from graph blast-radius; a "where is X defined" query is satisfied
by BM25. A fixed pipeline either over-retrieves (refactor) or under-retrieves
(lookup). Tested by `sg-nocurator`. This is C1.

### Q: Why fuse signals instead of picking one?
**A**: BM25 wins on lexical matches; graph wins on dependency chains; embeddings
win on semantic similarity. Real queries mix all three. Tested by `sg-nograph`,
`sg-nosummary`, and comparison to `hybrid` (which has signals but no curator).
This is C3.

### Q: Why \$0.27/M input / \$1.10/M output reference rates for imputed cost?
**A**: Public reference rates for hosted Qwen2.5-Coder-32B inference circa
2026-Q1. Applied uniformly across arms; arm-to-arm cost ratios are unaffected
by provider choice. Real-world deployment costs on cached APIs would reduce
all arms uniformly by ~30–50%.

### Q: Why don't we measure with prompt caching enabled?
**A**: The cacheable surface (system prompt + tool descriptions ≈ 1.2K tokens)
is identical across arms. Anthropic-equivalent caching (90% discount on cached
input) would reduce each arm's input cost by an identical absolute amount,
preserving relative rankings. Methodology section will state this with the math.

### Q: Why temperature=0, seed=42, max_turns=40?
**A**: Reproducibility (T=0 makes vLLM deterministic on same GPU class). The 40-turn
ceiling matches SWE-agent's default; running to convergence or 40, whichever
comes first.

### Q: Why is there no KV-cache analysis?
**A**: We use local vLLM with automatic KV cache (L1, intra-request). API-level
prompt caching (L2) is orthogonal — see above. SG's embeddings + index are
cached on disk (L3) — already optimized; doesn't need a separate experiment.

### Q: Why N=60 SWE-bench tasks instead of 150 or 500?
**A**: Budget constraint, disclosed. Statistical power: N=60 paired McNemar
detects a 10pp pass@1 difference at α=0.05 with power ≈0.7. Larger N would
require dropping arms — we chose breadth over depth.

### Q: Why no human user study?
**A**: Out of scope for an ML workshop submission. Would be required for
ICSE/FSE tool track — if we redirect to SE venue, add a 10-user study (3-week
post-eval addition).

---

## 9. What we already have on disk (don't rebuild)

These were validated in the smoke + earlier work. Don't redo:

### Smoke results (`eval/results/agent/SUMMARY.md`)
- 5 tasks × 4 arms × Qwen2.5-Coder-7B, T=0
- SG precision 0.32 vs BM25 0.09 vs grep 0.08 — **3.5× headline gap**
- SG median rank = 1 (gold file is first hit)
- No arm achieved edited_gold (sample too small)
- All 4 arms submitted; `none` once max_turns'd at 18 turns avg
- **Use as smoke baseline**; if Tier A 32B numbers degrade vs this, debug the deployment

### Code artifacts (per IMPLEMENTATION_PLAN.md + this session)
- `eval/agent/run_agent.py` — rich per-run metrics (✅ done)
- `eval/agent/aggregate.py` — trajectory + consolidation + search dynamics tables (✅ done)
- `eval/agent/tools.py` — empty-submit guard, edits_made tracking, embeddings_used flag (✅ done)
- `eval/agent/react.py` — text-form tool call fallback (✅ done)
- `eval/agent/isolation.py` — Windows-safe rmtree (✅ done)
- `eval/backends/bm25_flat.py` — BM25 (✅ existing)
- `eval/backends/grep_sim.py` — grep (✅ existing)
- `eval/backends/hybrid.py` — BM25 + dense + cross-encoder (✅ done this session)
- `eval/backends/aider_map.py` — Aider RepoMap wrapper (✅ done this session)
- SG ablation toggles in SGConfig (✅ done) + tests (✅ 9/9 passing)
- `eval/REPRODUCIBILITY.md` (✅ done)

### What still needs building (blockers)
1. **`run_agent.py` model default bug** — defaults `"qwen-32b"` not in `MODELS` dict.
   15-min fix.
2. **`sg-nocurator` ablation** — needed for Tier B B2. Add `enable_curator: bool = True`
   to SGConfig; in `classify_query`, when False, return fixed `MODE_SPECS[BUILD_GUIDED]`.
   Wire in `tools._retrieve`. Add test. ~1 hr.
3. **ContextBench loader** — needed for Tier B B1. `eval/scripts/extract_contextbench.py`
   joining ContextBench gold annotations to SWE-bench instance IDs; output to
   `eval/datasets/contextbench.jsonl` with `gold_blocks` + `gold_lines` schema.
   ~3 hr.
4. **`aider-chat` install** — `pip install aider-chat` + one-task smoke. ~30 min.

Pre-eval prep: **~4–5 hours of coding** before any AMD spend.

---

## 10. The full run-grid (single table for budget allocation)

| ID | What | Where | Arms | Tasks | Repeats | Runs | Cost |
|----|------|-------|------|-------|---------|------|------|
| **A0** | Smoke (validation) | Laptop 7B | 9 | 5 | 1 | 45 | \$0 |
| **A1** | Headline | AMD 32B | sg, bm25, grep, none | 60 | 1 | 240 | ~\$48 |
| **A2** | Strong baselines | AMD 32B | hybrid, aider | 60 (same set) | 1 | 120 | ~\$19 |
| **A3** | Core ablation | AMD 32B | sg-nosummary, sg-nograph | 40 (subset) | 1 | 80 | ~\$13 |
| **Setup + overhead** | | AMD | — | — | — | — | ~\$13 |
| **Tier A subtotal** | | | | | | **440** | **~\$93** |
| **B1** | ContextBench | AMD 32B | sg, bm25, hybrid | 50 | 1 | 150 | ~\$24 |
| **B2** | Curator ablation | AMD 32B | sg-nocurator | 60 (matches A1) | 1 | 60 | ~\$10 |
| **B3** | Mini-variance | AMD 32B | sg, bm25, hybrid | 30 (subset) | 2 extra | 180 | ~\$24 *(only if extra credits)* |
| **B4** | NIM 70B scale | NIM Llama-3.3-70B | sg, bm25 | 20 | 1 | 40 | \$0 (NIM free) |
| **Tier B trimmed** | | | | | | **+250** | **~\$34** |
| **Tier B full** | | | | | | **+430** | **~\$58** |

**Total Tier A** : 440 runs, ~\$93 AMD.
**Total Tier A + B trimmed**: 690 runs, ~\$127 AMD.
**Total Tier A + B full**: 870 runs, ~\$151 AMD.

---

## 11. Expected results — the targets we're trying to hit

Based on the 7B smoke results and reasonable extrapolation to 32B, here's what
"good" looks like. **If actual numbers diverge significantly, debug before
proceeding deeper.**

### Tier A headline table — expected numbers

| Arm | Expected pass@1 | Expected retrieval-precision | Expected consolidation_gap_files |
|-----|----------------|------------------------------|----------------------------------|
| none | 5–10% | 0.0 (no retrieval) | n/a (no read context) |
| grep | 8–15% | 0.05–0.10 | 0.75–0.85 |
| bm25 | 12–20% | 0.08–0.15 | 0.65–0.80 |
| hybrid | 18–28% | 0.20–0.35 | 0.45–0.60 |
| aider | 18–28% | 0.20–0.35 | 0.50–0.65 |
| **sg** | **20–32%** | **0.30–0.45** | **0.35–0.55** |

If `sg` lands at the top of these ranges, R1+R2+R5 are met → push Tier B.
If `sg` lands at the bottom of these ranges (≤22% pass@1), only R2 likely
holds → workshop only.

### Tier A ablation expectations
- `sg-nosummary` ≈ midway between `sg` and `bm25` on pass@1 (so ~16–24%)
- `sg-nograph` ≈ midway between `sg` and `bm25` on retrieval-precision (~0.18–0.30)
- If ablations land too close to `sg`: the component being ablated has small
  effect, and we should reconsider the contribution framing.
- If ablations land too close to `bm25`: the component does most of the work,
  and our story is stronger than expected.

### Worst-case salvage
If `sg` underperforms (pass@1 ≤ `bm25` + 2pp):
- **Reframe paper**: "Retrieval-quality-aware ablation study of code-editing
  agents" instead of "SG outperforms baselines."
- Lean on retrieval-precision and rank metrics, which we expect to hold.
- Add ContextBench (B1) to anchor the consolidation-gap claim — that's our
  most defensible result.
- Submit to a workshop, accept lower ceiling.

---

## 12. Risk register (what could go wrong)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Setup eats more than 6 hr (e.g., ROCm + vLLM friction) | Medium | -\$10 budget | Buffer is \$7. Test setup script on a free-tier VM first. |
| Qwen2.5-Coder-32B context >32K config fails on MI300X (KV cache OOM) | Low-Medium | `none` arm breaks | Use 16-bit KV cache; if still fails, cap `none` at 28K input + summary, document as limitation |
| pass@1 differential is small (<3pp SG vs BM25) | **Medium-High** | Conference push dead, paper weakens | Reframe to retrieval-quality + consolidation paper, accept workshop tier |
| `aider-chat` install breaks on Windows ROCm box | Medium | Aider arm broken | Skip aider arm or run via subprocess; document as limitation |
| `sg-nocurator` is functionally equivalent to `sg` (curator does nothing measurable) | Medium | C1 ablation has null result | Honest: drop C1 from contributions in the paper, lean on C2 + C3 |
| ContextBench dataset has join issues with SWE-bench IDs | Low-Medium | B1 delayed | Run B1 on raw ContextBench tasks if join fails, drop the cross-benchmark direct comparison |
| verify.py harness fails on some tasks (Docker dependencies, etc.) | High | Some pass@1 values uncertain | Report N_evaluated separately; SWE-bench typically loses 5-10% of tasks to harness issues |
| vLLM crashes mid-run, requires restart | Medium | +1-2 hr overhead | Already in 18% buffer |
| 5-day AMD credit window expires mid-eval | Medium-High | Budget cut short | Plan tight schedule; do Tier A within 2-3 days of credit redemption, decide on B within 24hr after |

---

## 13. Operational sequence (chronological)

### Phase 0 — Pre-eval engineering (before redeeming AMD credits)
**Total time: ~6–8 hrs of work, 0 AMD spend.**

1. Fix `run_agent.py` model default (15 min). Currently defaults to `"qwen-32b"`
   which isn't a key in `MODELS`. Fix to `"main"`.
2. Build `sg-nocurator` ablation:
   - Add `enable_curator: bool = True` to `SGConfig` + `save_config` (15 min)
   - In `src/skeletongraph/retrieval/classifier.py`, in `classify_query`, gate
     dynamic classification: when False, return `MODE_SPECS[QueryMode.BUILD_GUIDED]`
     (30 min)
   - Wire `sg-nocurator` backend in `eval/agent/tools.py` (15 min)
   - Add to `eval/agent/config.py` ARMS dict (5 min)
   - Add unit test in `tests/unit/test_ablations.py` (30 min)
3. Build ContextBench loader (~3 hr): see Task 5 in `IMPLEMENTATION_PLAN.md`
   for spec.
4. `pip install aider-chat` + one-task smoke on laptop (~30 min)
5. Run **A0 smoke** on laptop 7B: 5 tasks × 9 arms (including new
   `sg-nocurator`) = 45 runs. Validates all wiring. ~3-4 hrs of laptop time.
6. Aggregate smoke and confirm:
   - All arms produce non-zero retrieval-hit on ≥1 task
   - Ablations behave differently from full sg (different rank order or precision)
   - No crashes, no NaN, no empty patches across all arms
7. **STOP and review smoke output before spending AMD.** If anything is off,
   debug now.

### Phase 1 — Tier A on AMD (continuous run, ~2-3 days)
**Total AMD: ~40 hrs runtime + 6 hr setup, ~\$93.**

1. Day 0: redeem AMD credits. Provision MI300X. Set up vLLM with
   Qwen2.5-Coder-32B at 32K context. Verify with one chat completion. (~3-4 hr)
2. Day 0–1: run A0 again on AMD-32B as calibration (5 tasks × 4 arms = 20 runs,
   ~1.5 hr). Verify token counts match expected (~14K input for retrieval arms).
   This is the cross-server calibration vs the 7B smoke.
3. Day 1–2: A1 + A2 + A3 in one big batch. Use `run_stage` with custom stage
   config that runs all 9 arms across the same 60 tasks (with subset of 40 for
   ablations). ~33 hrs.
4. Day 2: `verify.py` on all completed runs (~1 hr, no GPU needed for the
   harness itself — Docker on a CPU machine).
5. Day 2: `aggregate.py` → SUMMARY.md.

### Phase 2 — Decision gate (review Tier A results)
**Time: ~2 hrs of analysis, 0 spend.**

1. Read the aggregated SUMMARY.md.
2. Check the 5 decision conditions in §7.
3. Decide: stop here (workshop) or proceed to Tier B (conference push).
4. Update this doc with the actual numbers + the decision.

### Phase 3 — Tier B (conditional, +\$34)
**Only execute if §7 decision matrix says go.**

1. B1 ContextBench: ~10 hrs AMD
2. B2 sg-nocurator: ~4 hrs AMD
3. B4 NIM 70B scale: ~few hours, NIM credits, in parallel with B1
4. Re-aggregate; produce final SUMMARY.md with Tier A + B data.

### Phase 4 — Paper writing
Independent of compute. ~3-4 weeks for first draft.

---

## 14. Side activities (parallel, low-priority)

These improve the paper but aren't on the AMD critical path:

### 14B-model side test (smoke before Tier A)
Once `aider-chat` is installed, run the 9-arm smoke on a **14B model** (e.g.
Qwen2.5-Coder-14B) locally if you have a 24-32GB GPU at home. This gives a third
data point between 7B and 32B without spending AMD credits. **~1 day of laptop
time, free.**

### NIM smoke (validate B4 will work)
Once you have an NVIDIA developer key, do one task on `meta/llama-3.1-70b-instruct`
via NIM to confirm context window + completion behavior. **~1 hr, ~1 credit.**

### IDE smoke expansion
Currently only 1 task tested end-to-end via IDE→MCP→SG. Expand to 5-10 tasks
manually with Claude Code or Cursor. Capture screenshots for paper's demo
section. **~half day, free.**

### Config cleanup (technical debt, post-eval)
Strip `AGENT_PRESETS` and `CLI_PROVIDER_PRESETS` from `src/skeletongraph/config.py`.
~2 hrs of work. Required before PyPI release, not for paper.

### Multi-language fixtures (post-eval, optional)
5 small fixtures (TS, Go, Java, Rust, Ruby), `sg build` works on each.
Screenshots / index dumps for the paper's "tool viability" section.
~1 day of work, free.

---

## 15. What this plan deliberately does NOT do

To prevent feature creep, here's what we're **not** doing for the paper:

- **No multi-language eval.** Multi-language is tool-viability, not paper-eval.
  Mention as "future work / shipped in v0.2 release."
- **No real-world IDE telemetry.** No deployment study with actual developers.
- **No user study.** Out of scope for ML workshop. Required only for SE venue
  redirect.
- **No alternative architectures (RAG-fusion variants, different embedding
  models, etc.).** SG is what we're evaluating; baselines are what they are.
- **No paper-level prompt-caching A/B test.** Methodological argument only
  (see §8).
- **No comparison to closed-source systems (Cursor's retrieval, Copilot's
  workspace, etc.).** Unreproducible. Mention as limitation.
- **No optimization of SG for the eval.** Don't tune hyperparameters on the
  eval tasks — that's leakage. SG is what it is on Day 0 of Tier A.

---

## 16. Update protocol (how this doc evolves)

This doc is the eval's source of truth. Update it after each phase:

- **After Phase 0 smoke**: add A0 smoke results table at bottom of §9.
- **After Phase 1 Tier A**: append a "Tier A actual results" section after §11.
  Include the actual numbers vs the expected ranges. Make the §7 decision
  explicitly.
- **After Phase 3 Tier B**: append "Tier B actual results."
- **After paper submission**: tag this doc with submission venue + date.

Use `git log docs/EVAL_PLAN_FINAL.md` for the audit trail.

---

## 17. Stage IDE — deployed-IDE testing (secondary eval stream)

This was missing from v3.1. Added in v3.2.

### What it is

Instead of running tasks through our vLLM + ReAct harness, run them through a
**deployed IDE agent** (Claude Code, Cursor, or Copilot) with and without SG
installed as an MCP server. Two conditions per task: **native** (IDE's built-in
retrieval only) vs **+SG** (SG mounted as MCP server alongside).

### Why it's worth doing

- **External validity**: shows SG helps in deployment, not just on synthetic
  eval — answers the "would real developers benefit?" question that ML eval
  cannot.
- **Closed-source models**: tests Sonnet 4.6, GPT-5.5, Gemini 3.1 Pro — models
  we cannot self-host on AMD.
- **Demo material**: screenshots and qualitative observations for the paper's
  introduction / motivation / demo section.
- **Cheap**: \$0 compute (uses existing IDE subscriptions). Only cost is human
  time.

### Why it's NOT in the headline table

- Not reproducible by reviewers (proprietary models with versioning drift)
- Can't compare across IDEs cleanly (different models, different orchestration,
  different prompt formats)
- Manual labor → small N
- Reviewers explicitly downweight "I ran it on Cursor and it worked" results

### Concrete shape

| Aspect | Value |
|--------|-------|
| IDE | Claude Code (cleanest MCP integration; Sonnet 4.6 default) |
| Tasks | 20 SWE-bench tasks (subset of A1's 60 — direct comparison anchor) |
| Conditions | 2 (native, +SG) |
| Total runs | 40 |
| Per-run time | ~5-10 min human supervision |
| Total human time | ~5-7 hr spread over a week |
| Compute cost | \$0 |

### Method

1. Pick 20 SWE-bench tasks from A1's set, balanced by repo
2. For each: prepare task workspace (same isolation as agent eval)
3. Open in Claude Code, paste the task description as the user message
4. **Native**: let agent solve using only Claude Code's built-in tools
5. **+SG**: add `skeletongraph` MCP server to `.claude/mcp.json`, restart, retry
6. For each run: capture (a) final patch, (b) screenshots of agent trajectory,
   (c) approximate token count from Claude Code's reporter, (d) wall-clock time
7. Verify patches with `verify.py` after both conditions complete

### Metrics
- pass@1 (binary, verified)
- Patch diff vs SG-eval patch (sanity check)
- Token cost estimate (Claude Code's billing telemetry)
- Qualitative notes per task

### Where it goes in the paper

- **Not Table 1** (that stays AMD-only for reproducibility)
- **Table 7 — Deployed-IDE results** (smaller, in a "Real-world deployment"
  subsection)
- Demo screenshots in introduction
- Limitations paragraph: "results not reproducible without identical model
  versions; reported as existence proof of deployment benefit"

### Status
- Planned, not started. **Out of critical path for Tier A/B AMD spend.**
- Suggested timing: parallel with Tier A (during the 2-3 days AMD is running),
  or in the gap between Tier A and Tier B.

---

## 18. The brutal conference-gap analysis

You asked: *why is conference harder than workshop, and what specifically are
we lacking?* Here is the honest breakdown.

### What conferences explicitly value (in rough priority order)

| Criterion | Why reviewers care | We have it? |
|-----------|--------------------|-------------|
| **Novel idea** — a new mechanism or insight | Reviewers must justify the paper's existence | ❌ — we combine known techniques |
| **Theoretical contribution** — proof, bound, formalism | Distinguishes "research" from "engineering" | ❌ — no theorem |
| **Surprising result** — counter-intuitive empirical finding | Memorable, gets cited | ❌ unless numbers land unexpectedly |
| **New evaluation framework** — others will adopt it | Influence > paper itself | ❌ — ContextBench beat us to consolidation-gap |
| **Generality** — works across many settings | "Not just one trick" | ❌ — Python-only eval |
| **Empirical rigor** — multi-benchmark, variance, significance | Hygiene | ✅ with Tier A+B |
| **Beats strongest baselines** — not strawmen | Shows the field is moved | ✅ if SG beats hybrid+aider |
| **Reproducibility** — code, data, configs | Conference assumed; workshop bar | ✅ |

**We are 3/8.** Conference acceptance typically requires 6+/8.

### The specific reviewer kill-shots we will face

1. **"This is engineering, not research."** We combine BM25 + graph + summaries
   + curator — all pre-existing techniques. The defense is "we measure which
   component contributes, via ablations" — but this is empirical work, not
   research per the strict review definition. Hard to deflect.

2. **"The curator is rule-based regex. Why not learn it?"** A learned curator
   (small fine-tuned classifier) would be a real contribution. Ours is
   hand-coded. Hard to deflect without that work.

3. **"You haven't shown generalization beyond Python."** True — multi-language
   adapters exist but aren't evaluated. Easy to acknowledge as future work,
   but it limits the contribution scope.

4. **"ContextBench introduced the consolidation gap. You're just measuring it
   on a different dataset."** Partially true. We apply ContextBench's metric to
   retrieval comparison (vs single-system evaluation in the original). Defensible
   but weakens C2 from "we introduce" to "we apply."

5. **"How does this compare to Codebase-Memory (Mar 2026)?"** They're the
   recent system owning the MCP+graph+token-reduction wedge. We need related
   work that addresses them. If we can't benchmark them as an arm (their
   open-source release is partial), we cite and contextualize.

### Why "good results + good implementation" ≠ conference

Because conference reviewers filter on **research insight**, not on
**engineering quality**. The implicit rubric:

- A paper that says "we combined 4 known techniques and got 8pp improvement" is
  *workshop* material.
- A paper that says "we discovered retrieval quality and downstream task
  success are decoupled in agentic code editing, and propose a metric to
  measure this decoupling that future systems should optimize" is *conference*
  material — even with weaker numbers.

The difference is **the size of the idea**, not the size of the experiment.

### What KIND of ideas go to conferences (recent code-retrieval examples)

- **RepoCoder** (EMNLP 2023): iterative retrieval (retrieve → generate → use
  generation to retrieve again). **NEW algorithm.** Conference accepted.
- **SWE-agent** (NeurIPS 2024): a tool interface designed specifically for
  LLM agents (ACI = Agent-Computer Interface). **NEW design principle.** Conference accepted.
- **AutoCodeRover** (ISSTA 2024): spectrum-based fault localization + LLM.
  **NEW combination of formal methods + LLM.** Conference accepted.
- **RepoGraph** (ICLR 2025): line-level dependency graph + dynamic exploration.
  **NEW data structure + algorithm.** Conference accepted.

What these share: **a new mechanism**, not a new system built from existing
mechanisms.

### Three honest paths to a conference resubmission (post-workshop)

**Path 1 — Learn the curator** (1-2 months work):
- Fine-tune small model (350M-1B) on (query, ideal_strategy) pairs
- Synthesize training data from stronger model
- Becomes: "We learn to route between retrieval strategies"
- Conference accept estimate: 25-35%

**Path 2 — Theoretical bound on the consolidation gap** (uncertain effort):
- Prove: for graph-walking retrieval, consolidation gap is bounded by
  graph density × query specificity
- Empirical validation against the bound
- Conference accept estimate: 30-40% if the math is clean

**Path 3 — Multi-language empirical study** (1 month + compute):
- Build the multi-language adapters (already partially done)
- 5 languages × 50 tasks each × 4 arms
- First systematic structural-retrieval study across languages
- Conference accept estimate: 25-30%

### Recommended strategy: two-paper sequence

1. **Submit workshop paper now** with Tier A + B results. Lower bar, fast
   review cycle (~2 months), banks the empirical contribution.
2. **Use 2-3 months post-workshop** to add Path 1 or Path 3 work.
3. **Resubmit to conference** with the strengthened version.

This beats trying to one-shot a conference submission. Workshop acceptance
also lends credibility to the conference resubmission ("this work was
presented at NeurIPS CodeML 2026").

---

## 19. Ablation coherence — why `sg-nograph` and `sg-nosummary` aren't trivial

Your intuition was right to push on this: "if you strip the graph from SG,
isn't it just BM25?" The answer is **no**, and here is exactly why each
ablation is a coherent, distinct comparison.

### What full `sg` actually does (the pipeline)

When `engine.heuristic_query(query)` runs, these steps execute in order:

1. **Intent analysis** — regex extracts file paths, function names, keywords
2. **Entity resolution** — maps extracted entities to FQNs via the skeleton table
3. **Mode classification (curator)** — picks one of 12 query modes (BUILD_GUIDED, DEBUG_TARGETED, EXPLAIN, REFACTOR, ...) → mode_spec
4. **Graph expansion** — blast_radius (callers) + dependency_chain (callees) at mode_spec depth
5. **BM25 fallback** — if direct entity match fails
6. **Centrality reranking** — hub score (in-degree / max in-degree) reweights candidates
7. **Tier assignment** — T1 for direct targets, T2 for 1-hop neighbors, T3 for 2-hop
8. **Summary attachment** — T2/T3 candidates get 1-line summaries from SummaryStore
9. **Top-N truncation** — sorted by (tier asc, score desc), top 15 returned

### What each ablation actually disables

**`sg-nograph`** (`enable_graph_expansion=False`):
- Disables **only step 4**
- Steps 1, 2, 3, 5, 6, 7, 8, 9 still run
- Result: a system with direct entity matches + BM25 fallback, ranked by the
  multi-signal Ranker (centrality, complexity, test-bonus, export-bonus,
  same-file-bonus), with summaries attached
- **NOT equivalent to `bm25` arm**. Differences:
  - `bm25` ranks every function by raw BM25 score, returns top-k FQNs
  - `sg-nograph` runs entity resolution first (exact-match priority for query
    function names), uses Ranker's multi-signal scoring (not raw BM25), and
    attaches summaries
- **What it tests**: does graph traversal add value over a smart-but-flat ranker?

**`sg-nosummary`** (`enable_summaries=False`):
- Disables **only step 8**
- Steps 1-7 + 9 still run
- Same files retrieved, same ranking, but no 1-line summary text attached
- Agent receives `file::FQN + signature + line range`, NOT `... + 1-line summary`
- **What it tests**: does the summary text itself help the agent, independently
  of which files are retrieved?
- **Important note for eval interpretation**: in the eval harness today,
  `heuristic_query` returns FQNs only (summaries aren't in the search-result
  text the agent sees). So `sg-nosummary` will produce identical retrieval
  rankings as `sg`. The pass@1 difference will appear because:
  - When the agent calls `read_file` on a retrieved FQN, the SG context
    includes (or omits) the summary in the file header
  - Less summary context → agent reads more files redundantly → consolidation
    gap widens → fewer turns left for editing → lower pass@1
- This is C2's direct mechanism.

**`sg-nocurator`** (`enable_curator=False`, to be built):
- Forces mode_spec to a fixed value (e.g., `BUILD_GUIDED` always) in **step 3**
- Steps 4-9 still run with fixed depth/direction/load_tests parameters
- **What it tests**: does adaptive mode selection help vs a one-size-fits-all
  retrieval policy?
- Most concrete effect: queries the curator would classify as `EXPLAIN`
  (deeper dep_depth) or `RETRIEVAL_FAST` (no graph expansion) all get
  BUILD_GUIDED's settings → worse for those query types

### Are these clean ablations?

**Yes, with the standard caveat that ablations are not strictly orthogonal**:

- Removing graph reduces the candidate pool, which reduces the surface area
  for summaries to attach to → `sg-nograph` partially measures
  "graph + downstream-effects-of-fewer-candidates"
- Removing summaries doesn't affect retrieval rankings → `sg-nosummary` is
  the cleanest of the three
- Removing the curator changes downstream mode_spec, which changes graph
  depth → `sg-nocurator` measures "curator + downstream-mode-effects"

This non-orthogonality is **standard in ablation studies** and reviewers
accept it as long as we are explicit:

> "Each ablation disables one component (enable_graph_expansion,
> enable_summaries, enable_curator) while keeping all others active. Effects
> are not strictly orthogonal — removing the graph reduces the candidate pool
> that summaries can attach to, for example — but the ablations bound the
> **marginal contribution** of each component."

### Why we keep all three

| Ablation | Tests claim | Necessity |
|----------|-------------|-----------|
| `sg-nosummary` | C2 (summaries reduce consolidation gap) | **Mandatory** |
| `sg-nograph` | C3 (graph signal adds value) | **Strongly desired** |
| `sg-nocurator` | C1 (adaptive routing helps) | **Mandatory if C1 stays in contribution list** |

Honest position: build `sg-nocurator` (1 hr), run it in Tier B (60 runs,
~\$10). If it shows null effect (curator doesn't actually help), drop C1 from
the paper's contributions and lean on C2 + C3 only. The numbers tell us.

---

## 20. Smoke-script plan (for Sonnet to implement)

**Spec only**. Sonnet implements `eval/scripts/run_smoke.py` from this spec.

### Behavior

```cmd
python -m eval.scripts.run_smoke --model 7b
python -m eval.scripts.run_smoke --model 14b
python -m eval.scripts.run_smoke --model 32b
python -m eval.scripts.run_smoke --model nim
python -m eval.scripts.run_smoke --model 7b --tasks 5 --arms sg,bm25,grep,none
```

### What it does

1. **Validates infrastructure** for the chosen model:
   - `7b` / `14b` / `32b`: pings `http://localhost:8000/v1/models`, confirms
     expected model ID is served
   - `nim`: pings NIM endpoint, confirms `NVIDIA_API_KEY` is set, confirms
     model is reachable
2. **Sets env vars** for `run_agent` / `run_stage` to consume:
   `SG_EVAL_API_BASE`, `SG_EVAL_API_KEY`, `SG_EVAL_MODEL`
3. **Selects tasks**: by default, first 5 tasks from `eval/datasets/stage0.jsonl`
   (stratified one per repo if possible)
4. **Selects arms**: by default, all 9 arms (sg, bm25, grep, none, hybrid,
   aider, sg-nograph, sg-nosummary, sg-nocurator)
5. **Runs each (task, arm) sequentially** via `run_agent.run_one`. Skip if
   `.json` exists already (resume support).
6. **Tags each run** with `model_label` in the JSON record so `aggregate.py`
   can group by model
7. **Aggregates** at the end with a model-label filter
8. **Reports** `SUMMARY-{model}.md` next to the main SUMMARY.md

### Model configuration (inside the script)

```python
MODEL_CONFIGS = {
    "7b": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "max_input_tokens": 32000,
        "label": "7b-local",
    },
    "14b": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "max_input_tokens": 32000,
        "label": "14b-local",
    },
    "32b": {
        "api_base": "http://localhost:8000/v1",     # or AMD VM IP
        "api_key": "EMPTY",
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "max_input_tokens": 32000,
        "label": "32b-amd",
    },
    "nim": {
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model_id": "meta/llama-3.1-70b-instruct",
        "max_input_tokens": 32000,                  # confirm via NIM smoke test
        "label": "nim-llama-70b",
        "rpm_limit": 40,                            # client-side throttle
    },
}
```

### Pre-flight checks (do NOT skip)

1. **Endpoint health**: `requests.get(f"{api_base}/models", timeout=5)`. Non-200
   → abort with clear error.
2. **Model availability**: confirm `model_id` is in the served list.
3. **Tokenizer sanity**: 50-token test prompt, verify completion returns
   sensible token counts.
4. **Context window check** (local models only): 25K-token prompt, confirm no
   truncation. Catches misconfigured vLLM context cap.
5. **Workspace cache**: confirm `eval/datasets/_repo_cache/` has the task repos
   cloned (skip download time during smoke).

### CLI arguments

```
--model {7b,14b,32b,nim}      [required]
--tasks N                      [default: 5]
--arms a,b,c                   [default: all 9 arms]
--task-ids id1,id2             [override --tasks for specific tasks]
--no-resume                    [re-run all even if .json exists]
--out-suffix LABEL             [override model label for output naming]
--rpm-limit N                  [client-side throttle, default: from config]
--dry-run                      [print plan, don't execute]
```

### Output

- Per-run JSONs in `eval/results/agent/` (existing convention)
- New: `eval/results/agent/SUMMARY-{label}.md` for this model
- New: cross-model comparison helper produced by separate
  `aggregate_by_model.py` (small follow-up tool)

### What the script does NOT do

- Doesn't run `verify.py` (separate step, no GPU needed)
- Doesn't manage vLLM lifecycle (caller starts the server separately)
- Doesn't aggregate across multiple model labels (one SUMMARY per model;
  cross-model comparison is a separate manual or scripted step)

### NIM-specific behavior

- Client-side throttle to `rpm_limit` via leaky-bucket
- Retry on 429 with exponential backoff (max 3 retries)
- Estimate NIM credit consumption from token counts (rough — for budget
  visibility only)
- Abort if 3 consecutive runs fail (suggests credit exhaustion or API issue)

### Estimated effort
- ~3 hours for Sonnet to write cleanly with all pre-flight checks + tests
- ~30 min follow-up: `aggregate_by_model.py` for cross-model comparison table

### Where this slots into the operational sequence

New **Phase 0.5 — Model-portability smoke** between Phase 0 (engineering) and
Phase 1 (AMD Tier A):

1. After Phase 0 engineering complete + smoke wiring validated on laptop 7B
2. Run `--model 7b` smoke (laptop)
3. Run `--model 14b` smoke (laptop or external GPU)
4. Run `--model 32b` smoke (AMD VM after setup)
5. Run `--model nim` smoke (NVIDIA API)
6. Compare SUMMARY-7b.md, SUMMARY-14b.md, SUMMARY-32b.md to confirm:
   - SG precision gap holds across model sizes
   - Token counts scale as expected
   - No arm produces all-zero/all-error results on any model

This is the **portability validation gate** — if SG only works on 7B and
collapses on 32B, we want to know BEFORE burning Tier A on AMD.

---

## 21. TL;DR for the next agent reading this

You're about to spend \$100 of AMD MI300X credits. Read this whole doc once
before doing anything irreversible.

1. Do the 4-5 hours of pre-eval engineering in §13 Phase 0.
2. Run the smoke on laptop 7B. Confirm everything works.
3. Spend Tier A (~\$93) on the headline + strong baselines + 2 ablations.
4. Aggregate. Check §7 decision rules.
5. If 3+ conditions met → spend Tier B (~\$34) for ContextBench + sg-nocurator.
6. If <3 met → stop, write workshop paper with Tier A only.

Acceptance target: **NeurIPS CodeML workshop** (Path A). Expected probability:
55–80% depending on results. Conference push only if Tier B happens AND
ContextBench numbers cooperate.

**Do not** spend additional AMD credits beyond Tier A + Tier B without
revising this doc and re-evaluating the conditional probabilities.

---

*End of plan. Source of truth is git.*
