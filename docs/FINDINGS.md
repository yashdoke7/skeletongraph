# Findings

Everything the run corpus actually supports, with the numbers, as of 2026-09-20.
Written to be read cold, without the conversation that produced it.

Every number here comes from the verified run JSONs under `eval/results/agent/`.
Percentages are always paired (same tasks, both arms) and are reported next to the
absolute change, because a percentage on a small base is misleading on its own —
`+58%` below is 2.5 turns and two cents.

---

## The headline

**A context/retrieval intervention has no transferable effect size.** The same
retrieval system, on the same 100 tasks, saves 64% of input tokens in one harness
and costs 58% more in another. The difference is not a property of the retriever.
It is predicted by two properties of the *host harness*, both measurable from the
baseline before anything is installed:

1. **how often the harness already reaches the right file** → bounds the accuracy gain;
2. **how many tokens the harness spends getting there** → determines the cost effect.

---

## F1 — The same retriever at eight harness operating points

Paired, same tasks, verified. `Δ tokens` is mean input tokens per task
(fresh + cache-read + cache-creation).

| harness operating point | n | baseline tok/task | Δ tokens | Δ % | Δ turns | Δ $/task | Δ pass@1 | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Claude Code 2.1.278 | 100 | 112,813 | **+65,991** | +58% | +2.5 | +$0.020 | +2 | 0.754 |
| Codex CLI 0.155.0 | 100 | 134,955 | +7,734 | +6% | −0.3 | −$0.004 | +6 | 0.210 |
| ReAct · nemotron v2 | 100 | 244,123 | −69,324 | −28% | −1.5 | −$0.019 | +1 | 1.000 |
| ReAct · nemotron v4 | 100 | 497,137 | **−317,067** | −64% | −1.7 | −$0.014 | **+7** | 0.143 |
| Claude Code 2.1.274 | 100 | 530,478 | −93,274 | −18% | −1.9 | −$0.026 | −10 | 0.006 |
| Claude Code 2.1.206–211 | 100 | 644,504 | −152,317 | −24% | −3.1 | −$0.063 | +1 | 1.000 |
| Claude Code 2.1.211–214 (SWE-rebench) | 49 | 1,048,603 | −255,641 | −24% | −4.7 | −$0.118 | −1 | 1.000 |
| Claude Code 2.1.214 (rebench, prose) | 50 | 1,119,864 | −228,276 | −20% | −4.0 | −$0.111 | −3 | 0.453 |

Correlation between log₁₀(baseline tokens/task) and the token effect: **r = −0.65**
across the eight points.

**Read the absolute column, not the percentage.** The relative effect flips sign,
but the absolute penalty in a lean harness is small and bounded (+2.5 turns,
2¢) while the absolute saving in a wasteful one is large and grows with the waste
(−4.7 turns, 12¢). Retrieval's downside does not scale; its upside does.

*The one significant solve result (−10, p=0.006) does not replicate: the same two
arms on the same 100 tasks scored 74 vs 75 in the previous run. See F8.*

Figure: `docs/paper/figures_v2/fig6_effect_is_a_harness_property.*`

---

## F2 — Why: retrieval substitutes for exploration

Two independent axes, both owned by the harness.

**Axis 1 — accuracy headroom = the localisation gap.** Retrieval can only win the
tasks the harness does not already reach.

| harness | reaches gold file unaided | with SG | Δ pass@1 |
|---|---|---|---|
| ReAct · nemotron | **65%** | **87%** | **+7** |
| Claude Code (SWE-rebench) | 90% | 92% | −1 |
| Codex CLI | 94% | 95% | +6 |
| Claude Code (Verified) | 97% | 97% | +2 |

At 97% reach and a 73–79% conversion rate from reach to solve, perfect retrieval
is worth **at most ~2–3 solves per 100**. The measured nulls are not
underpowered — they are bounded.

**Axis 2 — cost headroom = the harness's token appetite.** See F1. Crossover sits
around 200k tokens/task: below it retrieval's payload is net overhead, above it
retrieval substitutes for exploration the harness would have paid for anyway.

Figures: `fig1_localisation_ceiling.*`, `fig6_effect_is_a_harness_property.*`

---

## F3 — Retrieval quality itself is genuinely good, and it doesn't matter

Standalone file-level retrieval on SWE-bench Verified (no agent):

| backend | recall@5 | recall@10 | MRR |
|---|---|---|---|
| grep | 0.236 | 0.348 | 0.159 |
| bm25 | 0.626 | 0.719 | 0.482 |
| sg-rerank | 0.701 | 0.824 | 0.518 |
| bm25+dense | 0.714 | 0.843 | 0.551 |
| **bm25+dense+SG** | **0.785** | **0.856** | **0.658** |

In-agent first-search rank-1 rate rises the same way (Claude 59→89, Codex 32→70).
Across 15 retrieval backends in the controlled ReAct harness, pass@1 spans **41–45**
while first-search hit rate spans **0–80**. `none` — no retrieval at all — scores 44.

This matters because it rules out the obvious objection: the null is not "a bad
tool does nothing". It is a 3× retrieval-quality improvement doing nothing.

Figure: `fig2_retrieval_ladder.*`

---

## F4 — The corroboration gap: why faster localisation isn't a faster fix

Turn at which the agent first *reads* the gold file vs first *edits* anything
(controlled ReAct harness, n≈85 per arm):

| backend | reads gold file at turn | first edit at turn | gap |
|---|---|---|---|
| no retrieval | 5.6 | 12.4 | 6.8 |
| graphify | 6.9 | 13.9 | 7.1 |
| grep | 2.6 | 12.0 | 9.4 |
| bm25 | 2.5 | 11.5 | 9.1 |
| SG fusion | **2.1** | **11.4** | **9.3** |

Retrieval delivers the file **3.5 turns earlier** and the edit happens **1.0 turn
earlier**. The turn of the first edit is near-invariant (11.4–13.9) across every
backend including none.

The agent will not edit on the strength of having found a plausible file. It
corroborates first — reading callers, tests and neighbours to confirm the fix site
— and that phase expands to absorb whatever localisation saves. Retrieval
compresses the search phase, which is the cheap phase. It leaves the expensive
phase untouched.

This is the mechanism underneath F1 and F2 and it is the most actionable finding
here: an intervention aimed at *finding* code is aimed at the wrong phase.

Figure: `fig7_corroboration_gap.*`

---

## F5 — The natural experiment: Claude Code 2.1.274 → 2.1.278

Same 100 tasks, same arm (`native`), three weeks apart, nothing in our setup changed.

| | turns | tokens/task | searches | rank-1 | pass@1 |
|---|---|---|---|---|---|
| 2.1.274 | 11.9 | 530,478 | 3.5 | 59 | 79 |
| 2.1.278 | 5.7 | 112,813 | 1.7 | **85** | 71 |

McNemar on the solve change: 11 lost, 3 gained, p=0.057.

The vendor cut baseline token use **4.7×** and the built-in search's own rank-1
rate rose from 59 to 85 — closing 26 of the 30 points that had separated it from a
dedicated structural retriever. On 2.1.278, native (85) and SG (89) are tied.

**The same retrieval tool's effect flipped from −18% to +58% on tokens in three
weeks, without the tool changing.** This is the sharpest available evidence that
published effect sizes for agent context tooling have a shelf life.

---

## F6 — Behavioural scaffolding is a second intervention, and it also doesn't transfer

The shipped SkeletonGraph configuration adds behavioural directives (a CLAUDE.md
ruleset, an appended system prompt, session hooks, and instruction text inside tool
results). `sg-fusion-plain` is the identical retriever with all of that removed —
verified to produce identical rankings.

| harness | native | SG shipped | SG plain |
|---|---|---|---|
| Claude Code 2.1.274 / 2.1.278 | 79 | 69 | 73 |
| Codex CLI 0.155.0 | 53 | 58 | 59 |

Plain also retrieves better in both harnesses (rank-1 71→89 Claude, 48→70 Codex)
and removes the orientation overhead entirely (`sg_overview` called on 26/30 runs
shipped, 0 plain).

Report this as a replication of F1 on a second intervention class, not as a
result about scaffolding being bad in general: we authored these particular rules,
so this is one sample, and its value is that it behaves the same way the retriever
does — harness-dependent, non-transferable.

---

## F7 — Measurement hazards found the hard way

Each of these silently corrupted a result before it was caught.

**A mid-study repricing.** Fitting cost against token components per CLI version
recovers the price sheet exactly (residual 0.0–1.3%):

| version | cache-read $/Mtok | cache-write $/Mtok | output $/Mtok |
|---|---|---|---|
| 2.1.206–214 | 0.28–0.32 | 6.03–8.40 | 13.11–15.12 |
| 2.1.274–278 | 0.20–0.21 | 4.00–4.21 | 9.51–10.00 |

A uniform ~33% price cut landed mid-corpus. **Any dollar-denominated comparison
spanning it is invalid.** Report tokens; dollars are a secondary column with the
sheet stated.

**A run-to-run noise floor.** Identical arm, identical configuration, re-run:
9/100 verdicts flip (native), 16/100 (sg-fusion). A 25-task re-run flipped 9 of
25 (4 solved → 9 solved). 19–24 tasks differ by more than 2× in cost. Single-run
n=100 agent comparisons are underpowered for anything smaller than ~10 points.

**Harness version is a required covariate.** It explains the token behaviour
completely (ctx/turn 35–45k for 2.1.206–274, 18–23k for 2.1.278) and was not
recorded for the first eight months of runs. It is recorded now.

**Contamination channels, all closed:** web fetch of the upstream fix (33 runs
across all Claude tags; web tools now denied on every arm and Bash network use
flagged); `git fsck --unreachable` recovering a prior run's solution blobs (repo
scrub in `reset_repo`; two later attempts returned empty); a proxy bypass via
`curl --noproxy "*"` (detected, one run quarantined); the SWE-bench harness reusing
a stale per-instance verdict for a re-run task (`exclude_completed` is hardcoded —
verify now clears those logs).

---

## What is already published, and what is ours

Established in 2026 — **do not claim these**:

- Harness choice changes cost several-fold while barely changing success rate —
  [HarnessTax](https://harnesstax.github.io/), [Arena](https://arena.ai/blog/coding-agents-harness-tax),
  [Harness-Bench](https://arxiv.org/html/2605.27922v1) (23.8pp completion spread,
  6 harnesses × 8 models), [Claw-SWE-Bench](https://arxiv.org/html/2606.12344v1) (27.4pp).
- The harness must be disclosed — [Stop Comparing LLM Agents Without Disclosing the Harness](https://arxiv.org/pdf/2605.23950).
- Retrieval quality has a bounded payoff for repair, with residual non-localisation
  failure — [Beyond Localization](https://arxiv.org/pdf/2603.29067) (SWE-bench Lite,
  GPT-4.1 / DeepSeek-V3, **one fixed harness**).
- File-level localisation is near-saturated for modern methods —
  [SWE-Explore](https://arxiv.org/abs/2606.07297).

The gap: **every one of those holds either the harness or the intervention fixed.**
Beyond Localization varies retrievers on one harness. Harness-Bench varies harnesses
with no retrieval tools and states it "does not compare retrieval/context tool
variants directly". Stop-Comparing lists "systematic analysis of how specific
harness components influence relative agent rankings" as open future work.

Ours is the interaction: one intervention, eight harness operating points, plus a
within-product version change. Harness-Bench found weaker *models* are more
sensitive to the harness; we find weaker *harnesses* are more sensitive to context
tooling — the same shape on a new axis.

---

## Not established (do not write these as findings)

- That behavioural scaffolding is harmful in general. One authored ruleset, n=1.
- That 2.1.278 is "worse". It trades tokens for accuracy; we measured one point on
  that trade, at p=0.057.
- Any cross-run dollar comparison spanning the 2.1.214 → 2.1.274 boundary.
- Anything from `claude_v7_prose` (only 15 of 50 per arm verified) or from tags
  with fewer than ~30 verified runs.
- That the 200k-token crossover is precise. Eight points, r=−0.65; it is a trend,
  not a calibrated threshold.

---

## Corpus provenance

| tag | arms | CLI version(s) | n/arm | benchmark |
|---|---|---|---|---|
| `nemotron_v2` | 15 retrieval backends | ReAct loop (own harness) | 100 | Verified |
| `nemotron_v4` | 7 backends | ReAct loop | 100 | Verified |
| `claude_v7` | native, sg-fusion | 2.1.206 / .210 / .211 | 100 | Verified |
| `claude_v7_prose` | native, sg-fusion | 2.1.214 | 50 (15 verified) | Verified, paraphrased |
| `claude_rebench_v1` | native, sg-fusion | 2.1.211 / .214 | 50 | SWE-rebench |
| `claude_rebench_prose_v1` | native, sg-fusion | 2.1.214 | 50 | SWE-rebench, paraphrased |
| `claude_v7_rep2` | native, sg-fusion | **2.1.274** | 100 | Verified |
| `claude_v7_rep2` | sg-fusion-plain | **2.1.278** | 100 | Verified |
| `claude_v8` | native, sg-fusion-plain | **2.1.278** | 100 / 11 | Verified |
| `codex_v1` | native, sg-fusion, sg-plain | codex-cli 0.155.0 | 100 | Verified |

Quarantined contaminated runs live in `<tag>/_quarantine_contaminated_*/` and are
excluded from every number above.

Regenerate all figures: `python docs/paper/figures_v2/make_figures.py`
