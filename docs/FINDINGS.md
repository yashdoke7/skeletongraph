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
Across these observational panels it is associated with two baseline properties
of the *host configuration*:

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
| ReAct · nemotron v4 | 100 | 344,642 | **−164,573** | −48% | −1.7 | −$0.014 | **+7** | 0.143 |
| Claude Code 2.1.274 | 100 | 530,478 | −93,274 | −18% | −1.9 | −$0.026 | −10 | 0.006 |
| Claude Code 2.1.206–211 | 100 | 644,504 | −152,317 | −24% | −3.1 | −$0.063 | +1 | 1.000 |
| Claude Code 2.1.211–214 (SWE-rebench) | 49 | 1,048,603 | −255,641 | −24% | −4.7 | −$0.118 | −1 | 1.000 |
| Claude Code 2.1.214 (rebench, prose) | 50 | 1,119,864 | −228,276 | −20% | −4.0 | −$0.111 | −3 | 0.453 |

Correlation between log₁₀(baseline tokens/task) and the token effect: **r = −0.63**
across the eight points.

The 100-task 2.1.278 point pairs native `claude_v8` with plain SG from
`claude_v7_rep2`, recorded in separate tags. The strictly within-`claude_v8`
same-version panel currently has only 11 paired tasks. Treat the 100-task point as
a time-separated product-version comparison rather than a randomized arm contrast.

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

**Axis 1 — accuracy headroom = the localisation gap.** The field used below is
`edited_gold_file`: it measures whether the run edited a gold file, not merely
whether it read or reached one.

| harness | edits a gold file unaided | with SG | Δ pass@1 |
|---|---|---|---|
| ReAct · nemotron | **65%** | **87%** | **+7** |
| Claude Code (SWE-rebench) | 90% | 92% | −1 |
| Codex CLI | 94% | 95% | +6 |
| Claude Code (Verified) | 97% | 97% | +2 |

At 97% gold-file editing and a 73–79% conversion rate from that event to solve,
the remaining headroom through *file localisation alone* is small. This bounds
the likely effect size, but it does **not** establish statistical equivalence:
n=100 paired tasks is still underpowered for a ±2–3 point non-inferiority claim.

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
Across 15 retrieval backends in the controlled ReAct harness, pass@1 spans **37–45**
while first-search hit rate spans **0–80**. `none` — no retrieval at all — scores 44.

This rules out the narrow objection that every null comes from an obviously weak
retriever. It shows a large proximal retrieval improvement that does not transfer
reliably to pass@1; it does not establish that every backend is equivalent.

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
earlier**. The turn of the first edit is near-invariant (11.4–13.9) **among runs
for which an edit timestamp exists** across every backend including none.

This is consistent with the agent corroborating before editing — reading callers,
tests and neighbours to confirm the fix site — but the timing comparison is
conditioned on a post-treatment event. The intervention also changes whether an
edit is ever observed (75% for `none`, 91% for fusion in the v4 panel), so the
table alone cannot show that the edit phase is invariant. Occurrence and censored
time-to-event analyses are required alongside conditional means.

This is a candidate mechanism underneath F1 and F2 and the most actionable lead
here: an intervention aimed only at *finding* code may leave the harder
evidence-to-patch conversion phase untouched.

Figure: `fig7_corroboration_gap.*`

---

## F5 — The natural experiment: Claude Code 2.1.274 → 2.1.278

Same 100 tasks and arm (`native`), three weeks apart. Our runner configuration did
not change, but the product bundle did: the advertised tool and skill surfaces
changed with the CLI version, so this is a longitudinal product-version contrast,
not an isolated context-policy intervention.

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

The Codex comparison holds CLI version fixed. The headline Claude comparison does
not: shipped SG is almost entirely 2.1.274 and plain SG is 2.1.278. It therefore
cannot identify a scaffolding effect. The only same-version Claude comparison is
the incomplete `claude_v8` panel (n=11 paired), where native solves 6 and SG plain
solves 8 (McNemar p=0.50). Treat the Claude result as confounded and the Codex
result as one controlled sample, not as a replicated intervention class.

---

## F7 — Measurement hazards found the hard way

Each of these silently corrupted a result before it was caught.

**A mid-study repricing.** Fitting cost against token components per CLI version
recovers the price sheet closely (mean absolute residual 0.0–5.4%; 5.4% is 2.1.214):

| version | cache-read $/Mtok | cache-write $/Mtok | output $/Mtok |
|---|---|---|---|
| 2.1.206–214 | 0.28–0.32 | 6.03–8.40 | 13.11–15.12 |
| 2.1.274–278 | 0.20–0.21 | 4.00–4.21 | 9.51–10.00 |

A uniform ~33% price cut landed mid-corpus. **Any dollar-denominated comparison
spanning it is invalid.** Report tokens; dollars are a secondary column with the
sheet stated.

**Run-to-run instability, but not yet a clean noise-floor estimate.** The 9/100
native and 16/100 SG verdict flips compare `claude_v7` (2.1.206–211) with
`claude_v7_rep2` (mainly 2.1.274), so version change is inseparable from stochastic
variation. The 25-task rerun is version-matched, but it deliberately oversamples
prior SG failures (21/25); 9 verdicts flip, including 7 failures becoming passes.
These observations motivate repeated trials, but neither sample estimates the
unconditional single-run noise floor. Single-run n=100 comparisons remain
underpowered for effects much smaller than about ten points.

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

The original gap statement — that every paper holds either harness or intervention
fixed — is no longer true. [Prompt-Induced Waste](https://arxiv.org/abs/2608.01347)
reports harness-dependent responses to an effort-control intervention, and
[Same Model, Different Harness](https://arxiv.org/abs/2608.26218) studies context
shortening across harnesses and benchmarks. The narrower gap that remains is a
retrieval-specific, trajectory-level account of *where* an intervention's benefit
is gained or lost. Beyond Localization still varies retrievers in one fixed
harness; Harness-Bench varies harnesses without comparing retrieval/context tools.

Our potentially novel contribution is therefore not the generic existence of a
harness × intervention interaction. It is the retrieval-to-repair funnel across
operating points, together with evidence that first-hit improvements are not a
stable treatment effect and often do not propagate through reading, editing,
patch production, and verified resolution. This causal wording requires the
factorial follow-up specified in the independent audit below.

---

## Not established (do not write these as findings)

- That behavioural scaffolding is harmful in general. One authored ruleset, n=1.
- That 2.1.278 is "worse". It trades tokens for accuracy; we measured one point on
  that trade, at p=0.057.
- Any cross-run dollar comparison spanning the 2.1.214 → 2.1.274 boundary.
- Anything from `claude_v7_prose` (only 15 of 50 per arm verified) or from tags
  with fewer than ~30 verified runs.
- That the 200k-token crossover is precise. Eight points, r=−0.63; it is a trend,
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

---

## Independent audit — corrections, new findings, and paper direction

This section is a fresh audit of the raw corpus and transcripts, written after
forming an independent view and then comparing it with F1–F7. Where it conflicts
with an earlier section, this section supersedes it. The reproducible census and
paired calculations are in `eval/scripts/corpus_audit.py`; its machine-
readable output is `tmp/corpus_audit.json` (regenerated, not committed).

### Audit scope and what can actually be pooled

The tree contains **5,150 raw run records**, of which **5,092 are active** and 58
are archived or quarantined. There are **3,813 graded active records**, 32 tags,
89 arm labels, and 257 unique active task IDs. No active tag contains a duplicate
`(task, arm, repeat)` key, and no within-tag disagreement was found for benchmark
identity, base commit, or gold files. This is a large corpus, but not 5,092
independent verified trials: many records are ungraded pilots, incomplete panels,
or repeated observations of the same tasks. The unit for uncertainty must be the
task, and cross-tag pooling must preserve version, harness, benchmark, and repeat.

The token fields are not longitudinally interchangeable. In the older ReAct
records, `billed_input` is cumulative prompt traffic. New Claude and Codex records
also contain `total_input_tokens` plus cache-read/cache-creation components; a
small Claude `billed_input` can simply mean that almost all input was cached.
Cross-harness token analysis should use one documented semantic definition and
retain the raw components.

### Evidence ledger

| status | claim the corpus supports | qualification |
|---|---|---|
| **solid** | Retrieval changes where the agent looks far more reliably than it changes verified repair | Paired first-hit gains are large in several panels; solve effects are small, heterogeneous, and usually non-significant |
| **solid** | Token effects depend strongly on the host operating point | Same intervention saves 18–64% in high-traffic configurations and adds traffic in lean configurations; this is descriptive, not yet a causal harness effect |
| **solid** | The best Verified standalone result is hybrid fusion, not SG reranking alone | BM25+dense+SG beats BM25+dense by +0.106 MRR and +0.071 recall@5; recall@10 changes only +0.013 |
| **solid** | Standalone backend rankings do not transfer to SWE-bench Pro | On Pro, BM25 beats SG reranking by +0.106 file MRR and +0.120 FQN MRR |
| **suggestive** | Retrieval benefit decays through a search→read→edit→patch→solve funnel | Strongest in the controlled ReAct v4 panel; transcript/event missingness requires a pre-specified multi-state analysis |
| **suggestive** | There is no stable set of “retrieval-sensitive” tasks | Per-task treatment-effect correlations are −0.07 to 0.34 and winning-task Jaccard overlap is 0.00–0.20 across five panels, but discordant sets are small |
| **confounded** | “Harness weakness” causes larger retrieval gains | ReAct, Claude, and Codex also change model family, integration, prompts, and tool protocol; the current corpus is not a model×harness factorial |
| **confounded** | Claude behavioral scaffolding hurts | The rich arm is mainly CLI 2.1.274 and the plain arm 2.1.278; only Codex holds version fixed |
| **confounded** | Claude 2.1.278 isolates a context-policy change | Tools and advertised skills changed with the product version |
| **measurement artefact** | First-search miss means failure to localise | Direct reads and later searches can reach the gold file; some no-search runs are coded as misses and still pass |
| **measurement artefact** | Conditional mean edit turn proves retrieval leaves editing unchanged | The intervention changes whether an edit timestamp exists, so conditioning selects different populations |

### The new central result: an attenuating retrieval-to-repair funnel

The cleanest mechanistic panel is `nemotron_v4`, `none` versus `fusion`, on the
same 100 tasks:

| stage | no retrieval | fusion | absolute change |
|---|---:|---:|---:|
| first search hits a gold file | 0% | 80% | +80 |
| ever reads a gold file | 87% | 97% | +10 |
| edits a gold file | 65% | 87% | +22 |
| produces a patch | 74% | 90% | +16 |
| verified solve | 35% | 42% | +7 |

This is more informative than either “retrieval works” or “retrieval does
nothing.” It works at its proximal target, but most of the gain is lost before
the benchmark endpoint. Pairing reveals an additional surprise. Of the 26 tasks
that changed from not editing a gold file to editing one under fusion, only four
became solves. Across the 61 tasks where both arms edited a gold file, fusion had
a net six-solve advantage. Thus the observed +7 is not explained mainly by
closing the gold-edit gap; much of it arises after both agents have already
localized and edited the right area. The endpoint difference is itself uncertain
(McNemar p=0.143), so this is a mechanism-generating result, not a mediation claim.

The attenuation pattern varies. In ReAct v2, 5 of 11 new gold-edit transitions
became solves; in Claude v7 there was only one such transition and no corresponding
solve; in Claude rep2 there were none; and in Codex only two, with one net solve.
This heterogeneity is the real scientific target: **when does better evidence
change merely the trajectory prefix, and when does it improve the conversion from
located code to a correct patch?**

### `retrieval_hit` is not a causal mediator

Pooled over active graded runs, a first-search hit correlates with passing (52.0%
versus 38.7%), but within arms the sign reverses repeatedly. Claude v7 native
passes 69.9% with a recorded hit and 85.2% without one; Claude rep2 SG passes
64.7% with a hit and 93.3% without; Codex native passes 39.3% with a hit and 74.4%
without. ReAct panels show the intuitive positive association. These are not
paradoxes about retrieval quality: search is chosen by the policy, and easy runs
may directly open a known file or never search. `retrieval_hit` is a post-treatment,
policy-dependent event. It is useful as a funnel observable, not as an independent
predictor or an adjustment covariate.

### Standalone retrieval reverses across benchmarks

The Verified table in F3 is correct as a ranking of aggregate means, but the
strongest paired result is specifically **hybrid fusion**. Relative to BM25+dense,
adding SG improves MRR by +0.106 (approximate paired 95% CI +0.064 to +0.149) and
recall@5 by +0.071 (+0.017 to +0.124), while recall@10 changes by only +0.013
(−0.023 to +0.048). This looks like early-rank reordering, not new top-10 coverage.
SG reranking alone versus BM25 improves Verified MRR by only +0.036, with an
interval that crosses zero.

On the 129-task SWE-bench Pro file panel, the direction reverses: BM25 beats SG
reranking by +0.106 MRR (+0.031 to +0.181), +0.117 recall@5, and +0.071 recall@10.
At FQN granularity (118 common tasks), BM25 leads by +0.120 MRR (+0.049 to +0.191),
+0.082 recall@5, and +0.121 recall@10. Therefore “structure-aware retrieval is
genuinely better” is not corpus-wide. The defensible result is: fusion materially
improves early ranking on this Verified sample, while standalone SG reranking is
dataset-sensitive and loses to lexical retrieval on Pro.

Do not use the standalone latency numbers as a speed result. For example,
BM25+dense is recorded at 214 seconds/task while the nominally more complex fusion
is 37 seconds/task, an ordering consistent with cache, warmup, or experiment-order
effects rather than algorithmic latency.

### Pairing, replication, and multiplicity

The important verified paired outcomes are:

| panel | baseline → treatment pass@1 | paired Δ | McNemar p | interpretation |
|---|---:|---:|---:|---|
| Claude v7 | 74 → 75 | +1 | 1.000 | null at this resolution |
| Claude rep2, rich SG | 79 → 69 | −10 | 0.006 | real within-panel difference; does not replicate v7 |
| Codex, SG plain | 53 → 59 | +6 | 0.210 | suggestive, imprecise |
| ReAct v2, SG rerank | 44 → 45 | +1 | 1.000 | null |
| ReAct v4, fusion | 35 → 42 | +7 | 0.143 | suggestive, imprecise |
| Rebench | 55.1 → 53.1 | −2.0 | 1.000 | null |
| Rebench paraphrase | 50 → 44 | −6 | 0.453 | imprecise |

The 15-arm v2 sweep creates 105 pairwise accuracy contrasts. The lowest isolated
contrast found here is 37 versus 45 for `sg-dense-rerank` versus `sg-rerank`
(unadjusted p=0.039); it should not be promoted after an exhaustive sweep without
a pre-specified contrast or multiplicity correction. For the final paper, use a
small hierarchy of primary contrasts and Holm or false-discovery control for the
rest.

The v7→rep2 comparison cannot estimate stochastic noise because the CLI changes
from 2.1.206–211 to mainly 2.1.274. The version-matched 25-task retry is selected:
21 of 25 tasks were prior SG failures, and 7 of those failures pass on retry. It
demonstrates instability but not its population rate. A repeated, randomized panel
is necessary. Likewise, the cross-panel model does not reveal a reproducible
“retrieval-sensitive task” subset: signed per-task effect correlations range from
−0.07 to 0.34 and treatment-only win-set Jaccard overlap from 0.00 to 0.20.

### Transcript and provenance audit

The transcript metadata confirms nontrivial version imbalance. In Claude v7,
paired native/SG tasks use exactly the same CLI version in only 72/100 cases. In
rep2, rich SG and native are almost all 2.1.274, while 99/100 rich-versus-plain
pairs cross from 2.1.274 to 2.1.278. The 2.1.278 transcript headers also advertise
more skills and a changed tool surface. Version must therefore be a blocking
factor, not a footnote.

No persistent Claude project-memory file was found for these runs. One Codex run
that successfully reached an external fix is quarantined and replaced; active
Codex records may contain blocked network attempts, which are not themselves proof
of leakage. The `claude_v8` records omit `leak_reached` rather than recording
`false`, so detector coverage should not be described as uniform across all tags.
The all-zero 80-run `nemotron_pro` fusion panel looks like stale or failed
verification and should remain excluded pending re-verification.

### Position against the 2025–2026 literature

Several broad claims are already published.
[Agentless](https://arxiv.org/abs/2407.01489) established staged
localize–repair–validate pipelines;
[LocAgent](https://arxiv.org/abs/2503.09089),
[RepoGraph](https://openreview.net/pdf?id=dw9VUsSHGB), and
[CodeRAG-Bench](https://aclanthology.org/anthology-files/pdf/naacl/2025.naacl-findings.176.pdf)
already show that graph or structure-aware retrieval can improve localization;
[Beyond Localization](https://arxiv.org/abs/2603.29067) and
[SWE-Explore](https://arxiv.org/abs/2606.07297) show that file finding does not
exhaust repair difficulty. [HarnessTax](https://harnesstax.github.io/),
[Harness-Bench](https://arxiv.org/abs/2605.27922),
[Claw-SWE-Bench](https://arxiv.org/abs/2606.12344),
[Scaffold Effect](https://arxiv.org/abs/2607.22585), and
[Stop Comparing](https://arxiv.org/abs/2605.23950) establish that harness choice
changes quality, cost, and failure modes.
[Prompt-Induced Waste](https://arxiv.org/abs/2608.01347) now directly reports a
harness × effort-control interaction, so the generic interaction is not novel.
[On Randomness in Agentic Evals](https://arxiv.org/abs/2602.07150) reports 2.2–6
point single-run variation over a much larger repeated SWE-bench corpus, and
[SWE-rebench](https://arxiv.org/abs/2505.20411) motivates fresh, decontaminated,
time-sliced evaluation.

The remaining publishable gap is narrower and stronger: no cited work gives a
paired, cross-operating-point **multi-stage decomposition of retrieval's effect**
from first exposure through verified patch, while separating retrieval ranking,
evidence packaging, tool-registration overhead, and behavioral policy. That is the
paper this corpus can motivate, but the present observational cross-harness panels
cannot finish the causal argument alone.

### Recommended paper thesis and decisive experiment

**Proposed thesis:** *Repository retrieval is an information intervention whose
proximal localization gains attenuate through an agent-specific repair funnel;
its value is governed less by standalone recall than by how the host policy
converts evidence into diagnosis, edits, validation, and a correct patch.*

The decisive question is:

> Once the correct code has been found, can better evidence packaging and
> corroboration policy increase conversion to a verified patch, or does retrieval
> mostly change where the agent looks?

Run a version-pinned, blocked factorial with the **same model exposed through at
least two harnesses**, two fresh benchmarks, and three independent repeats per
task. Separate four interventions that the current corpus bundles together:

1. tool registered but returning a neutral payload (registration/placebo cost);
2. lexical or dense ranked files (ranking effect);
3. the same files augmented with structural neighbours, callers, and tests
   (evidence-packaging effect);
4. the same payload plus behavioral directives (policy/scaffolding effect).

Add oracle-file and oracle-function/evidence-pack arms. These identify whether
the bottleneck is location granularity or the policy's use of already-correct
evidence. Freeze model snapshot, CLI/harness commit, tool schema, repository state,
network policy, prompt, budget, and verification image. Randomize arm order within
task and keep retries fresh. A practical core design is 100 tasks × 2 harnesses ×
5 arms × 3 repeats = 3,000 runs; a sequential design can stop futile arms while
preserving pre-registered error control.

Pre-register verified solve as the primary endpoint and a ±3 percentage-point
equivalence/non-inferiority margin. Model task-clustered solve outcomes with
harness, intervention, benchmark, and their interactions; report marginal effects
with task-cluster bootstrap intervals. Treat first hit, ever-read, gold-edit,
patch creation, testing, and solve as a multi-state process, including censored
non-events rather than conditioning them away. Report tokens by semantic component,
wall time, and cost under an explicitly dated price sheet. Finally, blind-code a
sample of paired discordant transcripts for hypothesis formation, corroboration,
test execution, edit scope, and evidence use, then validate any automated labels
against that rubric.

This framing turns the large existing corpus into the observational discovery and
replication layer, while the new factorial supplies the causal contribution an
top-venue paper needs.


---

## Corrections and additions, 2026-09-21 (paper v2)

All numbers below come from `python -m eval.scripts.paper_v2_analysis`, whose output
is frozen in `docs/paper/submission/numbers_v2.json`.

**Correction — ReAct token count.** In the ReAct records `billed_input` is the sum of
`prompt_tokens` over turns, which *already includes* cached tokens; `cached_input` is a
subset. Adding them double-counted. The no-search baseline in `nemotron_v4` is 344,642
tokens per task, not 497,137, and SG fusion's saving against it is −48% (−164,573 tokens),
not −64%. `nemotron_v2` is unaffected (its baseline had no cached tokens). F1 and fig6 are
corrected; r is −0.63, not −0.65.

**Correction — the ReAct `none` arm is not closed-book.** It has no search tool but can
list and read files (`config.py`: "No retrieval (blind file navigation)"). The earlier
manuscript's "a closed-book model solved 35%" was wrong; the arm is now labelled
"no search (list and read only)" in the paper and in `fig_ablation`.

### The funnel, every panel (without → with the retriever, all paired tasks)

| setting | n | 1st-search hit | saw gold code | edited gold | patch | solved | differ / both edited |
|---|---|---|---|---|---|---|---|
| ReAct (v4) | 100 | 0 → 80 | 88 → 97 | 65 → 87 | 74 → 90 | 35 → 42 | 17 / 10 |
| ReAct (v2) | 100 | 0 → 76 | 95 → 97 | 80 → 82 | 86 → 88 | 44 → 45 | 13 / 7 |
| Claude Code 2.1.206–211 | 100 | 73 → 91 | 100 → 99 | 96 → 96 | 100 → 100 | 74 → 75 | 13 / 11 |
| Claude Code 2.1.274 | 100 | 69 → 85 | 99 → 100 | 97 → 96 | 100 → 100 | 79 → 69 | 12 / 10 |
| Claude Code 2.1.278 | 100 | 90 → 92 | 97 → 98 | 97 → 97 | 100 → 100 | 71 → 73 | 10 / 9 |
| Claude Code, SWE-rebench | 49 | 67 → 82 | 98 → 98 | 90 → 92 | 98 → 98 | 55 → 53 | 11 / 10 |
| Claude Code, rebench prose | 50 | 66 → 80 | 100 → 96 | 88 → 88 | 100 → 98 | 50 → 44 | 7 / 7 |
| Codex CLI (plain) | 100 | 61 → 88 | 99 → 98 | 94 → 95 | 99 → 100 | 53 → 59 | 16 / 14 |
| Codex CLI (shipped) | 100 | 61 → 85 | 99 → 99 | 94 → 93 | 99 → 100 | 53 → 58 | 19 / 14 |

Across the six production-agent comparisons (plain Codex, not shipped), outcomes differ on
69 tasks; on 61 of them (88%) both arms had edited a gold file. The retriever made an agent
newly edit a gold file on 3 tasks in all six settings together. "Saw gold code" = a Read of
the gold file or a search/retriever result containing its path (Claude); the gold path in
any completed command or tool result (Codex); `files_read` (ReAct).

### What the native agent did differently under 2.1.278 (same 100 tasks)

| per task | 2.1.274 | 2.1.278 |
|---|---|---|
| turns with a tool call | 10.9 | 4.7 |
| before / after first edit | 5.8 / 5.1 | 2.8 / 1.9 |
| file reads (chars returned) | 3.2 (10,135) | 1.1 (1,811) |
| searches (of which list-files) | 3.5 (0.9) | 1.7 (0.0) |
| code or test runs | 1.5 | 0.1 |
| ran code after first edit | 40 / 100 | 2 / 100 |
| output tokens | 5,093 | 1,447 |

The Claude Code changelog for 2.1.275–2.1.278 documents no change to how the search or
read tools return results (error-handling fixes only) and none to default effort. It does
document removal of the TaskOutput tool (2.1.277) and loading of account-synced skills and
plugins into terminal sessions (2.1.276), which match the observed 28→25 tools (with our
denial of the two web tools) and 59→63 advertised commands. The behaviour change is
measured, not attributed.

**Why the retriever costs more on 2.1.278 — addition, not substitution.** Native → with
retriever, same release: tool turns 4.7 → 7.2, all before the first edit (2.8 → 5.0; after
1.9 → 2.0); reads 1.1 → 1.3 and searches 1.7 → 1.5 (not replaced), plus 1.0 tool-lookup and
1.3 retriever calls.

**Phase split, uniform definition** (tool-using turns; before / after first edit; tasks that
ran code after editing): 2.1.206–211 6.7→6.6 / 6.8→3.8 / 49→11; 2.1.274 5.8→5.8 / 5.1→3.2 /
40→5; rebench 12.8→9.7 / 7.5→4.9 / 25→7; 2.1.278 2.8→5.0 / 1.9→2.0 / 2→2. Codex (actions,
n=99): native 4.5 / 1.3 / 4; shipped 6.3 / 0.9 / 1; plain 4.4 / 1.1 / 1. Codex checks too
rarely in any arm to separate retrieval from the shipped "don't re-run" instruction.

**Web-use sensitivity** (drop tasks where either arm used WebSearch/WebFetch): run 1 tokens
−23.6% → −26.2%, solves 74→75 becomes 69→69 (n=92); SWE-rebench −24.4% → −10.1% (n=44);
rebench prose −20.4% → −15.0% (n=45). Direction holds everywhere; the SWE-rebench magnitude
is sensitive to five tasks.

**Integration on Codex, one release:** shipped +39.6% [+17, +66] (+53.5k tokens, +2.1 turns),
plain +5.7% [−7, +21] (+7.7k, −0.3); the shipped orientation tool ran on 89/100 tasks and added
1.8 actions before the first edit; solved 58 and 59 (native 53); rank-1 on invoked tasks 48% vs 71%.
