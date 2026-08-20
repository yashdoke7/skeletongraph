# SkeletonGraph — Verified Findings Ledger

Every number here was recomputed from the run JSONs on 2026-07-22. This is the
backup-of-record: if the paper and this file disagree, **this file is right**.

Conventions:
- "cost" = `imputed_cost`, uniform *within* a harness — the fair unit. Two sources,
  do not mix them across harnesses:
  - **Claude Code arm**: Claude Code's own reported `total_cost_usd` (Anthropic's
    real billing, prompt-cache discounts already applied).
  - **React loop**: computed by `config.impute_cost()` from a fixed published price
    sheet ($0.27/M fresh in, $0.07/M cached in, $1.10/M out). NOT "uncached" —
    cached tokens are billed at the cheap rate, same as production.
- **Token counts include the system prompt and tool definitions**, not just retrieval
  payloads. Claude Code context/turn = `input + cache_read + cache_creation`; react
  loop = the API's `prompt_tokens`. SG registers 7 MCP tools whose schemas sit in the
  re-sent prefix every turn, so **SG's own overhead is charged to SG** — the cost
  comparison is conservative in SG's disfavour, not flattering. This is also why the
  unchanged peak context (44.1k vs 44.9k) is a stronger result than it looks.
- "rec@1" = fractional FILE recall after the FIRST search (`search_calls[0].cumulative_recall`),
  averaged over tasks. NOT hit-rate. On multi-file tasks recall is stricter.
- All arm-vs-arm deltas are PAIRED on the arms' common `task_id` set.
- pass@1 counts a run only when it carries a real (non-null) `resolved` verdict;
  an empty patch counts as a fail.

---

## 0. The headline, in one paragraph

Structural retrieval's advantage is **entity-anchored**: it depends on the issue
text naming a symbol that exists in the repo. Strip those cues and SkeletonGraph's
retrieval edge collapses to zero — yet its **cost advantage persists**. In the
prose-stripped SWE-Verified condition SG's first-search recall is statistically
indistinguishable from lexical grep (−0.006) while still costing **21% less**.
Retrieval quality is therefore *not* the mechanism behind the cost saving; bounding
the agent's exploration is. Combining all three non-LLM retrieval paradigms at once
(lexical BM25 + semantic dense + topological graph = `sg-fusion`) does not escape
this ceiling.

---

## 1. THE 2×2 GRID — memorization × location-cues (Claude Sonnet, Claude Code harness)

All four cells are **paired** (same tasks, both arms). The SWE rows are restricted
to the **same 15 task_ids** as the prose run so raw-vs-prose is a fair comparison.

| Condition | n | cost Δ | turns Δ | rec@1 native→SG | pass@1 native / SG |
|---|---|---|---|---|---|
| SWE-Verified raw (**all 100**) | 100 | **−14.6%** | −21.4% | 0.663 → 0.836 (**+0.173**) | 74/100 / 75/100 |

> **Which recall number the paper uses: 0.862, not 0.836.** The 0.836 above counts
> all 100 tasks, including 3 where the agent never invoked SG at all. Those are
> adoption events, not retrieval failures — scoring them as misses attributes to the
> retriever a decision it never made. Excluding them gives **0.862 (86.2%)**, which is
> the canonical figure in the paper and README. Do not mix the two across artifacts.
| SWE-Verified raw (the 50 prose tasks) | 50 | −8.9% | −15.4% | 0.688 → 0.877 (+0.189) | not adjudicated |
| SWE-Verified **PROSE** (same 50) | 50 | **−7.6%** | −19.2% | 0.736 → 0.859 (**+0.123**) | not adjudicated |
| SWE-rebench raw (unseen repos) | 50 | **−15.0%** | −20.3% | 0.455 → 0.585 (+0.130) | 27/49 / 26/49 |
| SWE-rebench **PROSE** (same 50) | 50 | **−15.7%** | −18.9% | 0.464 → 0.577 (**+0.113**) | 25/50 / 22/50 |

> **THESE ROWS SUPERSEDE THE n=15 VERSION (2026-08).** The prior grid read
> 0.661→0.861 / 0.696→0.762 / 0.577→0.737 / 0.493→0.549 with cost −32.2/−21.1/
> −26.4/−31.3. Every one of those cells moved materially at n=50 and the
> raw-vs-prose collapse they appeared to show is gone (§1b). Do not cite the old
> numbers from any older artifact.
>
> **SWE-prose solve rate is NOT adjudicated** — only 15 of the 50 have `resolved`.
> Deliberate: the cell is a null we do not report, so the Docker verify was skipped.
> Never quote a pass@1 for either SWE row of this grid.

> **RECALL CONVENTION — decided 2026-07-31, applies EVERYWHERE.** Tasks where the
> agent never invoked SG are **excluded** from recall averages (adoption events, not
> retrieval failures). This is the same rule that makes the headline 86.2% not 83.6%.
> Excluded counts at n=50: 2 / 2 / 3 / 4 across the four rows. Cost and turns use
> every paired task — adoption does not change what a run was billed. `_rec1()` in
> `make_paper_figures.py` returns `None` (not `0.0`) so this cannot silently regress.

### 1a. SAMPLING CAVEAT — do not skip this
The 15-task prose subset is **not representative** of the full 100: SG saves
**−32.2%** on those 15 but only **−14.6%** across all 100. Any raw-vs-prose claim
MUST use the matched 15-task rows. Comparing the n=100 figure against an n=15
figure produces a spurious "the effect grows" conclusion. (This error was made and
corrected during analysis; it is the single easiest mistake to re-make here.)

### 1b. What the grid actually shows — REVISED 2026-08 AT n=50 (supersedes the n=15 read)

> **WITHDRAWN: the prose-strip collapse.** Everything the n=15 grid appeared to show
> about location cues was an artifact. Do not restore any version of it.

- **FIRST-ORDER — NULL, and well-powered.** SG's own rec@1 when symbols are removed,
  within-task, pooled over both benchmarks:
  **+0.008, 95% CI [−0.049, +0.068], n=93, P(>0)=0.61, 9/93 tasks moving.**
  The superseded n=15 figure was +0.125, CI [+0.010, +0.260], n=26. n rose 26→93 and
  the CI tightened ~3.5× *onto zero* — this is a real null, not an underpowered one.
- **Manipulation check passes, so the null is about the world, not the treatment.**
  Stripping removed 38% of SWE issue chars (median 35%), 26% of rebench chars.
  **No dose-response:** r = +0.07 (SWE), −0.04 (rebench). The 18 SWE tasks that lost
  >50% of their text changed by +0.046 (CI spans 0). Untouched and gutted tasks are
  indistinguishable.
- **SECOND-ORDER — also null** (was already flagged not-significant at n=15, and stays
  that way): margin change SWE **+0.045, CI [−0.076, +0.171]**; rebench **+0.028,
  CI [−0.088, +0.143]**.
- **The margin is STABLE, not collapsing: +0.113 to +0.189 across all four cells.**
  This is a *positive* result for SG and the opposite of the old framing.
- **What actually moves recall is the REPOSITORY, ~30× the text effect:**
  prose effect −0.017 (SWE) / −0.008 (reb); benchmark effect −0.292 (raw) / −0.282
  (prose). **CAVEAT HARD:** prose axis is within-task; benchmark axis is BETWEEN task
  sets, confounded with repo size/domain/issue-style/patch-selection. It is an UPPER
  BOUND on a memorization effect, never an estimate of one.
- **Cost advantage persists in every cell but is SMALLER than the n=15 read:**
  now **−7.6% to −15.7%** (was −21% to −31%). Still decoupled — the largest margin
  (+0.189) sits with the smallest saving (−8.9%); the two rebench cells save ~2× more
  on a smaller margin. Ranking cells by margin vs by saving gives near-opposite orders.
- **pass@1 never moves.** SWE 75/74 (McNemar p=1.0); reb raw 26/27 (p=1.0);
  reb prose 22/25 (p=0.45). SWE-prose was NOT adjudicated (15/50 verified) — do not
  report a solve rate for that cell anywhere.

### 1c. The ceiling that survives — retrieval quality does not CONVERT
The old "symptom-only localization is unsolvable by similarity" argument is dead: SG
beats lexical on symptom-only issues by +0.123 (SWE) and +0.113 (reb). The surviving,
better-evidenced ceiling is that a large durable retrieval win buys no outcome:

```
sg-fusion rec@1:   0.877  →  0.859  →  0.585  →  0.577
native    rec@1:   0.688  →  0.736  →  0.455  →  0.464
margin:            +0.189    +0.123    +0.130    +0.113     <- FLAT
          (SWE raw) (SWE prose) (reb raw) (reb prose)   all n=50
```
Three legs, all measured:
1. **+0.11..+0.19 recall in every cell, zero solve-rate movement** (above).
2. **Native's CUMULATIVE recall overtakes SG in 3 of 4 cells** — SWE-prose .989 v
   .971; reb-raw .695 v .643; reb-prose .713 v .659. Only SWE-raw stays SG-ahead
   (.938 v .907). Given turns, the agent that explored located code *better*.
3. **Iteration asymmetry:** SG gains +0.05..+0.11 over 1.6–2.3 searches/task; native
   gains +0.19..+0.22 over 3.4–5.4. A ranked list saturates; a grep→read→grep
   feedback loop keeps paying. (The old "+0.000 saturation" number was also n=15;
   it is +0.047 on rebench raw at n=50.)

Reads still displaced hard: 3.75→1.41 (SWE raw), 4.44→1.16, 5.24→2.28, 5.50→2.46.

---

## 2. NEMOTRON REACT-LOOP (nemotron_v4, n=100, verified) — the controlled ablation

Identical action space across arms; only the retrieval backend varies.

| arm | n | pass@1 | rec@1 | funcHit | tokens | turns | cost |
|---|---|---|---|---|---|---|---|
| **fusion (SG)** | 100 | **42.0%** | 0.737 | **57%** | 180,070 | 21.9 | **$0.0523** |
| bm25 | 100 | 41.0% | 0.642 | 43% | 263,922 | 24.6 | $0.0744 |
| graphify (competitor) | 100 | 41.0% | 0.223 | 9% | 274,764 | 25.6 | $0.0776 |
| grep | 100 | 39.0% | 0.647 | 0% | 281,793 | 22.4 | $0.0793 |
| aider (competitor) | 98 | 36.7% | — | — | 1,125,908 | 18.1 | $0.1603 |
| **none (closed-book)** | 100 | **35.0%** | 0.000 | 0% | 344,642 | 23.6 | $0.0661 |
| sg-rerank | 99 | 32.3% | **0.747** | 57% | 274,138 | 23.2 | $0.0553 |
| sg-chain *(incomplete)* | 13 | 23.1% | 0.769 | 69% | 226,719 | 23.9 | $0.0478 |
| sg *(incomplete)* | 15 | 20.0% | 0.489 | 33% | 332,553 | 28.5 | $0.0633 |

**Why this table matters:** on a model that memorizes *less* than Sonnet, the
closed-book floor is **35.0%** and `fusion` reaches **42.0%** — a real +7pp
retrieval effect, and it is simultaneously the **cheapest** arm ($0.0523 vs grep
$0.0793) with the **best function-level localization** (57% vs grep's 0%). This is
the cleanest "retrieval helps" evidence in the project, and it complements (does not
duplicate) the Claude deployment study.

**`sg-rerank` row is bugged — excluded from the paper.** This nemotron_v4 `sg-rerank`
arm did not actually run rerank-mode retrieval (unlike the correctly-configured
`sg-rerank` in the nemotron_v2 run and in the agent-free intrinsic ablation, §3b).
Do not cite these numbers anywhere (paper, README, resume, posts) — removed from
Table~tab:react in the paper 2026-07-24.

**Incomplete arms:** `sg-chain` (n=13) and `sg` (n=15) are partial — report as
partial or omit; do NOT compare them against the n=100 arms.

**cbmem:** not present in nemotron_v4 at all. The Claude-side wiring is known-broken
(index built empty). Report cbmem from nemotron_v2 only, explicitly labelled, or omit.

---

## 3. WITHDRAWN / DO-NOT-USE

- **Serena** — verified `serena_calls=0` on **all 11 tasks**; the MCP server never
  entered the tool manifest (headless startup race). The 7/11 "result" is native
  Claude Code with a dead server. **Do not publish as a Serena comparison.** The
  publishable finding is the *structural exclusion* itself: a slow-starting MCP
  server never gets used by a headless agent.
- **"188 successful runs"** — wrong. Actual solved runs (native+sg, claude_v7) = **149 of 200**.
- **"p10 = $0.142"** — wrong; actual p10 of solved runs = **$0.134**. (Minimum solved cost
  **$0.097** is correct.)
- **"rebench-prose 0% pass@1"** — was fabricated when written; rebench-prose is now
  genuinely verified at native 6/14, SG 7/15.
- **Amdahl-style "cost ceiling" arithmetic** (avg − p10 floor) — unsound: p10 tasks are
  cheap because the *edit* is trivial, not because retrieval was free, so a single global
  floor cannot bound savings on the average task. The tail decomposition (§4) is the
  rigorous version of the same intuition; use that instead.

---

## 3b. INTRINSIC RETRIEVAL ABLATION (n=100, SWE-bench Verified, no agent) — NOW IN PAPER

Files: `eval/results/paper_verified_{grep,bm25,bm25-dense,sg-rerank,bm25-dense-sg}_file.json`
(dataset `eval/datasets/graphify_100.jsonl` — despite the name, confirmed same 100
SWE-Verified tasks used throughout the paper, task_ids match e.g. astropy__astropy-12907).
Found 2026-07-23 — this is the "retrieval only benchmark on SWE for the whole 100
tasks" the user recalled; was sitting unused despite the `paper_` filename prefix.

| backend | MRR | recall@10 |
|---|---|---|
| grep | 0.159 | 0.348 |
| bm25 | 0.482 | 0.719 |
| sg-rerank | 0.518 | 0.824 |
| bm25+dense | 0.551 | 0.843 |
| bm25+dense+sg (fusion) | **0.658** | **0.856** |

Monotone: each added signal improves both metrics, fusion wins outright. Corroborates
Table~tab:retrieval (agent-observed) with a clean agent-free measurement — rules out
"it's the agent's search habits, not the retriever" as an alternative explanation.
Now in paper as Table~tab:retrieval-intrinsic, §5.1.

---

## 4. TAIL DECOMPOSITION (claude_v7, n=100) — the original headline, still valid

| percentile | native | +SG | Δ |
|---|---|---|---|
| 50th (median task) | $0.255 | $0.260 | **+1.9%** |
| 75th | $0.581 | $0.489 | −15.8% |
| 90th | $1.010 | $0.752 | −25.6% |
| 95th (worst tasks) | $1.559 | $0.896 | **−42.5%** |
| mean | $0.434 | $0.371 | −14.6% |

Paired bootstrap 95% CI on the mean change: **[−25.3%, −1.2%]**. McNemar on pass@1:
**p = 1.0** (no difference). Retrieval removes the expensive tail; it does nothing for
the typical task.

Stability check (incremental n, claude_v7): the median stays near zero and p90/p95 stay
strongly negative at **every** checkpoint from n=20 to n=100. The shape is stable; p75 is
the noisy column and should not be leaned on.

---

## 5. MECHANISM — why the entity anchor is load-bearing (traced to code)

1. `intent.py::analyze_intent` emits a function/class entity **only** when a query token
   literally equals a known FQN suffix (`short == name or short.endswith("."+name)`).
   Pure symptom prose names no symbol → **zero entities**.
2. `resolver.py` seeds graph expansion, PageRank hub scores, and the same-file bonus
   **entirely** from `target_fqns` (lines ~331, ~350). No entities → the graph contributes
   nothing.
3. With entities empty the structural leg falls back to BM25 (`enable_bm25_fallback=True`)
   while `enable_dense_fallback=False` — so on entity-less prose, **two of fusion's three
   legs collapse into the same lexical signal**, and the semantic leg is the only one left.

Empirically corroborated: on rebench the agent issues prose token-bags
(`"asdict() call on field info fetch_stac_items_updates queryables"`) rather than symbol
names, because it cannot guess symbols in an unfamiliar repo.

**Redundancy measurement (claude_v7, sg-fusion):** 121 of 141 native `Read` calls (86%)
target a file SG had already returned. This is *not* wasted work — at `body_top=0`
`sg_search` returns only a signature line, so the agent must still fetch the code to
construct an exact `Edit`. The doc-claimed "one-line prompt fix" for this is invalid.

---

## 6. TOOL-CALL DISPLACEMENT (claude_v7, avg/task)

Recomputed 2026-08 over the 100 paired tasks (an earlier version of this table read
Bash 4.05 / Read 3.86→1.55 / Grep 3.00→1.43 / Edit 2.69→2.50; those predate a
`--reprocess` backfill of tool counts and are superseded):

| | native | sg-fusion |
|---|---|---|
| Bash | 3.87 | 0.67 |
| Read | 3.75 | 1.41 |
| Grep | 3.10 | 1.34 |
| Edit | 2.50 | 2.30 |
| sg_search / sg_expand | — | 1.53 / 1.81 |
| ToolSearch (MCP deferral tax) | 0.08 | 1.05 |
| **total** | ~13.5 | ~10.4 |

SG collapses native's exploration (Bash 3.87→0.67, Read 3.75→1.41, Grep 3.10→1.34).
Note SG pays a ~1.05/task `ToolSearch` deferral tax that native does not — its true
efficiency is slightly *understated*. **3.75→1.41 is the read-displacement figure the
paper quotes**; do not use the old 3.86→1.55.

---

## 7. FAILED INTERVENTIONS (all real, all reported)

| attempt | change | outcome |
|---|---|---|
| fusion-v2 | rank-1 body inline + grep blocked | −19% cost, **−2 solves** (same 2 tasks, 2 independent runs) |
| fusion-v3 | line-numbered expand (remove re-read need) | −6% cost, **−5 solves @ n=44** (0 v3-only wins vs 5 v1-only) |
| fusion-v4 | context envelope + forced verification | **0/4 pass@1**, +45% cost, +90% tokens |

**Mechanism (v3):** the "redundant" re-reads were the on-ramp to a verify-and-iterate
loop. Removing the *need* to re-read short-circuited the agent into premature unverified
submit. **Mechanism (v4):** forcing verification caused defensive scope creep (15 edits vs
v1's 6) and test-churn.

**The durable finding:** less exploration hurts AND more verification hurts ⇒ v1 sits at a
local optimum; every knob available is worse. Reported as a subsection, not hidden.

---

## 7b. ITERATING `sg_search` SATURATES — the design constraint for any localizer

rec@1 (first search) vs rec@cum (after ALL searches), with searches issued per task:

**RECOMPUTED 2026-08 at n=50** (exclude-convention, cumulative = max over all searches):

| condition | arm | rec@cum | lift over rec@1 | searches |
|---|---|---|---|---|
| SWE raw | native | 0.907 | +0.186 | 3.4 |
| SWE raw | sg-fusion | **0.938** | +0.074 | 1.6 |
| SWE PROSE | native | **0.989** | +0.216 | 4.1 |
| SWE PROSE | sg-fusion | 0.971 | +0.113 | 1.8 |
| rebench raw | native | **0.695** | +0.204 | 5.4 |
| rebench raw | sg-fusion | 0.643 | +0.047 | 2.3 |
| rebench PROSE | native | **0.713** | +0.211 | 5.3 |
| rebench PROSE | sg-fusion | 0.659 | +0.069 | 2.3 |

> The old n=15 table (and its **+0.000 saturation** on rebench raw) is superseded.
> Saturation is not literally zero — it is **+0.047** there — but the *asymmetry* is
> the real finding and it got stronger: SG gains +0.05..+0.11 over 1.6–2.3 searches,
> native +0.19..+0.22 over 3.4–5.4. Quote the asymmetry, not "+0.000".

**The inversion is now 3 of 4 cells, not 2.** Native's cumulative recall beats SG's on
SWE-prose (0.989 v 0.971), rebench-raw (0.695 v 0.643) and rebench-prose (0.713 v
0.659). Only SWE-raw stays SG-ahead (0.938 v 0.907). This got *stronger* with n and is
now one of the paper's load-bearing results.

Native's iteration works better because it has a genuine feedback loop: grep → read a
file → learn the repo's real vocabulary → grep better. SG breaks that loop precisely
because it looks confident — the agent trusts the ranked list and stops exploring
(Read 3.75 → 1.41/task on SWE raw; 5.50 → 2.46 on rebench prose).

⇒ A retrieval layer delivers candidate **files**, not knowledge of the repo, and a
confident ranked list suppresses the agent's own acquisition of that knowledge. The
agent pays until **certain**, not until it has files. That is why recall and cost
decouple — and it is what the revised title ("why better localization does not buy
better outcomes") now refers to.

> **DEAD — do not restore.** The former "direction, not files" ordering argument
> (0.861/0.762/0.737/0.549 ordered by how much direction is available) is withdrawn.
> The ordering still happens to be monotone at n=50 (0.877/0.859/0.585/0.577) but it
> is driven ~30:1 by *which benchmark*, not by cue availability, so it no longer
> supports treating memorization and location cues as one quantity. §1b has the
> decomposition and the confound caveat.
That is the mechanism behind the recall/cost decoupling, and what makes the title
("SkeletonGraph: A Zero-LLM Structural Retrieval Engine for Coding Agents, and Why
Its Gains Land in the Cost Tail, Not the Median" — **retitled 2026-08**, was
"Frontier Coding Agents Don't Have a Retrieval Problem"; see below) mean something
concrete. **The tail-vs-median split is the load-bearing claim now, not the word
"frontier"** — on the open-weight react-loop model retrieval IS worth +7pp (fusion
42.0% vs closed-book 35.0%, §2), which is the counter-example to reading the title as
"retrieval never matters." Retrieval matters little to the median where the model
already knows the repo and a lot where it does not; both remain true and both must
stay in the paper regardless of title wording.

> **TITLE CHANGED 2026-08.** The old title's "frontier" qualifier drew its whole
> defense from one paragraph in the paper ("On the title, and the word 'frontier' in
> it.") — a title that needs a dedicated paragraph to walk itself back is a liability,
> not a feature: it's the first thing a skimming reviewer reads, unqualified. Two
> concrete risks it created: (1) "Frontier Coding Agents" (plural) generalizes a
> single-agent (Claude Code) result — Threats already has to hedge this explicitly;
> (2) an earlier proposed replacement ("retrieval is an anti-catastrophe layer, not a
> median improvement") made a categorical claim about retrieval-as-a-concept that
> directly conflicts with LocAgent's and Codebase-Memory's own reported median-level
> gains (§3 Related Work) — do not resurrect that phrasing as a title. The current
> title scopes the claim to SkeletonGraph specifically (not "retrieval" broadly, not
> "frontier agents" broadly), so it can't be attacked on either axis.

**Design consequence:** a localization loop built on repeated `sg_search` calls is a
dead end — proven, not assumed. Any iterative localizer must inject *new* information
each round, which means structural navigation (`outline` for repo vocabulary/structure,
`neighbors` for graph traversal to code that is causally related but lexically
dissimilar). `sg_search` is the probe, not the engine. Both primitives already exist:
`graph/dependency.py::blast_radius` (callers) and `::dependency_chain` (callees).

---

## 7c. THE BUILT LOCALIZER IS A WASH — `sg-understand` (n=15 rebench-prose, retrieval-only)

> **STATUS: INTERNAL ONLY — NOT CLAIMED IN THE PAPER.** Too lightly tested (n=15,
> retrieval-only, one dataset) to state as a result, and asserting "an LLM in the
> loop also fails" over-negativizes a paper whose story is a positive one (better
> retrieval + lower cost than the frontier agent's own infra) with an honest wall.
> The paper argues the wall from MEASURED data only: the prose-strip collapse (§1c)
> and retriever-iteration saturation (§7b). Keep this record for our own honesty
> about what we ran; do not cite it. If ever revived, it needs full react-loop
> pass@1 at real n, not a retrieval probe.

Built per §7b's prescription: a tool-calling loop (`sg_outline` + `sg_search` +
`sg_neighbors` → `commit`) over a small NIM model, with a never-worse floor
(any no-answer/error/timeout path falls back to plain `retrieve_fusion`). Code:
`eval/backends/localizer.py`; probe backend `sg-understand`; floor baseline
`product-fusion` (the REAL `skeletongraph.retrieval.fusion.retrieve_fusion`, not
the eval-only `bm25-dense-sg` reimpl — those are different code and must not be
cross-compared).

**Result: no reliable lift over the fusion floor.** On the 15-task rebench-prose
slice, ~9/15 tasks are gated as already-confident (fusion has a query-named symbol
in top-3), leaving ~6 addressable. Among those the loop improves ~1 and worsens ~1
per run, aggregate MRR indistinguishable from the floor (floor 0.557; sg-understand
0.534–0.577 across runs). **Replicated across models and merge policies:**

| model | merge policy | agg MRR | vs floor 0.557 |
|---|---|---|---|
| nemotron-super-49b | override (loop-first) | 0.577 | +0.020 (wash) |
| nemotron-super-49b | override (rerun) | 0.543 | −0.014 (wash) |
| llama-3.1-70b-instruct | override | 0.536 | −0.021 (wash) |
| llama-3.1-70b-instruct | additive/RRF blend | 0.534 | −0.023 (wash) |

The additive-merge fix (make the never-worse floor cover RANKING, not just the
error path) fixed one regression and created two others — net still zero. Two
merge mechanics both land at the floor, so the arbitration rule is not the lever.

**Why (traced, not assumed):** (1) prose queries rarely name a symbol, so the loop
can't anchor — same blind spot as the static paradigms; (2) the model cannot tell
when its own `commit` should override fusion's ranking, so correct and incorrect
overrides cancel. This is the empirical backing for the paper's §ceiling second leg.

**Bugs fixed en route (all real, all in `localizer.py`):** dense-timeout race made
the floor itself non-deterministic run-to-run (fixed via prewarm, `warm_repos.py`);
SDK `max_retries` default silently 3×'d every timeout; `tool_choice="auto"` let the
model answer in plain text (→ `"required"`); `max_tokens=700` truncated tool-call
JSON (→ 1024); empty-string `arguments` poisoned conversation history for the next
turn (→ `or "{}"`); thinking-mode not disabled on the reasoning model (5–104s/turn).
None of these changed the qualitative wash — they were prerequisites to measuring it.

**Caveat:** n=15, retrieval-only (not a full react-loop pass@1). Bounded negative
result. `nemotron_l3_v1` (n=15 SWE-Verified react loop, 40% vs 27% pass@1) predates
every fix above, is on the memorization-saturated benchmark, and has no significance
test — DO NOT cite it as a localizer win.

---

## 8. RETRIEVAL LATENCY (product-relevant)

Warm: fusion ~250ms–1s, rerank ~125ms, native `rg` ~60–120ms. Cold dense encode
~11 min/repo on CPU → prewarm required. `SG_DENSE_TIMEOUT_S` default 20s vs a ~24.7s
cold model load is a **reproducibility hazard** — pin to 120 for eval runs.

---

## 9. OPEN / NOT CLAIMED

- rebench-prose n=15, and native is missing one task (`pycqa__isort-2491`, `stopped=error`,
  76 turns, $3.24) → native scores 14, SG 15. Either rerun that task or drop it from both
  arms when reporting.
- `sg-chain` / `sg` nemotron arms incomplete (n=13 / n=15).
- The prose-stripped condition is a **synthetic ablation**, not a naturally occurring
  distribution. Label it as such.
- No claim is made that SG improves pass@1. It does not, anywhere, significantly.
