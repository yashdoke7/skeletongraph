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
| SWE-Verified raw (the 15 prose tasks) | 15 | −32.2% | −38.5% | 0.661 → 0.861 (+0.200) | 11/15 / 12/15 |
| SWE-Verified **PROSE** (same 15) | 15 | **−21.1%** | −22.6% | 0.696 → 0.762 (**+0.065**) | 11/15 / 11/15 |
| SWE-rebench raw (unseen repos) | 15 | **−26.4%** | −35.6% | 0.577 → 0.737 (+0.160) | 10/15 / 9/15 |
| SWE-rebench **PROSE** (same 15) | 15 | **−31.3%** | −29.3% | 0.493 → 0.549 (**+0.056**) | 6/14 / 7/15 |

> **RECALL CONVENTION — decided 2026-07-31, applies EVERYWHERE.** Tasks where the
> agent never invoked SG are **excluded** from recall averages (adoption events, not
> retrieval failures). This is the same rule that makes the headline 86.2% not 83.6%;
> it was previously applied to the headline but NOT to this grid, which is how the
> prose cell came to read −0.006 ("parity") while the headline used exclusion. Under
> one consistent rule the prose cell is **+0.065**, and the honest claim is that
> ~2/3 of the advantage is lost, not that it reaches parity. Excluded counts: 0 / 1 /
> 2 / 3 across the four rows. Cost and turns use every paired task — adoption does not
> change what a run was billed. `_rec1()` in `make_paper_figures.py` now returns
> `None` (not `0.0`) so this cannot silently regress.

### 1a. SAMPLING CAVEAT — do not skip this
The 15-task prose subset is **not representative** of the full 100: SG saves
**−32.2%** on those 15 but only **−14.6%** across all 100. Any raw-vs-prose claim
MUST use the matched 15-task rows. Comparing the n=100 figure against an n=15
figure produces a spurious "the effect grows" conclusion. (This error was made and
corrected during analysis; it is the single easiest mistake to re-make here.)

### 1b. What the grid actually shows
- **Retrieval edge collapses under prose, in both benchmarks — by the same fraction.**
  Memorized: +0.200 → +0.065. Unseen: +0.160 → +0.056. Both lose ~2/3 of the edge.
- **Cost advantage persists in every cell** (−21% to −31%), including the cell where
  retrieval is *identical* to the baseline. → the decoupling.
- **Direction of the prose effect on cost differs by benchmark**: on memorized SWE it
  *shrinks* the advantage (−32.2% → −21.1%); on unseen rebench it *grows* it
  (−26.4% → −31.3%). Do NOT claim a monotone trend.
- **pass@1 is a tie or ±1 in every cell.** Never significant. Do not lead with it.

### 1c. The L3 ceiling — the proof
`sg-fusion` is all three non-LLM paradigms simultaneously (BM25 lexical + jina
dense semantic + graph topological). Its first-search recall as cues are removed
and repos become unfamiliar:

```
sg-fusion rec@1:   0.861  →  0.762  →  0.737  →  0.549      (−36%)
native    rec@1:   0.661  →  0.696  →  0.577  →  0.493      (−25%)
          (SWE raw) (SWE prose) (reb raw) (reb prose)
```
If **any** of the three paradigms could reason from symptom to root cause, recall
would hold when the symbols disappear. It does not — and the lexical baseline falls
with it, so this is a property of the whole non-LLM category, not an SG defect.

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

| | native | sg-fusion |
|---|---|---|
| Bash | 4.05 | 0.59 |
| Read | 3.86 | 1.55 |
| Grep | 3.00 | 1.43 |
| Edit | 2.69 | 2.50 |
| sg_search / sg_expand | — | 1.52 / 1.77 |
| ToolSearch (MCP deferral tax) | 0 | 1.05 |
| **total** | ~13.9 | ~10.4 |

SG collapses native's exploration (Bash 4.05→0.59, Read 3.86→1.55, Grep 3.00→1.43).
Note SG pays a ~1.05/task `ToolSearch` deferral tax that native does not — its true
efficiency is slightly *understated*.

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

| condition | arm | rec@1 | rec@cum | lift | searches |
|---|---|---|---|---|---|
| SWE raw | native | 0.663 | 0.834 | +0.171 | 3.14 |
| SWE raw | sg-fusion | 0.836 | 0.908 | +0.072 | 1.53 |
| SWE PROSE | native | 0.717 | **0.967** | +0.250 | 4.13 |
| SWE PROSE | sg-fusion | 0.711 | 0.883 | +0.172 | 1.87 |
| rebench raw | native | 0.500 | 0.611 | +0.111 | 5.60 |
| rebench raw | sg-fusion | 0.639 | 0.639 | **+0.000** | 2.33 |
| rebench PROSE | native | 0.394 | 0.572 | +0.178 | 5.13 |
| rebench PROSE | sg-fusion | 0.439 | 0.506 | +0.067 | 1.47 |

> **NOTE (2026-07-31): the rec@1/rec@cum columns above use the OLD convention**
> (no-search scored 0.0). Recomputed under the decided convention (exclude), on the
> same kept set for both arms: SWE-prose native 0.696→**0.964**, SG 0.762→**0.946**;
> rebench-raw native 0.577→0.705, SG 0.737→**0.737**; rebench-prose native
> 0.493→**0.715**, SG 0.549→**0.632**. Two things survive intact: the **+0.000
> saturation on rebench raw** (0.737→0.737, unchanged), and the **inversion** —
> native's cumulative recall beats SG's in BOTH prose cells, now visible on rebench
> prose too (0.715 vs 0.632) where it is wider than on SWE prose (0.964 vs 0.946).
> The paper quotes the recomputed numbers.

**On rebench raw, SG issues 2.33 searches and recovers +0.000 recall.** Re-querying
returns the same files: reformulations on an unfamiliar repo all draw from the same
impoverished vocabulary (the issue text), so different phrasings produce the same
ranking. No new information enters the loop.

Native's iteration works *better* (+0.111 to +0.250) because it has a genuine feedback
loop: grep → read a file → learn the repo's real vocabulary → grep better. SG breaks
that loop precisely because it looks confident — the agent trusts the ranked list and
stops exploring (Read 3.86 → 1.55/task). On SWE-prose this backfires: **native's
cumulative recall (0.967) exceeds SG's (0.883)**. SG gets there cheaper, not further.

**NOW THE PAPER'S UNIFYING FRAME (§ceiling, "direction, not files").** Two facts in
this table carry it and are now stated in the paper:
1. Ordering the 4 conditions by *how much direction is available* (memorization
   and/or location cues — the same quantity, held by the model vs supplied by the
   issue) orders SG's rec@1 **exactly**: 0.861 / 0.762 / 0.737 / 0.549. Native does
   NOT order as cleanly (its SWE-prose 0.696 > its SWE-raw 0.661 at n=15) — so rest
   the claim on SG's ordering only, and say so.
2. **Native's cumulative recall OVERTAKES SG's in BOTH prose cells** (convention B):
   SWE-prose 0.964 vs 0.946, rebench-prose 0.715 vs 0.632. Given enough turns the agent
   that explored located the code *better* than the agent handed a ranked list. SG gets
   there in half the searches and cheaper — not further.
⇒ A retrieval layer delivers candidate **files**, not knowledge of the repo, and a
confident ranked list suppresses the agent's own acquisition of that knowledge
(Read 3.86 → 1.55/task, §6). The agent pays until **certain**, not until it has files.
That is the mechanism behind the recall/cost decoupling, and what makes the title
("Retrieval Is Not Reasoning") mean something concrete.

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
