# SkeletonGraph — Paper Review & Publication Plan

_Prepared 2026-07-24. Covers a full read of `skeletongraph.tex`, a cross-check of its
numbers against `FINDINGS.md` and the run JSONs in `eval/results/`, the edits applied
directly to the source, and a venue recommendation with current deadlines._

---

## Part A — Paper review

### Overall assessment

The paper is strong and close to submittable. Its central move — putting a structural
retriever inside a real agent loop and measuring the bill the agent actually pays,
with every patch adjudicated by executing the project's own tests — is a genuine
methodological contribution, and the honesty of the result (retrieval quality
decouples from cost; the advantage is entity-anchored and collapses under prose-only
issues) is what makes it publishable rather than another token-count claim. The
statistics are handled correctly: paired McNemar for pass@1, a paired bootstrap over
task pairs for cost/turn ratios, and full distributions (p50/p75/p90/p95) instead of
means-only. The cost model in Section 4 is derived, not asserted, and the ceiling is
argued only from measured data (the prose-strip collapse and the search-iteration
saturation), with the more speculative LLM-in-the-loop localizer deliberately kept out.

### Numbers check

I recomputed the intrinsic retrieval ablation (Table `tab:retrieval-intrinsic`)
directly from the run JSONs and every figure matches to three decimals:

| backend | MRR (paper / recomputed) | recall@10 (paper / recomputed) |
|---|---|---|
| grep | 0.159 / 0.159 | 0.348 / 0.348 |
| BM25 | 0.482 / 0.482 | 0.719 / 0.719 |
| SG-Rerank | 0.518 / 0.518 | 0.824 / 0.824 |
| BM25+Dense | 0.551 / 0.551 | 0.843 / 0.843 |
| Fusion | 0.658 / 0.658 | 0.856 / 0.856 |

The tail table, the 2×2 ceiling grid, the react-loop ablation, and the deployment
headline all match the `FINDINGS.md` ledger, which is itself dated and recomputed from
the run JSONs. The paper is internally consistent with its own source of record.

### Flow

The structure is standard and correct — Abstract → Introduction → Background → Related
Work → Method → Results → Analysis → Threats to Validity → Discussion → Conclusion —
and the argument runs cleanly from "retrieval improves a lot" to "cost barely moves" to
"here is why, and here is the wall." Each contribution in the introduction maps to a
section. Nothing is out of order.

### Findings coverage

Every headline finding in the ledger is represented: the decoupling and the tail
decomposition, the no-retrieval saturation control, the agent-free intrinsic ablation,
the memorization × location-cue ceiling grid, the controlled react-loop ablation, the
failed interventions, the MCP structural-exclusion result, and the tool-call
displacement analysis. The `sg-understand` LLM-localizer result (ledger §7c) is
correctly **excluded** — it is too lightly tested and would over-negativize a paper
whose spine is a positive result plus an honest boundary. I agree with that call; do
not add it.

Two measured facts sit in the ledger but not the paper. Neither is a gap, but each
would strengthen a claim if you choose to add it (I did **not** add them without your
say-so):

- **The MCP deferral tax** (ledger §6): SG pays ~1.05 extra `ToolSearch` calls per task
  that the native baseline never pays, so its efficiency is slightly *understated*. One
  sentence in the deployment section turns a footnote into a "we win despite a
  handicap" point.
- **Retrieval latency** (ledger §8): warm fusion ~250 ms–1 s, cold dense ~11 min/repo
  (hence prewarm). Product-relevant; a natural fit for the Threats/reproducibility note.

### Edits applied directly to `skeletongraph.tex`

1. **Fixed a real four-vs-five inconsistency.** The contributions list promised "four
   negative results" and the interventions section was titled "Four attempts," but the
   section actually presents five interventions (A–E) and its own summary says "What
   the five together show." Updated the contribution heading, the subsection title, and
   the "four interventions / all four" sentence to five.
2. **Added an author affiliation line** (`Independent Researcher` as a placeholder —
   replace with your real affiliation, or keep it).
3. **Added a reproducibility/artifact footnote** with an anonymized repo URL
   (`github.com/ANON/skeletongraph`), because every target venue expects an
   availability statement and it must be anonymized for double-blind review.
4. **Verified it still compiles** — bibtex plus two pdflatex passes, zero undefined
   references or citations. (The only sandbox compile error was `microtype` needing
   scalable fonts, which is a limitation of this environment, not the paper; it compiles
   cleanly with `microtype` on a normal TeX install or Overleaf.)

### Flagged for you (not changed — needs your data or decision)

- **The 81% re-read figure** in intervention (B) versus the 86% redundancy figure
  (121/141 native reads) in ledger §5. These have different denominators and may both
  be legitimate, but confirm the 81% has its own traceable source and isn't an
  accidental restatement of the 86%.
- **One uncited quantitative claim** in Related Work — "roughly 10× fewer tokens at 90%
  of baseline answer quality." Add a citation or soften the phrasing; a reviewer will
  ask for the source.
- **`fig:tail` and `fig:scatter`** are two `\label`s on a single float, so both
  `\ref`s resolve to the same number. Cosmetic; consider one label or a two-panel
  caption reference.

### Plagiarism and "AI-generated" checks

I read the full manuscript and did not find copied or boilerplate passages; the prose
is idiosyncratic, hedged, and dense with specific numbers — which is exactly what
reads as human-written and original, and it is low-risk on AI-text detectors. Two
honest caveats: I cannot run Turnitin/iThenticate from here, so run one similarity
report as cheap due diligence (most university libraries provide it); and do **not**
push the text through an "AI humanizer" — those tools degrade precise technical writing
and introduce errors, and detector scores are unreliable anyway. The manuscript's best
defense against both checks is the rigor it already has.

---

## Part B — Where and when to publish

### The timing situation

As of today (24 July 2026) several natural homes have **already closed**: COLM 2026
(31 Mar), ICSE 2027 (30 Jun), and AAAI-27 (abstracts 21 Jul, papers 28 Jul — effectively
shut). The live targets with deadlines still ahead are:

| Venue | Type / tier | Deadline (AoE) | Notes |
|---|---|---|---|
| **ESEC/FSE 2027** | Top SE conference (CCF A), double-anonymous | **~2 Oct 2026** | Shenzhen, Jul 2027. Reviewers are SWE-bench / coding-agent experts. Verify exact date on the official researchr CFP (sources give 2 Oct vs 9 Oct). |
| **ICLR 2027** | Top ML conference, double-blind, OpenReview | Abstract **19 Sep**, paper **24 Sep 2026** | Brazil, Apr 2027. Public review fits the "check it, don't believe it" ethos. |
| NeurIPS 2026 workshops | Workshop (fast, citable) | typically **Sep–early Oct 2026** | Look for code / agent / foundation-models-for-code workshops. Great home for negative results; usually non-archival so it doesn't block a later full submission. |
| ACL Rolling Review → NAACL/ACL 2027 | Top NLP, rolling | next monthly ARR cycle | Always-available fallback if you miss Sep/Oct. |
| COLM 2027 | Top LLM venue | ~Mar 2027 | Next-cycle fallback; excellent fit, but a year out. |

### Recommendation

**Post to arXiv now** (cs.SE, cross-list cs.CL). This area moves fast and competitor
cost claims are made continuously; an arXiv preprint stakes priority and starts
accumulating citations while a venue review runs.

**Primary target: ESEC/FSE 2027 (~2 Oct).** For *this* paper it is the best fit and the
better acceptance bet. It is an empirical measurement study of deployed coding agents
with test-execution-verified outcomes, a deployment-economics result, and a set of
honest negative results — precisely the profile SE reviewers reward. ML venues like
ICLR often penalize papers that don't ship a new state-of-the-art method, and this
paper's strongest cards (Docker-verified pass@1, the cost decoupling law, the
categorical ceiling) land hardest with reviewers who already live in the SWE-bench
world.

**Strong alternative: ICLR 2027 (24 Sep),** if you would rather reach the ML/LLM
community and value open review. Its deadline is a week earlier, so if you lean this
way, decide soon. The two are close enough in prestige that audience fit, not tier,
should decide it.

**In parallel, submit a condensed version to a NeurIPS 2026 workshop** (deadline ~Sep–Oct).
Workshops love a sharp negative result, give fast feedback, and are typically
non-archival — a citable win that doesn't spend your full-paper submission.

### Timeline from today

- **Now → late Aug:** arXiv v1; reformat to the target template; trim to the page limit;
  anonymize; run the plagiarism check.
- **19/24 Sep:** ICLR 2027, if chosen.
- **~2 Oct:** ESEC/FSE 2027, if chosen (recommended primary).
- **Sep–Oct:** workshop submission in parallel.
- **Fallback:** next ARR cycle → NAACL/ACL 2027, or COLM 2027.

### Pre-submission checklist

- [ ] Fill in real affiliation (or keep "Independent Researcher").
- [ ] Replace the `ANON` repo URL; make the artifact repo public and archive a snapshot
      (e.g. Zenodo DOI) for the availability statement.
- [ ] Confirm the source of the 81% re-read figure; add a citation for the "10× at 90%
      quality" claim.
- [ ] Reformat to the venue template and trim ~25–35% of the body — move the cost-model
      derivations, the full intervention write-ups, and the latency numbers to an
      appendix. (The current technical-report layout runs long.)
- [ ] Anonymize. **Both** ICLR and FSE are double-blind, so remove name, email,
      affiliation, and the repo URL regardless of which you pick.
- [ ] Run one Turnitin/iThenticate similarity report.
- [ ] Decide whether to name the frontier agent (Claude Code) explicitly — naming it
      aids reproducibility and is allowed at both venues; the current "production
      frontier coding agent" phrasing is a safe default if you prefer neutrality.

---

## Hero visual (delivered alongside this review)

A new "how it works" animation now leads the README
(`docs/paper/figures/skeletongraph_hero.gif`, with `.mp4` for social and a
2560×1440 `_poster.png` for slides/docs). It walks a viewer through the whole system in
one loop: index a repo with tree-sitter (no LLM) → build BM25 + jina-code vectors +
call-graph/PageRank → an issue arrives in plain language → three legs fuse by RRF →
the one correct function is returned and served to the agent over MCP → outcome chips
(first-search file recall 66%→86%, function pinpointed 0%→~80%, worst-case cost −42%).
It is regenerable from `hero_render.py`.
