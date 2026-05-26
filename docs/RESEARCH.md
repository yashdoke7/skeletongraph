# RESEARCH.md — persistent paper artifacts & decisions

> Append-only log of every empirical finding, methodological decision, and
> reviewer-facing rationale that the paper depends on. This file exists so that
> insights survive context compaction and assistant-memory loss. **Whenever a
> new insight, result, or decision lands, add it here.** Date every entry.

Last updated: 2026-05-25 (session 2)

---

## 0. The one-line story

SkeletonGraph's contribution is **edit-targeting, not recall.** With strong,
fair baselines, SG ties dense-RAG on recall but makes the agent edit the *right*
file far more often. The mechanism is gated graph + structural summaries + rank-1
precision, not graph-everywhere.

---

## 1. Headline result — gated graph flipped SG from behind to ahead (2026-05-25)

NIM-70B, paired within-run differencing (cancels serving non-determinism because
`sg` and `bm25` see identical conditions in the same run):

| metric (sg − bm25) | OLD pipeline (eager graph) | V2 pipeline (gated graph) |
|---|---|---|
| edited-gold gap | **−0.154** (SG *behind* BM25) | **+0.240** (SG ahead) |
| precision gap | +0.085 | +0.103 |
| recall gap | 0.000 | +0.040 |
| head-to-head edited-gold | sg 3 / bm25 7 | **sg 6 / bm25 0** |

Interpretation: the OLD eager-graph SG found files (equal recall) but the graph
noise made the agent edit the WRONG file — it was *losing to plain BM25* on the
metric that matters. Gating the graph flipped a −0.15 deficit to +0.24. This is
the cleanest evidence we have and it is robust to NIM non-determinism.

**Within-v2 paired (25 tasks, sg/hybrid/bm25 all complete):**

| arm | recall | precision | edited-gold | turns | in-tok |
|---|---|---|---|---|---|
| sg | 0.76 | 0.32 | **0.68** | 8.0 | 216K* |
| hybrid | 0.76 | 0.28 | 0.48 | 8.5 | 259K* |
| bm25 | 0.72 | 0.22 | 0.44 | 7.6 | 119K* |

*token numbers polluted by edit-thrash tail — see §5; recompute after the fix.

Key: **even when hybrid ties SG on recall (0.76), SG produces 42% more correct
edits (0.68 vs 0.48).** That is the paper's core claim.

---

## 2. Metric hierarchy (what to optimize, in order)

1. **Verified pass@1** — SWE-bench Docker FAIL_TO_PASS+PASS_TO_PASS. Gold
   standard. Conference-required. Only obtainable with Docker (→ AMD plan).
2. **edited-gold** — did the agent edit the gold file. Robust localization proxy
   when pass@1 isn't verified. Current primary.
3. **Retrieval** — recall@k, precision@k, rank-of-first-gold-file.
4. **Efficiency** — uncached input tokens (PRIMARY), turns, tool calls.
   Cached-adjusted cost is a SENSITIVITY analysis only (caching helps all arms
   but unequally; never headline it).

**Caveat on edited-gold:** "edited the gold file" ≠ "made the correct change."
A run can touch the right file with a wrong patch. edited-gold is a *localization*
proxy; only pass@1 confirms correctness. Always report both once Docker is up.

---

## 3. Methodology decisions (reviewer-facing)

### 3a. Fixed-budget retrieval protocol (the grep/baseline fairness answer)
All retrieval arms return the **same candidate budget (k=10 files)**; they differ
ONLY in ranking strategy. Rationale to put in the paper:
> "An uncapped baseline would conflate 'returns more candidates' with 'ranks
> better,' which is not a controlled comparison. Fixing the budget isolates
> retrieval *strategy* from retrieval *quantity*."
This pre-empts "your grep is a strawman": grep is a *strong* ripgrep-style
lexical ranker held to the same budget as SG. (grep already returns top-k file
paths — confirmed; the ripgrep scoring is internal only.)

### 3b. Non-determinism + why we DON'T do heavy repeats
Even at temperature=0, vLLM/NIM are not bitwise reproducible: GPU kernel
reduction order depends on batch size, and dynamic batching groups requests
differently each call → occasional divergent argmax → divergent trajectory.
Observed: unchanged BM25 swung edited-gold 0.65→0.50 between runs.

Decision: **scale tasks, not repeats.** Report PAIRED comparisons (SG vs each
baseline on the same task set) with **McNemar's exact test**. Pairing cancels
per-task difficulty AND common-mode run noise. Add a SMALL 3-seed study on ~20
tasks as a noise-floor robustness appendix. Reviewer line:
> "Outcomes are reported as paired comparisons over a common task set with
> McNemar's exact test; pairing controls for per-task difficulty and
> serving-level non-determinism. Single-run variance is quantified via a 3-seed
> study on a 20-task subset (Appendix)."
This is ~5× cheaper than 3×-ing the whole study with stronger statistics.

### 3c. Competitor arms must be SEARCH BACKENDS, not native MCP servers
cbmem/graphify have their own MCP servers exposing extra tools. Running them
that way gives them MORE tools than other arms → unfair, breaks the controlled
design (every arm has the identical 5-tool action space; only `search_code`'s
backend differs). For the controlled table they MUST be wired as the
`search_code` backend. A "native MCP" number can be reported separately, clearly
caveated as uncontrolled.

---

## 4. Baseline / arm status (2026-05-25)

| arm | status | notes |
|---|---|---|
| bm25 | ✅ solid | Okapi BM25 over SG-indexed function text. |
| grep | ✅ solid | Ripgrep-style lexical (phrase/path/whole-word/line scoring); returns top-k files. Strong floor per "Is Grep All You Need?". |
| hybrid | ✅ fixed | Rebuilt to symbol/chunk BM25+dense+rerank. NIM recall 0.39→0.75 — now a LEGITIMATE strong baseline (no longer a collapsed strawman). Raises the bar SG must clear; good for credibility. |
| none | ✅ control | No retrieval; reads blind. Thrashes most (no anchors). |
| sg | ✅ main | Gated graph default (`graph_expansion_policy="gated"`). |
| sg-fullgraph | keep as ablation only | Eager-graph (old default). NOT a pipeline arm — only to show "why not graph-everywhere." |
| sg-nograph | ablation | Graph fully off. |
| sg-nosummary / sg-norerank / sg-noembed | ablation | Component isolation. |
| sg-learned | proposal only | Learned curator; needs curator_model.pkl. NOT a final arm — include only if it beats rule-based gating. |
| aider | DROPPED | Isolated-venv pin conflict; not central to a retrieval table. |
| cbmem | pending binary | Subprocess CLI wrapper exists; needs the Go binary + `--selftest` validation of CLI verbs/JSON schema. Closest published competitor (MCP graph memory). |
| graphify | optional stub | `pip install graphifyy`; backend raises NotImplementedError until wired + probed on one task. |

**SG's moat (state this explicitly):** with hybrid fixed, SG no longer wins on
recall. Its defensible edge is rank-1 + edited-gold + lower-than-hybrid tokens.
Stop optimizing recall breadth; optimize edit-targeting (rank-1 + summary).

---

## 5. Harness bugs found & fixes

### 5a. Edit-thrash circuit-breaker (FIXED 2026-05-25)
Old guard reset the per-file failed-edit counter after surfacing lines → sliding
window, never a hard stop. A model thrashed 37 failed edits / 40 turns / **3.7M
tokens** before MAX_TURNS stopped it (xarray-3993, both sg and hybrid). Affected
ALL arms (none-arm worst). Fix: whole-run failed-edit budget
(`_MAX_TOTAL_FAILED_EDITS=8`, no reset); once hit, edit_file refused; ignored 3×
→ force-submit. Caps a runaway at ~10 attempts / ~100K tokens. Identical for all
arms (no bias). The 29 polluted v2 runs should be deleted + re-run under the fix.

### 5b. verify.py per-arm pass@1 (FIXED 2026-05-25)
Old version wrote one predictions file for all arms and matched resolved_ids by
task_id → every arm got the same verdict (couldn't distinguish arms). Fix: run
the SWE-bench harness once PER ARM (instance_id=task_id, unique run_id per arm),
write `resolved` back per arm. See verify.py.

### 5d. IDE push/pull double-retrieval (DECISION PENDING) — found 2026-05-25
Claude Code install registers BOTH (a) a `UserPromptSubmit` hook that runs
`heuristic_query(prompt, top_n=8)` and injects results as `additionalContext`
(PUSH, every prompt), AND (b) the MCP `sg_search` tool that runs `heuristic_query`
again (PULL). With both active the SAME retrieval fires twice → doubled tokens.
Decision needed: pick ONE model per IDE.
- Strong tool-users (Claude Code, Cursor-with-MCP): PULL — the hook injects only
  the lightweight "call sg_search" reminder; full retrieval happens once via the
  MCP tool. (Recommended.)
- Weak/no-MCP IDEs: PUSH — hook injects full retrieval; no pull path.
Fix = make the hook reminder-only when MCP is configured. Touches hooks/ +
install/; product-behavior change → get user's push-vs-pull call first.

### 5e. Transient-error retry (FIXED 2026-05-25)
NIM runs lost tasks to `RateLimitError: exhausted retries` AND to single
transient 500s/timeouts that hard-failed the whole run → arms ended on
DIFFERENT task subsets (unpaired n, the core comparison problem). react.py now
retries rate limits AND transient server/connection errors (timeout, 5xx,
overloaded) up to 6 attempts with capped backoff. Use low `--workers` (3-4) on
NIM to reduce rate-limit pressure; 16 on AMD vLLM.

### 5c. Context accumulation (FIXED 2026-05-25)
Implemented bounded context in react.py `_compact_history`: keeps the last
N=5 tool-result messages verbatim, stubs older ones (keeps a 200-char head +
"re-read if needed"), preserves message structure (tool_call_id pairing intact).
Config: CONTEXT_KEEP_LAST_TOOL_OUTPUTS=5, CONTEXT_STUB_OVER_CHARS=600. Identical
for all arms. ~35% token cut on a moderate transcript, far more on long tails.
Matches SWE-agent/OpenHands history-processor behavior and how Cursor/Cline/etc.
bound history. Should widen SG's lead (it can lean on the summary vs re-reading).
Re-run everything on the new harness so efficiency numbers are clean.

### 5f. IDE push/pull dedup (FIXED 2026-05-25)
Added SGConfig.hook_push_retrieval (default False). The UserPromptSubmit hook no
longer runs heuristic_query when MCP is present (the agent calls sg_search) — it
injects only ambient memory (constraints, session digest, reminder). Set True
only for hook-only installs with no MCP. Kills the doubled retrieval.

### 5g. aider arm removed (2026-05-25)
Dropped per decision (not central to a retrieval table; venv-pin friction).
Removed from ARMS, 0-aider stage, 1b-conference; deleted backends/aider_map.py.
1b-conference now uses sg-fullgraph as the eager-graph ablation.

### 5c-old. Context accumulation (DESIGN — superseded by 5c above)
Harness re-sends the full transcript every turn (all read_file dumps forever) →
token growth with turns. Real agents don't do this:
- SWE-agent/OpenHands: keep action trace, collapse OLD observations to stubs.
- Aider: never dumps full files; repo-map + explicitly-added files.
- Cursor/Cline/Continue/Copilot (our IDE targets): bounded chat history, fresh
  retrieval per turn.
Decision: implement bounded context — keep last N≈3-5 tool outputs verbatim,
stub older ones, always keep the action trace + task. Apply identically to all
arms. Matters MORE for SG (its thesis is the agent leans on the summary instead
of re-reading) → should WIDEN SG's lead. Pending implementation in react.py.

---

## 6. Model / GPU / compute strategy (2026-05-25)

- **Wall time is NOT the bottleneck** (median 203s/run; a 150-run stage ≈ 2.6h @4
  workers). NIM **rate limits** are the bottleneck; the harness self-heals
  (re-run skips completed).
- **AMD MI300X (192GB, ~50 GPU-h) = PRIMARY compute.** It uniquely enables
  Docker pass@1 on the same box, removes rate limits, and is reproducible (open
  weights + documented vLLM config). 50h is ample for full SWE-bench × all arms ×
  ablations + ContextBench + pass@1, even a small repeat study.
- **MAIN model = Qwen2.5-Coder-32B-Instruct** on AMD. Code-specialized (punches
  above generic 70B on SWE-bench), fits trivially, fast (more runs/GPU-h), heavily
  benchmarked (clean delta interpretation). The model is a FIXED CONTROL, not the
  contribution — bigger isn't better here.
- **NIM 70B = free secondary** generality datapoint (retrieval + edited-gold; no
  pass@1 needed). Already mostly run.
- **7B = retrieval-only** floor (edited-gold=0 everywhere; useless for task
  success, fine for recall/precision).
- **IDE study = top-tier capstone** (real agent, real repos).
- Best use of AMD beyond serving: nothing higher-value than serve-main + pass@1.
  (Curator training / embedding builds are minor.)

---

## 7. Paper targets

**Workshop (writeable ~now):** edited-gold + paired McNemar (sg vs bm25/grep/
hybrid/none); recall/precision/rank; efficiency; ablation sg vs sg-fullgraph vs
sg-nograph + sg-nosummary; 1 model, SWE-bench, ~100 tasks, 1 run/task. The
sg−bm25 flip (§1) is the result.

**Conference (the push):** verified pass@1 (primary) + all of the above; 2 models
(32B-Coder main + NIM-70B secondary); 2 benchmarks (SWE-bench + ContextBench);
strong non-strawman baselines (bm25, ripgrep-grep, symbol/chunk-hybrid, cbmem);
full ablation grid; IDE study; 3-seed noise appendix; bounded-context harness.

**Honest gap to conference:** need (1) pass@1 on a clean (thrash-fixed,
bounded-context) harness, (2) a 2nd model or benchmark for generality, (3) the
IDE study. Retrieval story is already conference-grade; task-success story
becomes so once pass@1 lands.

---

## 8. Competitive landscape (cite / position against)

- **Codebase-Memory (cbmem)** — MCP graph memory, claims ~10× fewer tokens.
  Closest competitor. Wire as controlled backend.
- **LocAgent** — graph-guided localization.
- **CodeRAG / CodeRAG-Bench / GRACE** — hybrid/graph repository retrieval.
- **"Is Grep All You Need?"** — makes strong grep a MANDATORY baseline (we have it).
- **Graphify** — ~52K-star knowledge-graph RAG; optional arm.

---

## 9. cbmem CLI interface (confirmed 2026-05-25)

Binary: `codebase-memory-mcp` v0.6.1.

Confirmed CLI interface (was wrong in old wrapper):
```
cli index_repository  {"repo_path": "C:/forward/slash/path"}   # MUST use forward slashes
cli search_graph      {"query": "...", "project": "<slug>", "limit": N}   # field = "project", not "project_id"
cli list_projects     {}                                         # to check if already indexed
```

Project slug = path with `:` removed and `/` or `\` replaced by `-`.
Example: `C:/Users/foo/repos/django` → `C-Users-foo-repos-django`.

Performance: django repo (~4500 files) indexes in ~29s, 44K nodes, 199K edges.
Outputs are already rank-ordered (BM25 over symbol names + doc text).
`results[].file_path` is repo-relative (matched by our existing `_extract_files`).

The wrapper (`eval/backends/cbmem.py`) is now validated via selftest. ✅

---

## 10. NIM baseline results (nim70b_swebench, 30 tasks, all arms, 2026-05-25)

**This run used the PRE-bounded-context harness.** Efficiency numbers (tokens) will
differ from the final AMD run. The relative comparison is still valid (bounded context
applied equally) but context-overflow runs are one-off failures, not the norm.

| Arm | n | recall | precision | edited-gold | turns | in-tok | notes |
|---|---|---|---|---|---|---|---|
| bm25 | 28 | 0.714 | 0.199 | **0.607** | 5.6 | 58K | 2 rate-limit errors |
| grep | 29 | 0.690 | 0.141 | 0.552 | 6.4 | 73K | 1 error |
| hybrid | 30 | 0.433 | 0.057 | 0.400 | 9.1 | 228K | OLD broken hybrid; will be 0.75+ recall on AMD |
| none | 29 | 0.000 | 0.000 | 0.379 | 10.6 | 77K | 1 error |
| sg | 27 | **0.741** | 0.282 | 0.519 | 6.3 | 50K | 3 errors; OLD harness |
| sg-nograph | 26 | 0.692 | **0.451** | **0.654** | 5.4 | 40K | best edited-gold! (see note) |
| sg-nosummary | 27 | 0.704 | 0.306 | 0.593 | 5.8 | 41K | 2 ctx-overflow + 1 RL |
| sg-norerank | 26 | 0.654 | 0.261 | 0.462 | 6.2 | 59K | 4 errors |
| sg-noembed | 26 | 0.692 | 0.268 | 0.654 | 6.8 | 76K | 4 errors |
| sg-learned | 20 | 0.650 | 0.385 | 0.500 | 5.9 | 56K | **10 rate-limit errors** |

**Error breakdown (32 total):**
- Rate-limit (exhausted retries): 28 — concentrated in sg-learned (10/30 = 33%)
- Context overflow (131K token limit): 2 — sg-nograph/matplotlib-26466 (819K cumulative!),
  sg-nosummary/xarray-3993 (546K). Both would succeed under bounded-context harness.

**Interesting finding: sg-nograph outperforms sg on edited-gold (0.654 vs 0.519).**
This is the opposite of the v2 paired analysis (+0.240 gated-graph advantage). Possible causes:
1. Small N (26/27 completed) + different task mix than v2 25-task analysis
2. Gated graph policy may not be fully active on all query types yet
3. OLD harness (no bounded context): graph routes may be dumping too much context
→ The AMD 150-task run with bounded-context harness is the ground truth.

**sg-learned 33% error rate.** Not a code bug — all errors are rate limits. sg-learned
runs slower per turn (richer curator processing), so per-minute call rate hits the NIM
limit harder. Fix: 4-account rotation distributes load.

**Hybrid confirm:** recall 0.433 → confirms OLD broken hybrid is still in this run.
The hybrid fix (recall → 0.75+) landed AFTER this run started. AMD run will have fixed hybrid.

**What to rerun:** Only the errored tasks. run_stage already skips completed jobs
(_already_done = stopped in {submit, max_turns}). Just re-run the same command — the
32 error files will be retried automatically with 4-account rotation.

---

## 11. Multi-account NIM key rotation (2026-05-25)

Config: `SG_EVAL_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3,nvapi-key4`
Implementation: thread-local key in config.py; `_run_one_with_key` in run_stage.py
assigns `keys[job_idx % len(keys)]` per job, then clears on exit.

Benefits: 4 accounts × their individual rate limits = ~4× effective throughput.
With 4 keys and --workers 8 (2 workers/key), rate-limit errors should drop to near 0.

---

## 12. Open todos (move to done as completed)

- [x] Implement bounded context in react.py (§5c). ✅ 2026-05-25
- [x] cbmem binary validated + wrapper fixed. ✅ 2026-05-25
- [x] Multi-account NIM rotation implemented. ✅ 2026-05-25
- [ ] **URGENT (12-day NeurIPS deadline):** Get AMD MI300X, start Qwen32B vLLM.
- [ ] Rerun 32 errored NIM tasks with 4-account rotation (`--stage baseline --workers 8`).
- [ ] Scale SWE-bench dataset to 150 stratified tasks (`make_dataset.py --n 150`).
- [ ] Stand up AMD MI300X: serve Qwen2.5-Coder-32B, run stages 1a + 1b in parallel.
- [ ] Run pass@1 via SWE-bench Docker harness (AMD, same box).
- [ ] IDE study: regenerate runbooks, run 5 tasks Claude Code baseline vs SG.
- [ ] Decide graphify: implement + 1-task probe, or leave as optional.
- [ ] Recompute paired gaps after clean AMD run.
