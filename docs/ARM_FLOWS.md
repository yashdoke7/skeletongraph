# Arm Flows & Architecture — what each pipeline actually does

_The reference for the paper's "what helps what" conclusions. For every arm: its
flow, its architecture, the tool surface the agent sees, what it is good at, where
it breaks, and the one sentence the paper draws from it. Last updated 2026-06-06._

> **Reading guide.** Three buckets: **METHOD** (`sg`, `sg-rerank` — reported in the
> main table), **BASELINES** (`bm25`, `grep`, `hybrid`, `none` — standard 5-tool
> harness), **COMPETITORS** (`cbmem`, `aider`, `graphify` — native tools, systems
> comparison). **ABLATIONS** (`sg-chain`, `sg-embed`, `sg-seed`, `summary-dense`,
> and the component toggles) live in the ablation table and exist to answer *which
> design choice carries the gain* — not to be headline numbers.

---

## 0. The common substrate (held identical for every arm)

Same model, same tasks, same `edit_file`/`submit`, same per-(task,arm) isolated
workspace, same SWE-bench verifier, same bounded-context policy (last 5 tool
outputs kept verbatim, older stubbed). **The ONLY thing that varies is how the
agent finds code.** That is what isolates the retrieval contribution.

What differs per arm is the **tool surface** (`profiles.py::build_profile`):
- METHOD + SG ablations get `search_code` + `read_symbol` + `expand`.
- BASELINES get the standard `search_code` + `read_file` (+ `list_files`).
- COMPETITORS get their own native tools (below).

---

## METHOD

### `sg` — SkeletonGraph lean structural core  *(precision / token-floor op-point)*

**Architecture.** Zero-LLM tree-sitter index built once per repo:
- *skeleton table* — every function/class with signature, span, file (the unit of
  retrieval and of `read_symbol` fetch);
- *call graph* — caller/callee edges for `expand`.

**Flow (per query).**
1. **Entity extraction** — pull identifiers the issue names (symbols, dotted paths,
   error types) from the issue text.
2. **Symbol resolution** — match those entities against the skeleton table.
3. **Gated graph expansion** — only when the match is ambiguous/thin, walk one hop
   of callers/callees to recover neighbours (gated = doesn't fire on clean hits, so
   precise queries stay precise).
4. **Centrality rerank** — order survivors by PageRank on the call graph (central
   functions first).
5. **Weak-entity BM25 fallback** — if entity resolution returns ~nothing, seed a
   narrow BM25 pass so the query isn't empty-handed.

**Tools the agent sees.** `search_code` (the above), `read_symbol` (fetch ONE
function body by `file::symbol` — the token-savings mechanism, no whole-file dump),
`expand` (callers/callees on demand).

**Good at.** Lowest tokens of any arm; best *function-precision* when the issue
names a resolvable symbol (you fetch exactly the right function and nothing else).
**Breaks on.** *Recall.* When the issue is described semantically and names no
resolvable symbol (a behaviour, not a function), entity resolution finds little and
`sg` can miss the target file entirely.

**Paper sentence.** *Structure gives precision and a token floor, but pure
structural retrieval is recall-limited — it only fires when the issue names a
symbol it can resolve.*

---

### `sg-rerank` — bm25 recall pool, reordered by SG structure  *(recall / rank op-point — the WINNER)*

This is exactly the user's mental model: **`sg` is only weak at recall, so seed
recall with bm25 and let structure do the ranking. `sg + bm25 = sg-rerank`.**

**Flow (generate-then-rerank).**
1. **Recall (generate)** — take BM25's *wide* candidate pool over function bodies.
   This inherits BM25's recall (it surfaces the right function even when the issue
   never names it).
2. **Rerank (structure)** — reorder that pool so structurally-confirmed and
   issue-named functions rise: call-graph centrality + entity overlap push the true
   target up the list.
3. **Fetch** — `read_symbol` pulls the chosen functions at function granularity.

**Crucially it is NOT.** Not RRF (no reciprocal-rank fusion of two score lists),
and NOT per-file-capped. In ablation, both of those choices *cost* recall — fusion
dilutes BM25's strong hits, the per-file cap drops co-located functions. `sg-rerank`
keeps the raw pool and only *reorders*.

**Result (nemotron_v2, Verified, n=100).** Best file recall **.924**, best function
recall **.404 / 56% funcHit**, at **~175k** tokens — i.e. BM25's recall AND SG's
rank, at SG's token cost. No other arm has both.

**Good at.** Everything the headline cares about: recall + rank + tokens together.
**Breaks on.** Nothing it's *worse* than baselines at; its ceiling is the
file→function gap (right file ~.92, right function ~.40) shared by all arms.

**Paper sentence.** *Structure is best deployed as a reranker over cheap lexical
recall and a function-level fetcher — not as a standalone graph retriever.*

---

### Why BOTH `sg` and `sg-rerank` stay in the main table

They are **two operating points of one method**, and reporting both is the honest,
stronger story:

| | `sg` | `sg-rerank` |
|---|---|---|
| Recall source | structural entity match | BM25 pool |
| Strength | lowest tokens, function-precision on symbol-named issues | best recall+rank overall |
| When it wins | issue names a concrete symbol | issue is semantic / names no symbol |
| Op-point | precision / token floor | recall / rank (winner) |

The Pareto pair makes the contribution legible: **both dominate every baseline and
competitor on tokens; `sg-rerank` additionally dominates on retrieval correctness.**
Keeping only one would hide that the *structural core itself* is the cheap precise
floor and that adding lexical recall to it is what buys the win — which is the
entire "what helps what" argument. (`sg-noagent`, the single-shot variant, stays in
the ablation table as the "is the agent loop worth it" point.)

---

## BASELINES (standard 5-tool harness)

### `bm25` — flat BM25 over function chunks
**Flow.** Tree-sitter chunk the repo into functions → Okapi BM25 over bodies →
`search_code` returns ranked functions. **Good at.** Recall (it's the recall pool
`sg-rerank` reranks). **Breaks on.** Noisy rank (lexical collisions) and token cost
— imprecise rank → the agent reads more candidates. *Headroom that `sg-rerank`
closes by reranking exactly this pool.*

### `grep` — ripgrep-style lexical, file-level
**Flow.** Literal/regex line match over raw files → returns files. **Good at.**
Cheap, zero index, exact-string lookups. **Breaks on.** File-level not
function-level (agent still reads whole files); no semantic or structural notion.

### `hybrid` — BM25 ∪ dense (SBERT) + cross-encoder rerank, file-level
**Flow.** Union BM25 and dense-embedding candidates → cross-encoder rerank → return
files. The "dense RAG" baseline. **Good at.** Bridges some vocabulary gaps BM25
misses. **Breaks on.** Heaviest compute, still file-level, and the dense signal
doesn't beat structural rank on these tasks — *dense RAG is not the answer here.*

### `none` — no retrieval (blind navigation)  *(the contamination control)*
**Flow.** No `search_code`. Agent navigates via `list_files`/`read_file` only. NOT
a long-context dump — nothing is pre-pasted. **Result.** **44.0%** solve rate, and
no pipeline is significantly above it (McNemar p>0.3). **Paper sentence.** *On
pre-cutoff popular repos, the model localizes from the issue text alone often enough
that solve rate is nearly retrieval-insensitive — so pass@1 cannot be the headline;
tokens and retrieval correctness must be.*

---

## COMPETITORS (native tools — systems comparison)

### `cbmem` — Codebase-Memory (MCP knowledge graph)
**Architecture.** Builds a code knowledge graph; **requires an LLM** for semantic
edge inference; indexes **asynchronously** (must poll until `nodes>0`).
**Native tools.** `cbmem_search`, `cbmem_trace` (call trace), `cbmem_snippet`,
`cbmem_arch` (architecture overview). **Result.** 44.4% / 254k tok / funcR
.228(33%). **Breaks on.** Function recall (graph nodes ≠ the changed function);
per-repo LLM build cost SG doesn't pay. **Paper sentence.** *A graph competitor
needs an LLM just to build its graph, and still under-recalls the target function.*

### `aider` — RepoMap (tree-sitter + PageRank), injected
**Architecture.** Closest prior art to SG: tree-sitter symbols ranked by PageRank.
**Flow.** The repo map is **injected into the system prompt** (no search tool — the
model reads the map and navigates). **Result.** 43.9% / **1167k tok** (6×) /
$0.319, $0.140 cached. **Breaks on.** The static map is re-sent **every turn** →
token cost explodes; `cost(cached)` softens but doesn't erase it. **Paper
sentence.** *Front-loading a whole-repo map is paid on every turn; a fetch-on-demand
tool is an order of magnitude cheaper for the same navigation.*

### `graphify` — knowledge graph (tree-sitter + NetworkX + Leiden), 70× claim
**Architecture.** Extracts a repo into `graphify-out/graph.json` (tree-sitter
structure + **LLM semantic extraction** + Leiden community detection); answers with
compact subgraphs. **Requires an LLM to extract** — `graphify extract` refuses
without an API key (proven by selftest). **Native tools.** `graphify_search`
(subgraph), `graphify_explain` (node detail). **The 70× token claim is tested
end-to-end here, not as token-count math.** **Model for extraction:** see
`docs/plan.md` / EVAL_PLAN — routed through **NIM** at the *same model tier as the
agent* to avoid a model-tier confound, and the extraction LLM cost is **counted
against graphify**. **Paper sentence.** *graphify's 70× is a per-query-vs-read-all
ratio; inside the loop its end-to-end cost includes a per-repo LLM graph build that
SG (zero-LLM tree-sitter) never pays.*

---

## ABLATIONS — which design choice carries the gain (ablation table only)

These exist to prove `sg-rerank` is the right call by showing the alternatives
**don't beat it** — each is genuinely good at *something*, which is the point: no
single alternative wins the aggregate of recall + tokens + pass@1.

| arm | flow (one line) | good at | why it's not the headline |
|---|---|---|---|
| `sg-chain` | full-body BM25 recall + SG precision + **short graph paths** + consensus files → minimal evidence chain | relational issues (A-calls-B reasoning) | graph traversal adds tokens without beating rerank on recall |
| `sg-embed` | SG∪BM25 pool → **dense (SBERT) semantic** rerank | vocabulary-gap / paraphrased issues | dense rerank ≈ structural rerank on rank, costs more compute |
| `sg-seed` | extract issue **tracebacks/backtick symbols** → augment query → SG | issues with stack traces / explicit symbols | overlaps `sg` on symbol-named issues, no aggregate gain |
| `summary-dense` | rank by function **SUMMARIES (purpose)** not raw code — "recall what a function *does*" | intent/behaviour-described issues | summary quality variance; doesn't beat code-pool recall at n=100 |
| `sg-nograph` | `sg` minus graph expansion | isolates: does gated graph earn its keep on recall? | component toggle, not a product |
| `sg-norerank` | `sg` minus centrality rerank | isolates PageRank's marginal value | component toggle |
| `sg-noagent` | retrieve once → one generation → patch (no agent loop) | "is the agent worth it" measure | single-shot, separate question |

**The conclusion the table licenses:** different SG variants win different *task
shapes* (graph-heavy → relational, summary → intent, seed → traceback), but
**`sg-rerank` is the one that wins the aggregate** because it fixes SG's only real
weakness (recall) with the cheapest possible recall source (BM25) and keeps
structure where structure is strongest (ranking + function-level fetch). That is
"what helps what": **recall comes from lexical breadth; correctness and cheapness
come from structural rank + function fetch.**

---

## One-paragraph synthesis (for the paper's Analysis section)

Across eight pipelines on a common substrate, solve rate is retrieval-insensitive
(`none`=44%, all n.s.), so the discriminating axes are **retrieval correctness** and
**token cost**. Lexical methods (`bm25`) carry recall but waste tokens on noisy
rank; structural retrieval (`sg`) is precise and cheap but recall-limited; dense and
graph competitors (`hybrid`, `cbmem`, `graphify`) add compute and an LLM build step
without improving function recall; map injection (`aider`) pays its map every turn.
**Composing cheap lexical recall with structural rerank and function-level fetch
(`sg-rerank`) is the only configuration that is simultaneously highest-recall and
lowest-token** — and it builds its index with no LLM at all. Token reduction is a
*consequence* of fetching the right function, not the goal; the goal is better
retrieval, and the agent loop is the only place that distinction is visible.
