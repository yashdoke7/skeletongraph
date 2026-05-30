"""Agent tools — identical action space across all arms.

Every arm gets the same five tools. Only `search_code` dispatches differently:
its backend is the arm under test. That parity is the experiment's control —
any difference in outcome is attributable to retrieval, not to tool affordances.

Tool set: search_code, list_files, read_file, edit_file, submit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

# OpenAI-format tool schemas advertised to the model.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for code relevant to a query. "
                           "Returns ranked file paths with short snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What to look for — symptom, "
                                             "symbol name, or description."},
                    "k": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a directory (relative to repo root).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file, optionally a line range (1-indexed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file. old_str must "
                           "appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Call when the fix is complete. Ends the task.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ── Loop-breaker budgets (identical for EVERY arm → no bias; just removes the
# pathological 40-turn tail that pollutes all arms' turn/token metrics). ──────
_MAX_UNIQUE_SEARCHES = 8     # global unique-query budget per task
_MAX_FAILED_EDITS = 4        # per-file failed edits before we surface the real lines
_SEARCH_NOREAD_NUDGE = 3     # consecutive searches w/o a read before nudging
# Hard GLOBAL cap on failed edits across the whole run. The per-file counter
# above only *nudges* (and previously reset itself, so a model could thrash
# forever — observed 37 failed edits / 3.7M tokens / 40 turns before MAX_TURNS
# finally stopped it). This is the true circuit-breaker: once a run accumulates
# this many failed edit_file calls total, edit_file is refused; if the model
# keeps hammering it, the run is force-submitted so it ends near ~12 turns
# instead of bleeding to the turn ceiling. 8 is far above any legitimate need
# (real fixes succeed in 1-3 edits; failures are rare) so good runs are untouched.
_MAX_TOTAL_FAILED_EDITS = 8     # whole-run failed-edit budget (hard stop)
_EDIT_BUDGET_IGNORES = 3        # times the budget message may be ignored before force-submit


class ToolExecutor:
    """Stateful tool runner bound to one run's repo + arm.

    Also records the first `search_code` result so Axis-2 retrieval recall can
    be scored from the same run (no separate retrieval pass needed).
    """

    def __init__(self, repo: Path, backend: str):
        self.repo = repo.resolve()
        self.backend = backend
        self.first_search_hits: List[str] = []   # file paths, for Axis 2
        self._search_calls = 0
        self.submitted = False
        # None = not applicable (non-SG arm). For SG arms: True iff the
        # semantic embedding index was actually built — a False here means the
        # run silently degraded to BM25-only and its numbers are NOT real SG.
        self.embeddings_used = None
        self.edits_made = 0                      # successful edit_file calls
        self._empty_submit_warned = False        # block one empty submit only
        # ── loop-breaker state (identical for every arm → no bias) ──────────
        self._query_counts: Dict[str, int] = {}  # normalized query -> times seen
        self._unique_queries: set = set()        # distinct normalized queries
        self._searches_since_read = 0            # searches with no intervening read
        self._failed_edits: Dict[str, int] = {}  # path -> consecutive failed edits
        self._total_failed_edits = 0             # whole-run failed-edit count (hard cap)
        self._edit_budget_ignores = 0            # times the exhausted-budget msg was ignored

    # ── dispatch ───────────────────────────────────────────────────────────

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "search_code":
                return self._search(args.get("query", ""), int(args.get("k", 10) or 10))
            if name == "list_files":
                return self._list(args.get("path", "."))
            if name == "read_file":
                return self._read(args["path"], args.get("start_line"),
                                  args.get("end_line"))
            if name == "edit_file":
                return self._edit(args["path"], args["old_str"], args["new_str"])
            if name == "submit":
                # Reject the first empty submit: an agent that submits without
                # editing anything yields an empty patch (guaranteed fail). Give
                # it exactly one nudge, then allow submit so a genuine "nothing
                # to change" decision is still possible.
                if self.edits_made == 0 and not self._empty_submit_warned:
                    self._empty_submit_warned = True
                    return ("ERROR: cannot submit — you have not edited any "
                            "file yet. The task is to FIX the bug: use "
                            "edit_file to apply the fix, then call submit.")
                self.submitted = True
                return "Submitted."
            return f"ERROR: unknown tool '{name}'"
        except Exception as e:  # tools must never crash the loop
            return f"ERROR: {type(e).__name__}: {e}"

    # ── search_code — the arm under test ───────────────────────────────────

    def _search(self, query: str, k: int) -> str:
        # ── reject empty / whitespace-only queries ──────────────────────────
        # Observed: the model occasionally calls search_code(query="") (1/30
        # llama33 SG runs hit this). Silently returning "No results." wastes
        # a turn and pollutes recall metrics. A short rejection nudges the
        # model to formulate a real query without consuming a search budget.
        if not (query or "").strip():
            return ("ERROR: search_code requires a non-empty query. "
                    "Describe the symptom or name a symbol you want to find.")
        # ── loop-breaker: block identical re-searches & cap unique searches ──
        # The first call always runs (counts start empty), so first_search_hits
        # and recall metrics are never affected. Only wasteful 2nd+ identical
        # queries and beyond-budget searches are short-circuited with a directive.
        norm = " ".join((query or "").lower().split())
        self._query_counts[norm] = self._query_counts.get(norm, 0) + 1
        if self._query_counts[norm] >= 2:
            hint = (f" Top result so far: {self.first_search_hits[0]}."
                    if self.first_search_hits else "")
            return ("NOTE: you already ran this exact search "
                    f"(attempt {self._query_counts[norm]}). Results are identical "
                    "— do NOT search it again." + hint +
                    " read_file a listed path and look for the fix, then edit_file.")
        self._unique_queries.add(norm)
        if len(self._unique_queries) > _MAX_UNIQUE_SEARCHES:
            return (f"NOTE: search budget reached ({_MAX_UNIQUE_SEARCHES} unique "
                    "queries). Stop searching. Based on results so far, read_file "
                    "the most likely file and apply the fix with edit_file.")
        self._searches_since_read += 1

        # SG arms return (fqn, summary) pairs so the agent sees a tier-2 preview
        # per hit (the C2 mechanism). sg-nosummary returns empty summaries; all
        # other arms return bare file/FQN strings. Metrics always use the FQN
        # list, so retrieval recall/precision are unaffected by the summaries.
        summaries: List[str] = []
        if self.backend in _SG_BACKENDS:
            pairs = _retrieve_sg(self.backend, query, self.repo, k)
            hits = [fqn for fqn, _ in pairs]
            summaries = [s for _, s in pairs]
        else:
            hits = _retrieve(self.backend, query, self.repo, k)
        if self._search_calls == 0:
            # record first (successful) call's file ranking for recall (Axis 2)
            seen: List[str] = []
            for fqn in hits:
                f = fqn.split("::")[0]
                if f not in seen:
                    seen.append(f)
            self.first_search_hits = seen
            # Loud warning if a NON-`none` backend returned zero hits on the
            # first call: this is the failure mode that made cbmem/graphify
            # silently report recall=0 on llama33_70b v3 even though both
            # binaries existed but were misconfigured. recall=0 across an
            # entire arm is almost always a wiring bug, not a real result —
            # flag it so the user notices BEFORE writing it into the paper.
            if not seen and self.backend != "none":
                import sys, os
                # ── rate-limited diagnostic ─────────────────────────────────
                # Originally one stderr line per missing first-search. With
                # 5 ablation arms × N tasks where the model picks the same
                # bad query (e.g. "ScalarFormatter" on seaborn-3187, where
                # the symbol is in matplotlib), this floods the console with
                # duplicate warnings. We now emit one line per (arm, query)
                # tuple per process; further hits are silent. Set
                # SG_EVAL_VERBOSE_EMPTY=1 to restore per-call warnings.
                if not os.environ.get("SG_EVAL_QUIET"):
                    verbose = os.environ.get("SG_EVAL_VERBOSE_EMPTY") == "1"
                    cls = type(self)
                    seen_set = getattr(cls, "_empty_warned", None)
                    if seen_set is None:
                        seen_set = set()
                        cls._empty_warned = seen_set
                    key = (self.backend, (query or "").strip().lower()[:80])
                    if verbose or key not in seen_set:
                        rid = self.repo.parent.name
                        sys.stderr.write(
                            f"[{self.backend}] first search returned 0 file paths "
                            f"(run={rid}, query={query!r:.60}).\n"
                        )
                        seen_set.add(key)
            # SG arms: did the semantic embedding index actually build?
            # Only meaningful for arms that INTEND embeddings (sg-full, sg-embed,
            # legacy full-base arms). For the lean default and the no-embed
            # arms, embeddings are off BY DESIGN — report None (N/A) so
            # aggregate's contamination warning doesn't fire on every lean run.
            # For embed-wanting arms, embeddings.npz absent = silent BM25-only
            # degradation (st missing) → flag it.
            if self.backend in _SG_BACKENDS:
                wants_embed = getattr(_sg_config(self.backend), "enable_embeddings", True)
                self.embeddings_used = (
                    (self.repo / ".skeletongraph" / "embeddings.npz").exists()
                    if wants_embed else None)
        self._search_calls += 1
        if not hits:
            return "No results."
        # SG-with-summaries: surface "fqn — summary" so the agent can triage
        # what to open without reading the file (this is what shrinks the
        # consolidation gap). Falls back to bare lines when no summary text.
        if summaries and any(summaries):
            lines = []
            for i, h in enumerate(hits[:k]):
                s = summaries[i] if i < len(summaries) else ""
                lines.append(f"{i+1}. {h}" + (f"\n   summary: {s}" if s else ""))
            result = "Ranked results (file::symbol + summary):\n" + "\n".join(lines)
        else:
            lines = [f"{i+1}. {h}" for i, h in enumerate(hits[:k])]
            result = "Ranked results (file::symbol):\n" + "\n".join(lines)
        if self._searches_since_read >= _SEARCH_NOREAD_NUDGE:
            result += (f"\n\n(You have searched {self._searches_since_read} times "
                       "without reading. read_file the top result now to make progress.)")
        return result

    # ── neutral file tools (identical for all arms) ────────────────────────

    def _list(self, path: str) -> str:
        d = (self.repo / path).resolve()
        if not str(d).startswith(str(self.repo)):
            return "ERROR: path escapes repo"
        if not d.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
        return "\n".join(entries) or "(empty)"

    def _read(self, path: str, start, end) -> str:
        self._searches_since_read = 0   # reading is progress — reset the nudge
        f = (self.repo / path).resolve()
        if not str(f).startswith(str(self.repo)):
            return "ERROR: path escapes repo"
        if not f.is_file():
            return f"ERROR: not a file: {path}"
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, int(start)) if start else 1
        e = min(len(lines), int(end)) if end else len(lines)
        body = "\n".join(f"{i}: {lines[i-1]}" for i in range(s, e + 1))
        return f"{path} [{s}-{e} of {len(lines)}]\n{body}"

    def _edit(self, path: str, old: str, new: str) -> str:
        # ── global circuit-breaker: stop runaway edit-retry loops ────────────
        # Once the run has burned its whole-run failed-edit budget, refuse
        # further edits. If the model ignores that repeatedly, force-submit so
        # the run terminates near here instead of bleeding to MAX_TURNS.
        if self._total_failed_edits >= _MAX_TOTAL_FAILED_EDITS:
            self._edit_budget_ignores += 1
            if self._edit_budget_ignores >= _EDIT_BUDGET_IGNORES:
                self.submitted = True
                return ("ERROR: edit budget exhausted and ignored repeatedly — "
                        "ending the task.")
            return (f"ERROR: edit budget exhausted ({self._total_failed_edits} "
                    "failed edits this task). Do NOT call edit_file again. Use "
                    "read_file to view the exact current text, then submit.")

        f = (self.repo / path).resolve()
        if not str(f).startswith(str(self.repo)):
            return "ERROR: path escapes repo"
        if not f.is_file():
            return f"ERROR: not a file: {path}"
        text = f.read_text(encoding="utf-8", errors="replace")
        
        # 1. Exact match
        n = text.count(old)
        if n == 1:
            f.write_text(text.replace(old, new, 1), encoding="utf-8")
            self.edits_made += 1
            self._failed_edits[path] = 0          # success resets the fail counter
            return f"Edited {path}."

        # 2. Line-ending normalized match (fixes \r\n vs \n issues on Windows)
        text_norm = text.replace('\r\n', '\n')
        old_norm = old.replace('\r\n', '\n')
        n_norm = text_norm.count(old_norm)
        if n_norm == 1:
            # Safely replace using normalized target, keeping file format intact elsewhere
            new_norm = new.replace('\r\n', '\n')
            f.write_text(text_norm.replace(old_norm, new_norm, 1), encoding="utf-8")
            self.edits_made += 1
            self._failed_edits[path] = 0
            return f"Edited {path}."

        if n > 1 or n_norm > 1:
            return f"ERROR: old_str matches multiple times — make it unique."

        # ── loop-breaker: edit-thrash. After repeated misses on the SAME file,
        # feed back the actual current lines so the model can copy exact text.
        # The per-file counter is NOT reset (resetting created a sliding window
        # that let a model thrash indefinitely); the global budget checked at
        # the top of this method is the real hard stop. ──────────────────────
        self._total_failed_edits += 1
        self._failed_edits[path] = self._failed_edits.get(path, 0) + 1
        if self._failed_edits[path] >= _MAX_FAILED_EDITS:
            anchor = next((ln.strip() for ln in old.strip().splitlines()
                           if ln.strip()), "")[:40]
            ctx = [f"{i+1}: {ln}" for i, ln in enumerate(text.splitlines())
                   if anchor and anchor in ln][:6]
            if ctx:
                return ("ERROR: old_str not found after several attempts. The "
                        "closest CURRENT lines in the file are below — copy "
                        "old_str EXACTLY from these (match indentation):\n"
                        + "\n".join(ctx))
            return ("ERROR: old_str not found after several attempts. Use "
                    "read_file to view the exact current text of this region, "
                    "then copy old_str verbatim before editing again.")
        return ("ERROR: old_str not found. Make sure you copy the EXACT text including "
                "leading spaces and indentation, and do not skip any lines.")


# ── retrieval backends ─────────────────────────────────────────────────────

# Every SkeletonGraph variant (full + ablations + experimental trials). These
# route through the summary-aware path so the agent can receive tier-2 previews
# (the C2 mechanism). The three trial arms (sg-lean, sg-router, sg-fusion) run
# leaner: they drop summaries/rerank and, for fusion, ensemble with BM25.
_SG_BACKENDS = {"sg", "sg-nograph", "sg-gatedgraph", "sg-fullgraph",
                "sg-norerank", "sg-nosummary", "sg-noembed", "sg-learned",
                "sg-weakfallback",
                "sg-lean", "sg-router", "sg-fusion",
                # ablations AROUND the new lean default (final-run stage)
                "sg-full", "sg-summary", "sg-embed"}

# The lean product default: structural core only. Summaries + embeddings are
# OFF — in the heuristic_query path (eval + IDE sg_search) embeddings feed only
# the confidence score, never candidate ranking (see resolver.py), and summaries
# are agent-facing previews, not a ranking signal. Both cost build/refresh
# compute + per-result tokens for no measured retrieval gain on one-shot tasks.
# Gated graph + centrality rerank + BM25 weak-entity fallback stay (recall).
_LEAN_SG = {"sg", "sg-lean", "sg-router", "sg-fusion", "sg-learned"}


# ── experimental-arm helpers: adaptive routing + rank fusion ─────────────────
# These power the three trial arms. They are deliberately small and dependency-
# free (regex + arithmetic) so the routing decision is fully inspectable for the
# paper and adds ~zero latency vs a learned classifier.

# A bare identifier or dotted path ("ScalarFormatter", "np.linalg.norm",
# "Foo.bar") — the signal that the user already knows the symbol's name.
_SYMBOL_RE = re.compile(r"^[A-Za-z_][\w.]*$")
# Relational / impact-analysis intent: who calls this, what does it touch, blast
# radius. These are exactly the queries graph traversal exists to serve.
_RELATION_RE = re.compile(
    r"\b(calls?|caller|callers|callee|invoke[sd]?|use[sd]?|used by|depend\w*|"
    r"reference\w*|refer|impact\w*|affect\w*|blast radius|propagat\w*|"
    r"downstream|upstream|trace\w*|ripple)\b",
    re.I,
)
# Whitespace ⇒ the query is a phrase / description, not a single symbol.
_NL_RE = re.compile(r"\s")


def _route(query: str) -> dict:
    """Adaptive per-query plan for sg-router (conditional computation).

    Returns {"graph_policy", "mode_hint"}:
      • relational/impact query → graph "always" + debug_investigate (traverse
        the dependency graph hard; that is the whole point of the query)
      • bare symbol, no whitespace → graph "off" + retrieval_fast (a direct
        lexical/embedding hit is enough; graph expansion only adds noise/cost)
      • natural-language description → graph "gated" + explain (let the gate
        decide; description queries benefit from a little neighborhood)
      • fallback → graph "gated", no mode override (use SG's own classifier)

    graph_policy "off" is applied by the caller as enable_graph_expansion=False;
    "gated"/"always" map straight onto SGConfig.graph_expansion_policy.
    """
    q = (query or "").strip()
    if not q:
        return {"graph_policy": "gated", "mode_hint": None}
    if _RELATION_RE.search(q):
        return {"graph_policy": "always", "mode_hint": "debug_investigate"}
    if not _NL_RE.search(q) and _SYMBOL_RE.match(q):
        return {"graph_policy": "off", "mode_hint": "retrieval_fast"}
    if _NL_RE.search(q):
        return {"graph_policy": "gated", "mode_hint": "explain"}
    return {"graph_policy": "gated", "mode_hint": None}


def _rrf(rank_lists: List[List[str]], k_rrf: int = 60) -> List[str]:
    """Reciprocal Rank Fusion (Cormack et al. 2009, k=60).

    score(item) = Σ_lists 1 / (k_rrf + rank + 1).  Robust, score-free fusion:
    it needs only the ordering from each retriever, so a structural ranker and a
    BM25 ranker (whose raw scores are not comparable) can be combined directly.
    Returns items sorted by fused score, descending.
    """
    scores: Dict[str, float] = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores, key=lambda it: -scores[it])


def _retrieve_fusion(query: str, repo: Path, k: int):
    """sg-fusion: RRF ensemble of structural (SG-lean) + lexical (BM25-flat),
    fused at FILE granularity.

    Rationale from the ablations: BM25 has the best raw file recall (0.90) while
    SG has the best precision/rank; fusing their orderings should keep SG's
    precision near the top while inheriting BM25's recall tail. Lean (no
    summaries) → returns (file, "") pairs to honor the _retrieve_sg contract.
    """
    _ensure_sg_on_path()

    def _files(fqns: List[str]) -> List[str]:
        seen: List[str] = []
        for fqn in fqns:
            f = fqn.split("::")[0]
            if f not in seen:
                seen.append(f)
        return seen

    # structural ranker — SG-lean (graph on, no summary/rerank overhead).
    # Pull deeper than k so the fusion has a tail to draw recall from.
    sg_files = _files([fqn for fqn, _ in _retrieve_sg("sg-lean", query, repo, k * 2)])
    # lexical ranker — flat BM25 over function text.
    from backends.bm25_flat import retrieve as bm25_retrieve
    bm_files = _files(bm25_retrieve(query, repo, k * 2))

    fused = _rrf([sg_files, bm_files])
    return [(f, "") for f in fused[:k]]


def preflight_arm(backend: str, repo: Path) -> str:
    """STRICT mode (SG_EVAL_STRICT=1): confirm an arm can run its EXACT intended
    config before the agent loop burns any tokens. Returns "" if OK, else a
    human-readable error string; the caller aborts the run with stopped="error"
    (excluded from metrics, auto-retried by run_stage).

    Today this guards the embeddings invariant that silently produced the v3
    BM25-only ablation runs: an arm whose config wants embeddings MUST have a
    built embeddings.npz, otherwise we ABORT instead of degrading. sg-noembed is
    the deliberate exception (it ablates embeddings, so nothing to assert).

    NOTE: in the heuristic_query path embeddings feed only the confidence score,
    not candidate ranking (see resolver.py) — so this guard is about run
    *determinism and honesty* (the arm did what its name claims), not about a
    large expected ranking delta. Keeping it strict means the `embeddings_used`
    flag can never silently flip a result's meaning again.
    """
    if backend not in _SG_BACKENDS:
        # Baselines own their deps; cbmem/aider preflight via `--selftest`.
        return ""
    _ensure_sg_on_path()
    cfg = _sg_config(backend)
    # sg-fusion drives sg-lean internally; both keep embeddings on by default.
    if not getattr(cfg, "enable_embeddings", True):
        return ""   # sg-noembed: embeddings intentionally absent.
    try:
        from skeletongraph.engine import SGEngine
        SGEngine(project_root=repo, config=cfg).get_store()   # build or load index
    except Exception as e:
        return (f"STRICT preflight: index build failed for {backend}: "
                f"{type(e).__name__}: {e}")
    if not (repo / ".skeletongraph" / "embeddings.npz").exists():
        return (f"STRICT preflight: arm '{backend}' requires embeddings "
                f"(enable_embeddings=True) but embeddings.npz was not built — "
                f"sentence-transformers is likely missing in this env. Aborting "
                f"instead of silently degrading to BM25-only. Fix: "
                f"pip install 'sentence-transformers>=3.0,<4' then re-run.")
    return ""


def _ensure_sg_on_path() -> None:
    """Put eval/ and the repo root on sys.path so backends + skeletongraph import."""
    eval_dir = str(Path(__file__).resolve().parent.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _sg_config(backend: str):
    """Build the SGConfig for an SG arm.

    SGConfig() dataclass defaults = FULL SG (summaries ON, embeddings ON, gated
    graph, centrality rerank ON). The product default `sg` is now LEAN (see
    _LEAN_SG) — summaries + embeddings OFF. Ablations isolate one decision each
    around that lean default; a few legacy arms keep their old full-base meaning
    so re-running an old stage still means what it used to.
    """
    from skeletongraph.config import SGConfig
    cfg = SGConfig()

    # ── the lean product default (sg + trial/learned arms) ──────────────────
    if backend in _LEAN_SG:
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        # router/fusion add per-query routing / RRF ensemble in _retrieve_sg;
        # learned picks the mode via curator. Config is identical lean otherwise.
        return cfg

    # ── ablations AROUND the lean default (final-run ablation stage) ─────────
    if backend == "sg-full":
        # Lean + BOTH add-ons = the OLD full SG. Reference point: "did going
        # lean cost us anything on recall/precision/pass@1?" (defaults = full).
        return cfg
    if backend == "sg-summary":
        # Lean + tier-2 summaries only. Isolates the summary contribution
        # (expected: no recall change, +tokens — summaries are previews).
        cfg.enable_embeddings = False
        return cfg
    if backend == "sg-embed":
        # Lean + embeddings only. Confirms embeddings are retrieval-inert in the
        # heuristic path (expected: recall/precision ≈ lean sg).
        cfg.enable_summaries = False
        return cfg
    if backend == "sg-nograph":
        # Lean − graph expansion. Direct entity matches only — does gated graph
        # earn its keep on recall?
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        cfg.enable_graph_expansion = False
        return cfg
    if backend == "sg-fullgraph":
        # Lean, always-expand instead of gated — is gating better than always?
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        cfg.graph_expansion_policy = "always"
        return cfg
    if backend == "sg-norerank":
        # Lean − hub/centrality reweight — PageRank's marginal value.
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        cfg.enable_centrality_rerank = False
        return cfg
    if backend == "sg-gatedgraph":
        # Lean, explicit gated policy (== default). Kept for completeness.
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        cfg.graph_expansion_policy = "gated"
        return cfg
    if backend == "sg-weakfallback":
        # Lean + gated weak-entity recall booster (only fires on ambiguous
        # matches, so precise-match precision is preserved).
        cfg.enable_summaries = False
        cfg.enable_embeddings = False
        cfg.enable_weak_entity_fallback = True
        return cfg

    # ── legacy ablation arms (OLD semantics: FULL base minus one component) ──
    # Kept so re-running a historical stage reproduces its original meaning.
    # New work should use the lean-based arms above.
    if backend == "sg-nosummary":
        cfg.enable_summaries = False        # full − summaries
        return cfg
    if backend == "sg-noembed":
        cfg.enable_embeddings = False       # full − embeddings
        return cfg
    return cfg


def _retrieve_sg(backend: str, query: str, repo: Path, k: int):
    """SG retrieval that also returns a tier-2 summary per hit (the C2 path).

    Returns a list of (fqn, summary) pairs. summary is "" for the sg-nosummary
    ablation; otherwise it is the function's tier-2 summary — the stored
    LLM/human summary if one exists, else SkeletonGraph's deterministic *local*
    summary (docstring first line, or name/params/return-type derived). The
    local tier needs no LLM and is fully reproducible, which matters because the
    async summary worker never runs inside an isolated eval workspace.
    """
    # sg-fusion is an ensemble, not a single SGEngine pass — handle separately.
    if backend == "sg-fusion":
        return _retrieve_fusion(query, repo, k)

    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine
    from skeletongraph.summary.local import build_local_summary

    cfg = _sg_config(backend)
    # mode_hint / graph_policy may be overridden per query below.
    mode_hint = None
    if backend == "sg-learned":
        # A trained classifier picks the retrieval mode for this query (the
        # learned curator), passed via mode_hint. Same index, only mode changes.
        try:
            from curator.curator import predict_mode   # eval/ is on sys.path
            mode_hint = predict_mode(query)
        except Exception:
            mode_hint = None                            # no model → rule-based
    elif backend == "sg-router":
        # Adaptive conditional computation: inspect the query and rewrite the
        # graph policy + mode for THIS call only. "off" disables graph expansion
        # outright; "gated"/"always" set the policy the gate honors.
        route = _route(query)
        mode_hint = route["mode_hint"]
        if route["graph_policy"] == "off":
            cfg.enable_graph_expansion = False
        else:
            cfg.graph_expansion_policy = route["graph_policy"]

    engine = SGEngine(project_root=repo, config=cfg)    # auto-builds
    res = engine.heuristic_query(query, top_n=k, mode_hint=mode_hint)
    store = engine.get_store()
    # Summaries are delivered iff the config left them on (off for sg-nosummary
    # and the lean trial arms). Reading the flag keeps this in lockstep with
    # _sg_config rather than re-listing arm names here.
    include = getattr(cfg, "enable_summaries", True)

    out = []
    for c in res.candidates:
        sk = c.skeleton
        summ = ""
        if include:
            stored = store.summaries.get(sk.fqn) or ""
            if stored and not stored.startswith("[pending"):
                summ = stored
            else:
                try:
                    summ = build_local_summary(sk)
                except Exception:
                    summ = ""
            summ = " ".join(summ.split())[:160]   # collapse to one capped line
        out.append((sk.fqn, summ))
    return out


def _retrieve(backend: str, query: str, repo: Path, k: int) -> List[str]:
    """Dispatch to the arm's retrieval implementation. Returns ranked FQNs/files.

    SG arms return FQN-only here; the agent-facing summary text is added by
    ToolExecutor._search via _retrieve_sg. retrieval_eval.py uses this FQN list
    for recall/precision, so summaries never affect the retrieval metrics.
    """
    if backend == "none":
        return []                       # no-retrieval arm: agent reads files blind

    _ensure_sg_on_path()

    if backend in _SG_BACKENDS:
        return [fqn for fqn, _ in _retrieve_sg(backend, query, repo, k)]

    if backend == "bm25":
        from backends.bm25_flat import retrieve
        return retrieve(query, repo, k)

    if backend == "grep":
        from backends.grep_sim import retrieve
        return retrieve(query, repo, k)

    if backend == "hybrid":
        from backends.hybrid import retrieve            # BM25+dense+rerank
        return retrieve(query, repo, k)

    if backend == "cbmem":
        from backends.cbmem import retrieve              # Codebase-Memory CLI
        return retrieve(query, repo, k)

    if backend == "graphify":
        # graphify dropped — see config.py for context. Raise loudly if anyone
        # still has stale runbook commands referencing this backend.
        raise ValueError(
            "graphify backend was removed (CLI-only; not a fair "
            "programmable retrieval lib). The graph competitor is cbmem.")

    if backend == "aider":
        from backends.aider_repomap import retrieve         # tree-sitter+PageRank
        return retrieve(query, repo, k)

    raise ValueError(f"unknown backend: {backend}")
