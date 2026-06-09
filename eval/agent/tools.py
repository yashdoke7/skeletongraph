"""Agent tools — identical action space across all arms.

Every arm gets the same five tools. Only `search_code` dispatches differently:
its backend is the arm under test. That parity is the experiment's control —
any difference in outcome is attributable to retrieval, not to tool affordances.

Tool set: search_code, list_files, read_file, edit_file, submit.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

# ── extra tool schemas for native pipelines (systems comparison, final v2) ────
# Composed per-arm by profiles.py; NOT in the baseline set. SG exposes these so
# the agent fetches functions / follows the call graph instead of reading whole
# files — the SG pipeline, not a bare ranker.
READ_SYMBOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_symbol",
        "description": "Read ONE function/class body by its `file::symbol` id (from "
                       "a search result) — far fewer tokens than read_file. Prefer "
                       "this over read_file for a specific function.",
        "parameters": {
            "type": "object",
            "properties": {"fqn": {"type": "string",
                                   "description": "file::symbol id, e.g. pkg/mod.py::Class.method"}},
            "required": ["fqn"],
        },
    },
}
EXPAND_SCHEMA = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Show the callers and callees of a function (by `file::symbol`) "
                       "so you can follow the real control flow across the codebase.",
        "parameters": {
            "type": "object",
            "properties": {"fqn": {"type": "string",
                                   "description": "file::symbol id to expand"}},
            "required": ["fqn"],
        },
    },
}

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
            if name == "read_symbol":
                self._searches_since_read = 0          # a read → reset the nudge
                from .sg_tools import read_symbol
                return read_symbol(self.repo, args.get("fqn", ""))
            if name == "expand":
                self._searches_since_read = 0
                from .sg_tools import expand
                return expand(self.repo, args.get("fqn", ""))
            if name == "cbmem_search":
                _ensure_sg_on_path()
                from backends.cbmem import search_native
                return self._native_search(args.get("query", ""),
                                           int(args.get("k", 10) or 10), search_native)
            if name == "cbmem_trace":
                _ensure_sg_on_path()
                from backends.cbmem import trace_calls
                return trace_calls(args.get("function_name", ""), self.repo,
                                   args.get("direction", "both"))
            if name == "cbmem_snippet":
                self._searches_since_read = 0
                _ensure_sg_on_path()
                from backends.cbmem import code_snippet
                return code_snippet(args.get("qualified_name", ""), self.repo)
            if name == "cbmem_arch":
                _ensure_sg_on_path()
                from backends.cbmem import architecture
                return architecture(self.repo)
            if name == "graphify_search":
                _ensure_sg_on_path()
                from backends.graphify import search_native
                return self._native_search(args.get("query", ""),
                                           int(args.get("k", 10) or 10), search_native)
            if name == "graphify_explain":
                self._searches_since_read = 0
                _ensure_sg_on_path()
                from backends.graphify import explain
                return explain(args.get("symbol", ""), self.repo)
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
        # ── none arm: no retrieval backend — stop the model immediately ──────
        # Without this guard the model calls search_code 6+ times/task, gets
        # "No results." each time, and wastes turns trying different queries.
        # Return a single directive so it pivots to list_files/read_file on the
        # first call. Does NOT touch _search_calls or first_search_hits so the
        # no-retrieval baseline metrics stay clean.
        if self.backend == "none":
            return ("No search backend available — this arm tests agent "
                    "navigation without retrieval. Explore with list_files "
                    "then read_file candidate source files directly.")
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

    # ── native-pipeline search (cbmem etc.) — records recall like _search ────

    def _record_first_hits(self, result: str) -> None:
        """Set first_search_hits (file paths) from a 'Ranked results' block, once."""
        if self._search_calls != 0:
            return
        seen: List[str] = []
        for line in (result or "").splitlines():
            m = re.match(r"^\s*\d+\.\s+(\S+)", line)
            if m:
                f = m.group(1).split("::")[0].replace("\\", "/")
                if f and f not in seen:
                    seen.append(f)
        self.first_search_hits = seen

    def _native_search(self, query: str, k: int, fn) -> str:
        """A system's native search tool (e.g. cbmem_search). Records the same
        first-search recall signal the harness scores, then returns fn's output."""
        if not (query or "").strip():
            return ("ERROR: search requires a non-empty query. Describe the "
                    "symptom or name a symbol.")
        try:
            result = fn(query, self.repo, k)
        except Exception as e:
            result = f"ERROR: {type(e).__name__}: {e}"
        self._record_first_hits(result)
        self._search_calls += 1
        self._searches_since_read += 1
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
# (the C2 mechanism). The trial arms run leaner: they drop summaries/rerank and
# add routing, fusion, or path-aware selection around the structural core.
_SG_BACKENDS = {"sg", "sg-nograph", "sg-gatedgraph", "sg-fullgraph",
                "sg-norerank", "sg-nosummary", "sg-noembed", "sg-learned",
                "sg-weakfallback",
                "sg-lean", "sg-router", "sg-fusion", "sg-chain", "sg-chain-nopath",
                # ablations AROUND the new lean default (final-run stage)
                "sg-full", "sg-summary", "sg-embed", "sg-hybrid-fusion",
                "sg-dense-rerank", "sg-keyword-dense", "sg-rerank", "sg-seed"}

# The lean product default: structural core only. Summaries + embeddings are
# OFF — in the heuristic_query path (eval + IDE sg_search) embeddings feed only
# the confidence score, never candidate ranking (see resolver.py), and summaries
# are agent-facing previews, not a ranking signal. Both cost build/refresh
# compute + per-result tokens for no measured retrieval gain on one-shot tasks.
# Gated graph + centrality rerank + BM25 weak-entity fallback stay (recall).
_LEAN_SG = {"sg", "sg-lean", "sg-router", "sg-fusion", "sg-chain",
            "sg-chain-nopath", "sg-learned"}


# ── experimental-arm helpers: adaptive routing + rank fusion ─────────────────
# These power the trial arms. They are deliberately small and dependency-
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


# ── sg-chain: path-aware evidence-chain retrieval ────────────────────────────

_CHAIN_RRF_K = 60
_CHAIN_PATH_SEEDS = 6
_CHAIN_PATH_MAX_DEPTH = 3
_CHAIN_PATH_MAX_PAIRS = 24
_CHAIN_PER_FILE_LIMIT = 2


def _dedupe_ranked(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _file_of(fqn: str) -> str:
    return fqn.split("::", 1)[0]


def _path_bridge_counts(
    graph,
    left_seeds: List[str],
    right_seeds: List[str],
    valid_fqns: Set[str],
    *,
    max_depth: int = _CHAIN_PATH_MAX_DEPTH,
    max_pairs: int = _CHAIN_PATH_MAX_PAIRS,
) -> Counter:
    """Count nodes on short graph paths between two seed sets.

    This is the small "evidence chain" step. BM25 often finds the issue-text
    lexical clue, while SG finds the structural symbol. Nodes that connect both
    are better evidence than nodes that are merely relevant in isolation.
    """
    counts: Counter = Counter()
    pairs = 0
    for left in _dedupe_ranked(left_seeds)[:_CHAIN_PATH_SEEDS]:
        if left not in valid_fqns:
            continue
        for right in _dedupe_ranked(right_seeds)[:_CHAIN_PATH_SEEDS]:
            if right == left or right not in valid_fqns:
                continue
            path = graph.shortest_path(left, right, max_depth=max_depth)
            if not path:
                continue
            for node in path:
                if node in valid_fqns:
                    counts[node] += 1
            pairs += 1
            if pairs >= max_pairs:
                return counts
    return counts


def _score_chain_candidates(
    sg_ranked: List[str],
    bm25_ranked: List[str],
    path_counts: Counter,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Score candidates as an evidence set, not as isolated documents."""
    sg_ranked = _dedupe_ranked(sg_ranked)
    bm25_ranked = _dedupe_ranked(bm25_ranked)
    scores: Dict[str, float] = defaultdict(float)
    reasons: Dict[str, Set[str]] = defaultdict(set)

    def add_ranked(items: List[str], label: str, weight: float) -> None:
        for rank, fqn in enumerate(items):
            scores[fqn] += weight / (_CHAIN_RRF_K + rank + 1)
            reasons[fqn].add(label)

    add_ranked(sg_ranked, "structural", 2.4)
    add_ranked(bm25_ranked, "lexical", 2.0)

    sg_files = {_file_of(fqn) for fqn in sg_ranked[:10]}
    bm25_files = {_file_of(fqn) for fqn in bm25_ranked[:10]}
    consensus_files = sg_files & bm25_files
    both_fqns = set(sg_ranked) & set(bm25_ranked)

    for fqn in set(scores) | set(path_counts):
        if fqn in both_fqns:
            scores[fqn] += 0.080
            reasons[fqn].add("consensus")
        if _file_of(fqn) in consensus_files:
            scores[fqn] += 0.035
            reasons[fqn].add("same-file")
        if path_counts.get(fqn, 0):
            scores[fqn] += 0.070 + 0.015 * path_counts[fqn]
            reasons[fqn].add("graph-path")

    ranked = sorted(scores, key=lambda fqn: (-scores[fqn], _file_of(fqn), fqn))
    return ranked, {fqn: sorted(labels) for fqn, labels in reasons.items()}


def _diverse_top_k(ranked: List[str], k: int, per_file: int = _CHAIN_PER_FILE_LIMIT) -> List[str]:
    counts: Counter = Counter()
    out: List[str] = []
    for fqn in ranked:
        file_path = _file_of(fqn)
        if counts[file_path] >= per_file:
            continue
        counts[file_path] += 1
        out.append(fqn)
        if len(out) >= k:
            break
    return out


def _chain_reason_summary(fqn: str, reasons: Dict[str, List[str]], store) -> str:
    labels = ", ".join(reasons.get(fqn, [])) or "ranked"
    summary = ""
    sk = store.skeleton_table.get(fqn)
    if sk is not None:
        try:
            from skeletongraph.summary.local import build_local_summary
            summary = build_local_summary(sk)
        except Exception:
            summary = getattr(sk, "docstring", "") or getattr(sk, "signature", "")
    summary = " ".join((summary or "").split())
    prefix = f"signals: {labels}"
    return (prefix + (f"; {summary}" if summary else ""))[:180]


def _retrieve_chain(query: str, repo: Path, k: int, use_path: bool = True):
    """sg-chain: lexical recall + structural precision + graph-path selection.

    Unlike sg-fusion, this keeps function granularity and (when use_path=True)
    explicitly rewards short paths connecting BM25 issue-text hits with SG
    structural hits. The agent still receives raw file paths/FQNs and can read
    the code; the summary string is only a compact navigation hint.

    use_path=False is the sg-chain-nopath ablation: identical BM25+SG fusion and
    consensus scoring, but NO graph-path bridging — isolates whether the
    evidence-chain step (the novel contribution) actually beats plain fusion.
    """
    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine

    depth_k = max(k * 3, 18)
    cfg = _sg_config("sg-chain")
    engine = SGEngine(project_root=repo, config=cfg)
    res = engine.heuristic_query(query, top_n=depth_k)
    store = engine.get_store()
    sg_ranked = [c.skeleton.fqn for c in res.candidates]

    from backends.bm25_flat import retrieve as bm25_retrieve
    bm25_ranked = bm25_retrieve(query, repo, depth_k)

    valid_fqns = set(store.skeleton_table)
    if use_path:
        path_counts = _path_bridge_counts(
            store.graph, sg_ranked, bm25_ranked, valid_fqns,
        )
    else:
        path_counts = Counter()      # ablation: no evidence-chain bridging
    ranked, reasons = _score_chain_candidates(sg_ranked, bm25_ranked, path_counts)
    ranked = [fqn for fqn in ranked if fqn in valid_fqns]
    picked = _diverse_top_k(ranked, k)
    return [(fqn, _chain_reason_summary(fqn, reasons, store)) for fqn in picked]


def _retrieve_rerank(query: str, repo: Path, k: int) -> List[str]:
    """sg-rerank: bm25 RECALL pool, REORDERED by SG structural confirmation.

    The sg-chain post-mortem: RRF-blending SG+bm25 and capping 2/file gave the
    best RANK but LOST bm25's recall (0.31 < 0.36). sg-rerank fixes that — it is
    NOT a blend and NOT capped:
      1. take bm25's wide candidate pool (keeps bm25's recall exactly),
      2. pull to the top the candidates SG structurally confirms (in SG's order)
         and the ones whose symbol name the issue literally mentions,
      3. leave the rest in bm25 order.
    So recall == bm25's pool while rank inherits SG's precision. Generate (bm25)
    then rerank (SG) — the standard localize-then-rank recipe, not RRF.
    """
    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine
    from backends.bm25_flat import retrieve as bm25_retrieve

    pool = bm25_retrieve(query, repo, max(k * 5, 40))     # wide bm25 recall pool, ordered
    if not pool:
        return []
    cfg = _sg_config("sg-chain")                          # lean SG
    engine = SGEngine(project_root=repo, config=cfg)
    sg_res = engine.heuristic_query(query, top_n=40)
    sg_order = {c.skeleton.fqn: i for i, c in enumerate(sg_res.candidates)}

    q_idents = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")}

    def _name(fqn: str) -> str:
        return fqn.split("::")[-1].split(".")[-1].lower()

    confirmed = sorted((f for f in pool if f in sg_order), key=lambda f: sg_order[f])
    cset = set(confirmed)
    named = [f for f in pool if f not in cset and _name(f) in q_idents]
    nset = set(named)
    rest = [f for f in pool if f not in cset and f not in nset]
    return (confirmed + named + rest)[:k]


# ── sg-embed: semantic (dense) rerank of the structural+lexical pool ──────────

def _func_text(store, repo: Path, fqn: str) -> str:
    """fqn + signature + docstring + body for one function (for dense encoding)."""
    sk = store.skeleton_table.get(fqn)
    if sk is None:
        return fqn
    body = ""
    try:
        lines = (repo / sk.file_path).read_text(encoding="utf-8",
                                                errors="replace").splitlines()
        s = max(0, getattr(sk, "line_start", 1) - 1)
        e = min(len(lines), getattr(sk, "line_end", s + 1))
        body = "\n".join(lines[s:e])
    except Exception:
        pass
    return "\n".join([fqn, getattr(sk, "signature", "") or "",
                      getattr(sk, "docstring", "") or "", body])


def _retrieve_embed(query: str, repo: Path, k: int) -> List[str]:
    """sg-embed: SG structural ∪ bm25 candidate pool, reranked by DENSE semantic
    similarity (query vs function). Activates the semantic signal the heuristic SG
    path leaves inert — catches functions whose PURPOSE matches the issue even
    when the exact symbol isn't named. Concept = structural recall + semantic rank.
    """
    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine
    cfg = _sg_config("sg-chain")
    engine = SGEngine(project_root=repo, config=cfg)
    res = engine.heuristic_query(query, top_n=max(k * 4, 30))
    store = engine.get_store()
    from backends.bm25_flat import retrieve as bm25_retrieve
    pool = _dedupe_ranked([c.skeleton.fqn for c in res.candidates]
                          + bm25_retrieve(query, repo, max(k * 4, 30)))
    pool = [f for f in pool if f in store.skeleton_table]
    if not pool:
        return []
    texts = [_func_text(store, repo, f) for f in pool]
    from backends.dense import rank as dense_rank
    return dense_rank(query, pool, texts, k,
                      repo / ".skeletongraph" / "dense_cache", tag="sgembed")


# ── sg-seed: use the issue's structured signals (tracebacks/symbols) as seeds ──

_SEED_STOP = {"the", "and", "for", "this", "that", "with", "from", "Error",
              "True", "False", "None", "Exception", "self", "return", "import"}


def _extract_issue_symbols(text: str) -> Set[str]:
    """High-signal code symbols an issue LITERALLY names — the precise anchors
    that prose retrieval misses. Strongest signal first:
      • Python traceback frames:  File "x.py", line N, in func   -> func
      • backtick code:            `Foo.bar`, `func_name`
      • code-fence declarations:  def/func/function/class NAME
      • error/exception types:    SomethingError / SomethingException
      • dotted paths:             pkg.module.Class.method
    """
    t = text or ""
    syms: Set[str] = set()
    # (1) traceback frames — the single most precise signal: names the function.
    syms |= set(re.findall(r'File\s+"[^"]+",\s*line\s+\d+,\s*in\s+([A-Za-z_]\w+)', t))
    # (2) backtick code spans.
    syms |= set(re.findall(r"`([A-Za-z_][\w.]{2,})`", t))
    # (3) declarations inside code fences.
    for block in re.findall(r"```[\s\S]*?```", t):
        syms |= set(re.findall(r"\b(?:def|func|function|class)\s+([A-Za-z_]\w+)", block))
        syms |= set(re.findall(r"\b([A-Za-z_]\w*(?:\.\w+)+|[A-Z][A-Za-z0-9]{2,})\b", block))
    # (4) error/exception type names anywhere in prose.
    syms |= set(re.findall(r"\b([A-Za-z_]\w*(?:Error|Exception|Warning))\b", t))
    # (5) dotted attribute paths in prose.
    syms |= set(re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\.[A-Za-z_]\w*)+)\b", t))
    # (6) generic "in NAME" (weaker; only if nothing stronger found below).
    syms |= set(re.findall(r"\bin\s+([a-z_]\w{3,})\b", t))
    return {s for s in syms if len(s) >= 3 and s not in _SEED_STOP}


def _augment_with_symbols(text: str) -> str:
    """Append extracted symbols so SG's intent analysis also seeds on them."""
    syms = _extract_issue_symbols(text)
    if not syms:
        return text or ""
    return (text or "") + "\n\nRelevant symbols: " + " ".join(sorted(syms)[:20])


def _is_test_path(fqn: str) -> bool:
    p = fqn.split("::", 1)[0].replace("\\", "/").lower()
    return ("/test" in p or p.startswith("test") or "/tests/" in p
            or "_test." in p or "conftest" in p)


def _seed_fqns_for(symbols: Set[str], store) -> Set[str]:
    """Resolve issue symbol NAMES to exact FQNs in the index (suffix match) so
    they can be passed as HARD seeds (top structural score). Prefers exact
    short-name matches; a dotted `Class.method` must match the FQN tail.

    SOURCE before tests: an issue's traceback names the buggy SOURCE function, and
    the same short name often also exists in a test file (e.g. Header.fromstring in
    both header.py and test_header.py). Seeding on the test pollutes the anchor, so
    drop test-file matches UNLESS every match is a test (rare; then keep them)."""
    if not symbols:
        return set()
    shorts = {s.split(".")[-1] for s in symbols}
    out: Set[str] = set()
    for fqn in store.skeleton_table:
        tail = fqn.split("::", 1)[-1]          # symbol part of file::symbol
        last = tail.split(".")[-1]
        if last in shorts or tail in symbols or any(tail.endswith("." + s) for s in symbols):
            out.add(fqn)
    non_test = {f for f in out if not _is_test_path(f)}
    return non_test or out


def _retrieve_seed(query: str, repo: Path, k: int) -> List[str]:
    """sg-seed: SG structural retrieval HARD-ANCHORED on the exact functions the
    issue names (tracebacks/code/backticks). Issue-named functions get the top
    structural score; the augmented query + weak-entity BM25 fallback supply
    recall AROUND them. Targets the file->function headroom (function precision)."""
    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine
    cfg = _sg_config("sg-chain")                          # lean SG + graph paths
    engine = SGEngine(project_root=repo, config=cfg)
    store = engine.get_store()
    symbols = _extract_issue_symbols(query)
    seeds = _seed_fqns_for(symbols, store)
    # cap so a noisy issue can't flood the seed set; keep the most specific
    # (dotted, then shorter) FQNs — those are the precise anchors.
    if len(seeds) > 12:
        seeds = set(sorted(seeds, key=lambda f: (-f.count("."), len(f)))[:12])
    res = engine.heuristic_query(_augment_with_symbols(query), top_n=k,
                                 seed_fqns=seeds or None)
    return [c.skeleton.fqn for c in res.candidates]


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
    # sg-fusion / sg-chain drive sg-lean internally; both keep embeddings off
    # by design because the tested contribution is retrieval policy, not dense
    # indexing.
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

    # PIN entity-first for EVERY eval SG arm. SGConfig.bm25_primary now defaults to
    # True — that is the MCP/CLI/hooks PRODUCT default (engine-side sg-rerank, set
    # in 02ef8a4) and must NOT leak into the eval. bm25_primary=True makes the
    # resolver pull a 25-wide BM25 pool on EVERY query and add ALL of it to the
    # candidate set, so high-centrality lexical neighbors (esp. TEST files that
    # repeat the symbol name) outrank the exact entity match → recall@1 collapses
    # (measured: v2 0.73 → v3 0.48 for lean sg; django-14725 rank 1→None; sympy-
    # 24066 rank 2→9). v2 had no such field (pure entity-first). This is the "pin
    # lean sg bm25_primary=False" commits 01005d9/26f2933 described but never wrote.
    # Set BEFORE the branches so the ablation arms (sg-nograph/-norerank/-full/...)
    # are covered too — none of them, nor the dense arms (which use
    # enable_hybrid_fusion / dense_rerank), need bm25_primary=True. It ALSO fixes
    # sg-rerank/sg-seed/sg-embed: their internal SG-confirmation pass calls
    # _sg_config("sg-chain"), so a BM25-polluted SG order degraded their rank.
    cfg.bm25_primary = False

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
    if backend == "sg-hybrid-fusion":
        cfg.enable_embeddings = True
        cfg.enable_summaries = False
        cfg.enable_hybrid_fusion = True
        cfg.enable_dense_fallback = False
        return cfg
    if backend == "sg-dense-rerank":
        cfg.enable_embeddings = True
        cfg.enable_summaries = False
        cfg.dense_rerank = True
        cfg.enable_dense_fallback = False
        return cfg
    if backend == "sg-keyword-dense":
        cfg.enable_embeddings = True
        cfg.enable_summaries = False
        cfg.keyword_embedded_dense = True
        # Base it off the hybrid fusion so we see how keywords affect the merged score
        cfg.enable_hybrid_fusion = True
        cfg.enable_dense_fallback = False
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
    # sg-fusion and sg-chain are ensembles, not a single SGEngine pass — handle
    # separately.
    if backend == "sg-fusion":
        return _retrieve_fusion(query, repo, k)
    if backend == "sg-chain":
        return _retrieve_chain(query, repo, k)
    if backend == "sg-chain-nopath":
        return _retrieve_chain(query, repo, k, use_path=False)
    # sg-rerank and sg-seed have DEDICATED retrieval logic, not a config-driven
    # SGEngine pass. They are in _SG_BACKENDS (so search_code routes here), so we
    # MUST dispatch them explicitly — otherwise they fall through to the generic
    # heuristic_query path below and silently behave like plain `sg` (this is the
    # bug that made sg-rerank = sg and sg-seed error out). Wrap their FQN lists in
    # the (fqn, summary) contract this function returns.
    if backend == "sg-rerank":
        return [(fqn, "") for fqn in _retrieve_rerank(query, repo, k)]
    if backend == "sg-seed":
        return [(fqn, "") for fqn in _retrieve_seed(query, repo, k)]

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

    if backend in ("summary-bm25", "summary-dense"):
        # Rank by function SUMMARIES (local/deterministic) instead of raw code.
        # Same chunking as bm25; bare-FQN output → only the ranking source differs.
        from backends.summary_search import retrieve
        method = "dense" if backend == "summary-dense" else "bm25"
        return retrieve(query, repo, k, source="local", method=method)

    if backend == "sg-rerank":
        # bm25 recall pool reordered by SG structural confirmation (generate-
        # then-rerank). Targets bm25's recall AND sg-chain's rank.
        return _retrieve_rerank(query, repo, k)

    if backend in ("sg", "sg-chain", "sg-nograph", "sg-norerank", "sg-noagent",
                   "sg-full", "sg-summary", "sg-weakfallback", "sg-hybrid-fusion",
                   "sg-dense-rerank", "sg-keyword-dense") or backend.startswith("sg-learned"):
        return _retrieve_seed(query, repo, k)       # traceback/symbol-seeded SG

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
        from backends.graphify import retrieve
        return retrieve(query, repo, k)

    if backend == "aider":
        from backends.aider_repomap import retrieve         # tree-sitter+PageRank
        return retrieve(query, repo, k)

    raise ValueError(f"unknown backend: {backend}")
