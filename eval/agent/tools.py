"""Agent tools — identical action space across all arms.

Every arm gets the same five tools. Only `search_code` dispatches differently:
its backend is the arm under test. That parity is the experiment's control —
any difference in outcome is attributable to retrieval, not to tool affordances.

Tool set: search_code, list_files, read_file, edit_file, submit.
"""

from __future__ import annotations

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
                if not os.environ.get("SG_EVAL_QUIET"):
                    # workspace layout: WORKSPACE_ROOT/<run_id>/repo
                    # so self.repo.parent.name is the run_id (informative);
                    # self.repo.name is always literally "repo" (useless).
                    rid = self.repo.parent.name
                    sys.stderr.write(
                        f"[{self.backend}] first search returned 0 file paths "
                        f"(run={rid}, query={query!r:.60}). "
                        f"If every task hits this, the backend is wedged.\n"
                    )
            # SG arms only: did the semantic embedding index actually build?
            # If embeddings.npz is absent the run is SG-minus-embeddings — flag
            # it so aggregate.py never reports a degraded run as real SG.
            if self.backend.startswith("sg"):
                self.embeddings_used = (
                    self.repo / ".skeletongraph" / "embeddings.npz").exists()
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

# Every SkeletonGraph variant (full + ablations). These route through the
# summary-aware path so the agent receives tier-2 previews (the C2 mechanism).
_SG_BACKENDS = {"sg", "sg-nograph", "sg-gatedgraph", "sg-fullgraph",
                "sg-norerank", "sg-nosummary", "sg-noembed", "sg-learned",
                "sg-weakfallback"}


def _ensure_sg_on_path() -> None:
    """Put eval/ and the repo root on sys.path so backends + skeletongraph import."""
    eval_dir = str(Path(__file__).resolve().parent.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _sg_config(backend: str):
    """Build the SGConfig for an SG arm, flipping off the ablated component."""
    from skeletongraph.config import SGConfig
    cfg = SGConfig()
    if backend == "sg-nograph":
        # Direct entity matches only — tests whether graph traversal drives gain.
        cfg.enable_graph_expansion = False
    elif backend == "sg-gatedgraph":
        # Proposed policy: graph only for exact/small seed sets, broad-impact
        # intents, or explicit graph/blast-radius requests.
        cfg.graph_expansion_policy = "gated"
    elif backend == "sg-fullgraph":
        # Historical policy: expand whenever the query mode permits graph
        # traversal. Useful as an ablation against the gated policy.
        cfg.graph_expansion_policy = "always"
    elif backend == "sg-norerank":
        # Same candidates, no hub/centrality reweight — tests PageRank's value.
        cfg.enable_centrality_rerank = False
    elif backend == "sg-nosummary":
        # No tier-2 summary text delivered to the agent — the C2 ablation.
        cfg.enable_summaries = False
    elif backend == "sg-noembed":
        # No semantic embeddings (lexical + graph only) — the embedding part of
        # the C3 ablation.
        cfg.enable_embeddings = False
    elif backend == "sg-weakfallback":
        # Full SG + the gated weak-entity recall booster (ablation arm). Only
        # fires when the entity match is ambiguous, so precise-match precision is
        # preserved. Tests whether it recovers the semantic-mismatch misses.
        cfg.enable_weak_entity_fallback = True
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
    _ensure_sg_on_path()
    from skeletongraph.engine import SGEngine
    from skeletongraph.summary.local import build_local_summary

    engine = SGEngine(project_root=repo, config=_sg_config(backend))  # auto-builds
    # sg-learned: a trained classifier picks the retrieval mode for this query
    # (the learned curator), passed via mode_hint. Same index, only mode changes.
    mode_hint = None
    if backend == "sg-learned":
        try:
            from curator.curator import predict_mode   # eval/ is on sys.path
            mode_hint = predict_mode(query)
        except Exception:
            mode_hint = None                            # no model → rule-based
    res = engine.heuristic_query(query, top_n=k, mode_hint=mode_hint)
    store = engine.get_store()
    include = backend != "sg-nosummary"

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
