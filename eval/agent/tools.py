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
        hits = _retrieve(self.backend, query, self.repo, k)
        if self._search_calls == 0:
            # record first (successful) call's file ranking for recall (Axis 2)
            seen: List[str] = []
            for fqn in hits:
                f = fqn.split("::")[0]
                if f not in seen:
                    seen.append(f)
            self.first_search_hits = seen
            # SG arms only: did the semantic embedding index actually build?
            # If embeddings.npz is absent the run is SG-minus-embeddings — flag
            # it so aggregate.py never reports a degraded run as real SG.
            if self.backend.startswith("sg"):
                self.embeddings_used = (
                    self.repo / ".skeletongraph" / "embeddings.npz").exists()
        self._search_calls += 1
        if not hits:
            return "No results."
        lines = [f"{i+1}. {h}" for i, h in enumerate(hits[:k])]
        return "Ranked results (file::symbol):\n" + "\n".join(lines)

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
        f = (self.repo / path).resolve()
        if not str(f).startswith(str(self.repo)):
            return "ERROR: path escapes repo"
        if not f.is_file():
            return f"ERROR: not a file: {path}"
        text = f.read_text(encoding="utf-8", errors="replace")
        n = text.count(old)
        if n == 0:
            return "ERROR: old_str not found"
        if n > 1:
            return f"ERROR: old_str matches {n} times — make it unique"
        f.write_text(text.replace(old, new, 1), encoding="utf-8")
        self.edits_made += 1
        return f"Edited {path}."


# ── retrieval backends ─────────────────────────────────────────────────────


def _retrieve(backend: str, query: str, repo: Path, k: int) -> List[str]:
    """Dispatch to the arm's retrieval implementation. Returns ranked FQNs/files."""
    if backend == "none":
        return []                       # no-retrieval arm: agent reads files blind

    # eval/ on path so the existing backends import
    eval_dir = str(Path(__file__).resolve().parent.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if backend == "sg":
        from skeletongraph.engine import SGEngine
        engine = SGEngine(project_root=repo)            # auto-builds index
        res = engine.heuristic_query(query, top_n=k)
        return [c.skeleton.fqn for c in res.candidates]

    if backend == "bm25":
        from backends.bm25_flat import retrieve
        return retrieve(query, repo, k)

    if backend == "grep":
        from backends.grep_sim import retrieve
        return retrieve(query, repo, k)

    if backend == "hybrid":
        # strong baseline — implement eval/backends/hybrid.py (BM25+dense+rerank)
        from backends.hybrid import retrieve            # noqa: validate on box
        return retrieve(query, repo, k)

    if backend == "aider":
        # strong baseline — implement eval/backends/aider_map.py (repo-map)
        from backends.aider_map import retrieve          # noqa: validate on box
        return retrieve(query, repo, k)

    if backend in ("sg-nograph", "sg-norerank", "sg-nosummary"):
        from skeletongraph.engine import SGEngine
        from skeletongraph.config import SGConfig
        cfg = SGConfig()
        if backend == "sg-nograph":
            # Disable graph expansion: only direct entity matches returned.
            # Expects fewer candidates; tests whether graph traversal drives gain.
            cfg.enable_graph_expansion = False
        elif backend == "sg-norerank":
            # Disable hub/centrality signal in ranking.
            # Same candidates, different order; tests PageRank contribution.
            cfg.enable_centrality_rerank = False
        elif backend == "sg-nosummary":
            # Disable Tier-2 summary text in assembled context.
            # Note: heuristic_query returns ranked FQNs (not summaries), so
            # retrieval metrics are identical to sg. The ablation's effect
            # manifests in full-pipeline (engine.query) where the assembled
            # context omits summary descriptions. File-level retrieval rank
            # is unchanged; pass@1 effect requires the assembly path.
            cfg.enable_summaries = False
        engine = SGEngine(project_root=repo, config=cfg)
        res = engine.heuristic_query(query, top_n=k)
        return [c.skeleton.fqn for c in res.candidates]

    raise ValueError(f"unknown backend: {backend}")
