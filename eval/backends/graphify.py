"""Graphify backend — knowledge-graph RAG via tree-sitter entity graph.

Graphify (~52K GitHub stars) builds an entity/relationship knowledge graph
of a codebase using tree-sitter parsing + LLM-driven semantic extraction +
NetworkX + Leiden community clustering. Claimed to outperform pure-vector
RAG on multi-hop code retrieval questions.

  GitHub:  https://github.com/safishamsi/graphify
  Install: pip install graphifyy

CACHE LOCATION: this backend writes its persistent graph to
`<repo>.parent / .graphify/` — a SIBLING of the repo directory, NEVER
inside the repo itself. The same trap that erred 29/30 hybrid runs on
NIM v2 (binary index committed into the patch) is structurally
impossible for graphify because the cache lives outside the git workspace.
The eval/agent/isolation.py workspace .gitignore also lists `.graphify/`
as a second line of defence.

CONTROLLED-EXPERIMENT NOTE: same as cbmem/hybrid, this is wired as the
`search_code` backend — the agent gets the identical 5-tool action space
across all arms; only the retrieval strategy differs. We do NOT expose
graphify's MCP server or its extra tools, because that would give it more
affordance than the other arms and break the controlled design.

Run `python -m eval.backends.graphify --selftest <repo>` to verify the
install + the cache path + the result schema BEFORE running an eval stage.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set

# File-path-like keys we walk the JSON for, mirroring cbmem._extract_files.
_PATH_KEYS = {"file", "path", "file_path", "filepath", "location", "rel_path",
              "relative_path", "uri", "source", "module"}

# Build-cost cap on first call. Graphify's tree-sitter parse + LLM-driven
# semantic extraction can take 30-120s on a large repo; longer than that
# almost certainly means the install is wedged, not that the repo is huge.
_BUILD_TIMEOUT = 300
_QUERY_TIMEOUT = 60


# ── module presence detection ────────────────────────────────────────────────

def _try_import():
    """Import graphifyy lazily; return the module or None if missing."""
    try:
        import graphifyy  # type: ignore
        return graphifyy
    except ImportError:
        return None


def _cache_dir(repo: Path) -> Path:
    """Cache directory — SIBLING of the repo, never inside it.

    Lives at `<repo>.parent/.graphify/`. The whole `<repo>.parent` tree is
    cleaned up by isolation.cleanup_workspace() at run end, so the cache is
    automatically isolated per run.
    """
    return repo.parent / ".graphify"


# ── retrieval (the function ToolExecutor calls) ──────────────────────────────

def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Query the graphify knowledge graph built over `repo`. Returns up to
    `k` ranked repo-relative file paths (forward-slash form).

    Two implementation paths, tried in order:
      1. Python package `graphifyy` (preferred — cleaner integration).
      2. CLI binary at GRAPHIFY_BIN (subprocess fallback).

    Either path:
      - builds the graph on first call (cached in <repo>.parent/.graphify/)
      - searches the graph for entities matching `query`
      - returns repo-relative file paths in graph-ranked order.

    Raises RuntimeError with install instructions if neither path is available.
    """
    repo = Path(repo).resolve()
    cache = _cache_dir(repo)
    cache.mkdir(parents=True, exist_ok=True)

    mod = _try_import()
    if mod is not None:
        return _retrieve_via_module(mod, query, repo, cache, k)

    cli = os.environ.get("GRAPHIFY_BIN") or shutil.which("graphify")
    if cli:
        return _retrieve_via_cli(cli, query, repo, cache, k)

    raise RuntimeError(
        "graphify backend unavailable. Either:\n"
        "  pip install graphifyy\n"
        "  OR set GRAPHIFY_BIN to the graphify CLI binary path.\n"
        "See https://github.com/safishamsi/graphify"
    )


# ── path A: in-process via the graphifyy python module ───────────────────────

def _retrieve_via_module(mod, query: str, repo: Path, cache: Path,
                         k: int) -> List[str]:
    """Use the `graphifyy` python module to build/query the graph.

    The exact symbol names in graphifyy have shifted historically; this
    function probes for the common ones (`build_graph`, `KnowledgeGraph`,
    `Indexer`) and degrades gracefully so a minor version bump doesn't
    silently break the arm. If none of the probes match, raises with the
    repr() of the module's public surface so the user can fix the wrapper.
    """
    graph = None
    # Common builder names across graphifyy versions
    for fn_name in ("build_graph", "build_index", "index_repository", "Indexer"):
        fn = getattr(mod, fn_name, None)
        if fn is None:
            continue
        try:
            # Class form (Indexer(...).build()) vs function form
            if isinstance(fn, type):
                obj = fn(repo_path=str(repo), cache_dir=str(cache))
                if hasattr(obj, "build"):
                    obj.build()
                graph = obj
            else:
                graph = fn(repo_path=str(repo), cache_dir=str(cache))
            break
        except TypeError:
            # signature mismatch — try the next candidate
            continue

    if graph is None:
        public = [s for s in dir(mod) if not s.startswith("_")]
        raise RuntimeError(
            "graphifyy module loaded but no known builder found. "
            f"Public surface: {public}. Patch eval/backends/graphify.py "
            "_retrieve_via_module to match this version's API."
        )

    # Search the graph
    results = None
    for q_name in ("search", "query", "retrieve", "find"):
        q = getattr(graph, q_name, None)
        if q is None:
            q = getattr(mod, q_name, None)
        if q is None:
            continue
        try:
            results = q(query, k=k * 5) if q_name != "find" else q(query)
        except TypeError:
            try:
                results = q(query)
            except TypeError:
                continue
        if results is not None:
            break

    if results is None:
        raise RuntimeError(
            "graphifyy: graph built but no known search function. "
            "Patch eval/backends/graphify.py _retrieve_via_module."
        )

    files = _extract_files_from_obj(results, repo)
    return files[:k]


# ── path B: CLI subprocess (mirrors cbmem.py's pattern) ──────────────────────

def _run_cli(cli: str, args: List[str], timeout: int) -> str:
    try:
        r = subprocess.run([cli, *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _retrieve_via_cli(cli: str, query: str, repo: Path, cache: Path,
                      k: int) -> List[str]:
    """Subprocess CLI fallback.

    Expected CLI shape (override via env vars if it differs):
      graphify index <repo> --cache <cache_dir>
      graphify search '<query>' --repo <repo> --cache <cache_dir> --limit N

    The output is expected to be JSON on stdout with file paths under common
    keys (file, path, file_path, ...). We walk the JSON like cbmem does.
    """
    # Build (idempotent — second call is cheap if cache is populated)
    _run_cli(cli, ["index", str(repo), "--cache", str(cache)],
             timeout=_BUILD_TIMEOUT)

    blob = _run_cli(cli, ["search", query, "--repo", str(repo),
                          "--cache", str(cache), "--limit", str(max(k * 5, 50))],
                    timeout=_QUERY_TIMEOUT)

    files = _extract_files_from_blob(blob, repo)
    return files[:k]


# ── result-extraction (shared by both paths) ─────────────────────────────────

def _extract_files_from_blob(blob: str, repo: Path) -> List[str]:
    """Pull repo-relative file paths from a JSON blob (CLI path)."""
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return []
    return _extract_files_from_obj(data, repo)


def _extract_files_from_obj(obj, repo: Path) -> List[str]:
    """Walk an arbitrary JSON-like object for path strings inside the repo."""
    out: List[str] = []
    seen: Set[str] = set()
    repo_resolved = repo.resolve()

    def _consider(raw):
        if not isinstance(raw, str) or not raw:
            return
        p = Path(raw)
        cand = p if p.is_absolute() else (repo / raw)
        try:
            rel = cand.resolve().relative_to(repo_resolved)
        except (ValueError, OSError):
            return
        if not cand.is_file():
            return
        s = str(rel).replace("\\", "/")
        if s not in seen:
            seen.add(s)
            out.append(s)

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if (k.lower() in _PATH_KEYS) and isinstance(v, str):
                    _consider(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, str):
            # bare string — treat as a path candidate
            _consider(node)
        # objects with attributes (graphifyy may return dataclasses)
        else:
            for attr in ("file", "path", "file_path", "filepath", "location"):
                v = getattr(node, attr, None)
                if isinstance(v, str):
                    _consider(v)

    _walk(obj)
    return out


# ── selftest CLI — `python -m eval.backends.graphify --selftest <repo>` ──────

def _selftest(repo: str) -> None:
    """Verify graphify is installed and the wrapper works end-to-end."""
    repo_p = Path(repo).resolve()
    print(f"repo:  {repo_p}")

    mod = _try_import()
    cli = os.environ.get("GRAPHIFY_BIN") or shutil.which("graphify")
    print(f"module graphifyy: {'YES' if mod else 'no'}")
    print(f"cli graphify:     {cli or 'no'}")
    if not mod and not cli:
        print("\nNeither path available. Install one of:")
        print("  pip install graphifyy")
        print("  OR set GRAPHIFY_BIN=/path/to/graphify")
        raise SystemExit(1)

    cache = _cache_dir(repo_p)
    print(f"cache dir: {cache}  (sibling of repo — outside git workspace)")
    cache.mkdir(parents=True, exist_ok=True)

    print("\n--- retrieve('error handler middleware', k=10) ---")
    try:
        files = retrieve("error handler middleware", repo_p, k=10)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    for f in files:
        print(" ", f)
    print(f"\n{len(files)} results. selftest OK.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.graphify --selftest <repo_path>")
