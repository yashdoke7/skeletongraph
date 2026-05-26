"""Codebase-Memory baseline — the closest published competitor to SG.

Codebase-Memory (DeusData/codebase-memory-mcp, arXiv 2603.27277) is a
tree-sitter knowledge-graph code-intelligence server: it indexes a repo into a
persistent graph and answers structural queries (symbol search, call paths, hub
detection) with large token savings. It ships as a single static binary with a
CLI mode, so we wrap it via subprocess — no MCP protocol, no Python dependency,
and therefore no env conflict with the sentence-transformers stack.

We hold the agent's tool surface fixed (every arm exposes the same
search_code/read_file/edit_file/submit). For this arm, search_code dispatches
to Codebase-Memory's graph search and returns ranked file paths — the same
contract as bm25 / hybrid / sg.

CLI interface (confirmed against v0.6.1 Windows binary):
    codebase-memory-mcp cli index_repository '{"repo_path":"C:/forward/slash/path"}'
    codebase-memory-mcp cli search_graph '{"query":"...","project":"<slug>","limit":N}'
    codebase-memory-mcp cli list_projects '{}'

Project slug: repo path with ':' removed and all separators ('/' or '\\') replaced by '-'.
Example: C:/Users/foo/repos/django → C-Users-foo-repos-django

Install (binary, separate from the Python env):
    # Windows: download the .exe from github.com/DeusData/codebase-memory-mcp/releases
    # Set CBMEM_BIN to the full path of the .exe
    set CBMEM_BIN=C:\\path\\to\\codebase-memory-mcp.exe

    # Linux/Mac:
    curl -fsSL https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/scripts/setup.sh | bash

Run `python -m eval.backends.cbmem --selftest <repo>` to verify.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Set

# File-path-like keys we look for in whatever JSON the binary returns.
_PATH_KEYS = {"file", "path", "file_path", "filepath", "location", "rel_path",
              "relative_path", "uri"}


def _bin() -> str:
    """Locate the codebase-memory-mcp binary (env override wins)."""
    env = os.environ.get("CBMEM_BIN")
    if env:
        if not Path(env).is_file():
            raise RuntimeError(
                f"CBMEM_BIN points to a non-existent file: {env}\n"
                f"Download the Windows .exe from "
                f"github.com/DeusData/codebase-memory-mcp/releases "
                f"and set CBMEM_BIN to its full path."
            )
        return env
    found = shutil.which("codebase-memory-mcp")
    if found:
        return found
    raise RuntimeError(
        "codebase-memory-mcp binary not found. Install it or set CBMEM_BIN."
    )


def _project_slug(repo: Path) -> str:
    """Compute the cbmem project identifier for a repo path.

    cbmem derives the project slug by:
        - converting backslashes to forward slashes
        - removing the colon from Windows drive letters  (C: → C)
        - replacing all '/' with '-'
    Example:
        C:/Users/foo/repos/django  →  C-Users-foo-repos-django
    """
    s = str(repo).replace("\\", "/")   # normalise separators
    s = s.replace(":", "")              # remove Windows drive colon
    s = s.replace("/", "-")             # slashes → hyphens
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def _run(bin_path: str, args: List[str], timeout: int = 300) -> str:
    """Run a cbmem CLI command; return stdout ('' on failure).

    stderr (level=info log lines) is discarded — only the JSON on stdout matters.
    timeout is generous because index_repository on a large repo can take ~30s.
    """
    try:
        r = subprocess.run(
            [bin_path, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _is_indexed(bin_path: str, slug: str) -> bool:
    """Return True if the project is already in cbmem's registry."""
    blob = _run(bin_path, ["cli", "list_projects", "{}"], timeout=15)
    try:
        data = json.loads(blob)
        projects = data.get("projects", [])
        return any(p.get("name") == slug for p in projects)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def _ensure_indexed(bin_path: str, repo: Path) -> None:
    """Index the repo if not already present in cbmem's project registry.

    cbmem persists the index in an OS data directory, so this is a one-time
    cost per repo path. On subsequent calls (same run or re-run of the same
    task), the check is cheap (~15ms) and indexing is skipped.
    """
    slug = _project_slug(repo)
    if _is_indexed(bin_path, slug):
        return
    # Forward slashes required — the binary rejects backslash paths.
    repo_str = str(repo).replace("\\", "/")
    payload = json.dumps({"repo_path": repo_str})
    _run(bin_path, ["cli", "index_repository", payload], timeout=300)


def _extract_files(blob: str, repo: Path) -> List[str]:
    """Pull repo-relative file paths (ordered, deduped) from cbmem JSON output.

    cbmem search_graph returns:
        {"results": [{"file_path": "django/db/models/query.py", ...}, ...]}

    We walk the JSON and collect any string value under a path-like key that
    resolves to a file inside the repo. Results are already ranked by cbmem
    (BM25 by default); we preserve that order.
    """
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return []

    out: List[str] = []
    seen: Set[str] = set()

    def _consider(raw: str) -> None:
        if not isinstance(raw, str) or not raw:
            return
        p = Path(raw)
        cand = p if p.is_absolute() else (repo / raw)
        try:
            rel = cand.resolve().relative_to(repo.resolve())
        except (ValueError, OSError):
            return
        if not cand.is_file():
            return
        s = str(rel).replace("\\", "/")
        if s not in seen:
            seen.add(s)
            out.append(s)

    def _walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() in _PATH_KEYS and isinstance(v, str):
                    _consider(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return out


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked file paths (forward-slash, repo-relative).

    Indexes the repo on first call (subsequent calls are cheap — already indexed).
    Dispatches to cbmem's search_graph with the raw query string (cbmem uses BM25
    over symbol names and doc strings internally).
    """
    repo = Path(repo).resolve()
    bin_path = _bin()
    _ensure_indexed(bin_path, repo)

    slug = _project_slug(repo)
    payload = json.dumps({
        "query": query,
        "project": slug,
        "limit": max(k * 5, 50),
    })
    blob = _run(bin_path, ["cli", "search_graph", payload], timeout=60)
    files = _extract_files(blob, repo)
    return files[:k]


def _selftest(repo: str) -> None:
    """`python -m eval.backends.cbmem --selftest <repo>` — verify the binary.

    Prints the project slug, raw JSON output, and parsed file paths so you can
    confirm the binary is working and parsing is correct.
    """
    repo_p = Path(repo).resolve()
    bin_path = _bin()
    slug = _project_slug(repo_p)
    print(f"binary : {bin_path}")
    print(f"repo   : {repo_p}")
    print(f"slug   : {slug}")

    print("\n--- indexing (may take 10-30s for a large repo) ---")
    _ensure_indexed(bin_path, repo_p)
    print("indexed OK")

    print("\n--- list_projects ---")
    lp = _run(bin_path, ["cli", "list_projects", "{}"], timeout=15)
    print(lp[:500])

    print("\n--- search_graph (query='handler parse config') ---")
    payload = json.dumps({"query": "handler parse config", "project": slug, "limit": 20})
    blob = _run(bin_path, ["cli", "search_graph", payload], timeout=60)
    print("raw output (first 2000 chars):")
    print(blob[:2000])

    print("\n--- parsed file paths ---")
    for f in _extract_files(blob, repo_p):
        print(" ", f)

    print("\n--- retrieve('QuerySet as_manager', k=10) ---")
    files = retrieve("QuerySet as_manager method", repo_p, k=10)
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.cbmem --selftest <repo_path>")
