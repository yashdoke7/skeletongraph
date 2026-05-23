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
contract as bm25 / hybrid / aider.

Install (binary, separate from the Python env):
    curl -fsSL https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/scripts/setup.sh | bash
    # Windows: download + run the installer from the releases page.
Point the harness at it with CBMEM_BIN if it isn't on PATH.

⚠ ON-BOX VALIDATION NEEDED (cannot be tested without the binary):
  1. The index subcommand (we try `<bin> index <repo>`; confirm the real verb).
  2. The `cli search_graph` JSON output schema (we scan for file-path-like
     fields; confirm the actual key names and adjust _extract_files).
Run `python -m eval.backends.cbmem --selftest <repo>` once the binary is
installed to print the raw JSON and verify the parsing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Set

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# File-path-like keys we look for in whatever JSON the binary returns.
_PATH_KEYS = ("file", "path", "file_path", "filepath", "location", "rel_path",
              "relative_path", "uri")
_STOP = {
    "the", "this", "that", "with", "from", "into", "when", "then", "should",
    "fix", "add", "error", "issue", "bug", "code", "test", "tests", "return",
    "self", "args", "kwargs", "for", "and", "not", "but", "use", "using",
}


def _bin() -> str:
    """Locate the codebase-memory-mcp binary (env override wins)."""
    env = os.environ.get("CBMEM_BIN")
    if env:
        if not Path(env).is_file():
            raise RuntimeError(
                f"CBMEM_BIN points to a non-existent file: {env}\n"
                f"You currently have the SOURCE, not a binary. Either download "
                f"the Windows release .exe from "
                f"github.com/DeusData/codebase-memory-mcp/releases, or build it "
                f"with Go, then set CBMEM_BIN to the real .exe."
            )
        return env
    found = shutil.which("codebase-memory-mcp")
    if found:
        return found
    raise RuntimeError(
        "codebase-memory-mcp binary not found. Install it (see eval/backends/"
        "cbmem.py header) or set CBMEM_BIN to its full path."
    )


def _run(bin_path: str, args: List[str], repo: Path, timeout: int = 120) -> str:
    """Run a cbmem CLI command inside the repo; return stdout ('' on failure)."""
    try:
        r = subprocess.run([bin_path, *args], cwd=str(repo),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _ensure_indexed(bin_path: str, repo: Path) -> None:
    """Index the repo. The binary auto-indexes on first query in many setups;
    we also try an explicit index verb (best-effort, never fatal)."""
    # NEEDS-VALIDATION: confirm the real index verb (`index` vs `scan` vs none).
    # Capped at 120s so a wrong/hanging CLI fails fast instead of blocking the
    # whole run for 10 min per repo (cbmem claims ms-level indexing — a long
    # hang means the invocation is wrong; run `--selftest` to diagnose).
    index_cmd = os.environ.get("CBMEM_INDEX_CMD", "index")
    if index_cmd:
        _run(bin_path, [index_cmd, "."], repo, timeout=120)


def _query_terms(query: str, limit: int = 8) -> List[str]:
    terms = [t for t in _IDENT_RE.findall(query or "") if t.lower() not in _STOP]
    # preserve order, dedupe, cap
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _extract_files(blob: str, repo: Path) -> List[str]:
    """Pull repo-relative file paths (ordered, deduped) from cbmem JSON output.

    Robust to schema variation: walks the JSON and collects any string value
    under a path-like key that resolves to a file inside the repo.
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

    Maps the free-text query to a Codebase-Memory graph search over symbol
    names, then collects the files of the matched nodes (graph order preserved).
    """
    repo = Path(repo).resolve()
    bin_path = _bin()
    _ensure_indexed(bin_path, repo)

    terms = _query_terms(query)
    if not terms:
        return []
    # search_graph matches symbol names by regex; OR the query identifiers.
    name_pattern = "(?i)(" + "|".join(re.escape(t) for t in terms) + ")"
    payload = json.dumps({"name_pattern": name_pattern, "limit": max(k * 5, 50)})

    # NEEDS-VALIDATION: confirm `cli search_graph <json>` and its output schema.
    blob = _run(bin_path, ["cli", "search_graph", payload], repo)
    files = _extract_files(blob, repo)
    return files[:k]


def _selftest(repo: str) -> None:
    """`python -m eval.backends.cbmem --selftest <repo>` — print raw output so
    the index verb + JSON schema can be verified on a box that has the binary."""
    repo_p = Path(repo).resolve()
    bin_path = _bin()
    print("binary:", bin_path)
    _ensure_indexed(bin_path, repo_p)
    payload = json.dumps({"name_pattern": "(?i)(handler|parse|config)", "limit": 20})
    blob = _run(bin_path, ["cli", "search_graph", payload], repo_p)
    print("--- raw search_graph output (first 2000 chars) ---")
    print(blob[:2000])
    print("--- parsed files ---")
    for f in _extract_files(blob, repo_p):
        print(" ", f)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.cbmem --selftest <repo_path>")
