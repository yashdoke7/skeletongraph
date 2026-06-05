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

    On failure prints a one-line warning to stderr so a wedged binary / wrong
    args / missing index doesn't silently turn into recall=0 across an entire
    eval stage (the failure mode that caused cbmem to look broken on the
    llama33_70b v3 run).  Set SG_EVAL_QUIET=1 to suppress the warnings.
    """
    import os
    import sys
    try:
        r = subprocess.run(
            [bin_path, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0 and not os.environ.get("SG_EVAL_QUIET"):
            sys.stderr.write(
                f"[cbmem] non-zero exit {r.returncode} for `{args[0] if args else '?'}`: "
                f"{(r.stderr or '').strip()[:200]}\n"
            )
        return r.stdout or ""
    except (subprocess.SubprocessError, OSError) as e:
        if not os.environ.get("SG_EVAL_QUIET"):
            sys.stderr.write(
                f"[cbmem] subprocess failed ({type(e).__name__}: {e}) "
                f"args={args[:2]}\n"
            )
        return ""


# cbmem's project slug derivation differs from ours, so a computed slug causes
# "project not found". We capture cbmem's OWN registered name instead, cached
# per repo path.
_PROJECT_CACHE: dict = {}


def _name_from(resp: str) -> str | None:
    """Pull a project name out of cbmem JSON (index response / list_projects)."""
    try:
        for d in _walk_dicts(json.loads(resp)):
            for k in ("project", "name", "project_name", "slug"):
                v = d.get(k)
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return None


def _find_indexed(bin_path: str, repo: Path) -> str | None:
    """cbmem's registered project name for this repo, matched by root_path (exact
    and unambiguous — avoids the stale eval/datasets projects still in the
    registry) AND requiring it to be actually built (nodes>0). None otherwise."""
    target = str(repo).replace("\\", "/").rstrip("/")
    slug = _project_slug(repo)
    blob = _run(bin_path, ["cli", "list_projects", "{}"], timeout=15)
    try:
        projs = json.loads(blob).get("projects", [])
    except Exception:
        return None
    built = [p for p in projs if p.get("name") and (p.get("nodes") or 0) > 0]
    for p in built:               # 1) exact root_path match — the reliable key
        if str(p.get("root_path", "")).replace("\\", "/").rstrip("/") == target:
            return p["name"]
    for p in built:               # 2) fallback: exact computed-slug match
        if p["name"] == slug:
            return slug
    return None


def _ensure_indexed(bin_path: str, repo: Path) -> str:
    """Index the repo if needed; WAIT for the async build to finish; return
    cbmem's ACTUAL registered project name (use this everywhere)."""
    import time
    repo = Path(repo).resolve()
    key = str(repo)
    if key in _PROJECT_CACHE:
        return _PROJECT_CACHE[key]
    name = _find_indexed(bin_path, repo)
    if not name:
        _run(bin_path, ["cli", "index_repository",
             json.dumps({"repo_path": key.replace("\\", "/")})], timeout=300)
        # cbmem indexes ASYNCHRONOUSLY — querying before the graph is built 404s
        # ("project not found or not indexed"). Poll until it appears with nodes>0.
        for _ in range(120):               # up to ~240s
            name = _find_indexed(bin_path, repo)
            if name:
                break
            time.sleep(2)
        name = name or _project_slug(repo)
    _PROJECT_CACHE[key] = name
    return name


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
    slug = _ensure_indexed(bin_path, repo)
    payload = json.dumps({
        "query": query,
        "project": slug,
        "limit": max(k * 5, 50),
    })
    blob = _run(bin_path, ["cli", "search_graph", payload], timeout=60)
    files = _extract_files(blob, repo)
    return files[:k]


# ── NATIVE cbmem agent tools (systems comparison, final v2) ──────────────────
# These expose cbmem the way it's meant to be used: structured graph search,
# call-path tracing, function-snippet fetch, and an architecture overview — each
# returning a pruned subgraph instead of file dumps. Parsing is defensive (cbmem
# JSON field names vary by version); run `--selftest-native` to confirm shapes.

def _walk_dicts(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_dicts(v)
    elif isinstance(node, list):
        for it in node:
            yield from _walk_dicts(it)


def _extract_symbols(blob: str, repo: Path) -> list:
    """Pull (qualified_name|name, file) pairs from cbmem search JSON, ranked."""
    try:
        data = json.loads(blob)
    except Exception:
        return []
    out, seen = [], set()
    for d in _walk_dicts(data):
        name = d.get("qualified_name") or d.get("name") or d.get("symbol")
        fp = (d.get("file_path") or d.get("file") or d.get("path") or
              d.get("relative_path") or "")
        if not name:
            continue
        fp = str(fp).replace("\\", "/")
        key = (str(name), fp)
        if key in seen:
            continue
        seen.add(key)
        out.append((str(name), fp))
    return out


def search_native(query: str, repo: Path, k: int = 10) -> str:
    """cbmem search_graph → ranked `file::symbol` lines (same format the harness
    parses for recall) + a usable qualified-name for get_code_snippet."""
    repo = Path(repo).resolve()
    bin_path = _bin()
    slug = _ensure_indexed(bin_path, repo)
    payload = json.dumps({"query": query, "project": slug, "limit": max(k * 3, 30)})
    blob = _run(bin_path, ["cli", "search_graph", payload], timeout=60)
    syms = _extract_symbols(blob, repo)
    if not syms:
        files = _extract_files(blob, repo)
        if not files:
            return "No results."
        return "Ranked results (file::symbol):\n" + "\n".join(
            f"{i+1}. {f}" for i, f in enumerate(files[:k]))
    lines = []
    for i, (name, fp) in enumerate(syms[:k]):
        short = name.split(".")[-1]
        lines.append(f"{i+1}. {fp}::{short}\n   qualified_name: {name}")
    return "Ranked results (file::symbol):\n" + "\n".join(lines)


def _pretty(blob: str, cap: int) -> str:
    try:
        return json.dumps(json.loads(blob), indent=1)[:cap]
    except Exception:
        return (blob or "").strip()[:cap] or "(empty)"


def trace_calls(function_name: str, repo: Path, direction: str = "both") -> str:
    """cbmem trace_call_path → inbound/outbound call tree (depth ≤5)."""
    bin_path = _bin()
    slug = _ensure_indexed(bin_path, Path(repo).resolve())
    payload = json.dumps({"function_name": function_name, "direction": direction,
                          "project": slug})
    return _pretty(_run(bin_path, ["cli", "trace_call_path", payload], timeout=60), 1800)


def code_snippet(qualified_name: str, repo: Path) -> str:
    """cbmem get_code_snippet → the source for a function by qualified name."""
    bin_path = _bin()
    slug = _ensure_indexed(bin_path, Path(repo).resolve())
    payload = json.dumps({"qualified_name": qualified_name, "project": slug})
    blob = _run(bin_path, ["cli", "get_code_snippet", payload], timeout=60)
    try:
        data = json.loads(blob)
        for d in _walk_dicts(data):
            code = d.get("code") or d.get("snippet") or d.get("source")
            if code:
                return str(code)[:3000]
    except Exception:
        pass
    return _pretty(blob, 3000)


def architecture(repo: Path) -> str:
    """cbmem get_architecture → languages/packages/routes/hotspots overview."""
    bin_path = _bin()
    slug = _ensure_indexed(bin_path, Path(repo).resolve())
    payload = json.dumps({"project": slug})
    return _pretty(_run(bin_path, ["cli", "get_architecture", payload], timeout=60), 2500)


def _selftest(repo: str) -> None:
    """`python -m eval.backends.cbmem --selftest <repo>` — verify the binary.

    Prints the project slug, raw JSON output, and parsed file paths so you can
    confirm the binary is working and parsing is correct.
    """
    repo_p = Path(repo).resolve()
    bin_path = _bin()
    print(f"binary : {bin_path}")
    print(f"repo   : {repo_p}")
    print(f"computed slug : {_project_slug(repo_p)}")

    print("\n--- list_projects (BEFORE) ---")
    print(_run(bin_path, ["cli", "list_projects", "{}"], timeout=15)[:600])

    print("\n--- indexing + resolving cbmem's ACTUAL project name ---")
    slug = _ensure_indexed(bin_path, repo_p)
    print(f"resolved project name = {slug!r}   "
          f"(if this differs from the computed slug above, that mismatch was the "
          f"'project not found' bug — now fixed)")

    print("\n--- NATIVE cbmem_search ---")
    print(search_native("header fromstring bytes", repo_p, k=8)[:1500])

    print("\n--- NATIVE cbmem_arch ---")
    print(architecture(repo_p)[:800])

    print("\n--- NATIVE cbmem_snippet (paste a qualified_name from cbmem_search above) ---")
    print("  (skipped — re-run with a real qualified_name to test get_code_snippet)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.cbmem --selftest <repo_path>")
