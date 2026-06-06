"""Graphify backend — the "70× token reduction" knowledge-graph competitor.

graphify (safishamsi/graphify, pip pkg `graphifyy`) extracts a repo into a graph
(tree-sitter + NetworkX + Leiden) and answers queries with compact subgraphs
instead of raw files. Native CLI:

    graphify extract .            # builds ./graphify-out/graph.json
    graphify query "<nl query>"   # returns a subgraph (JSON)
    graphify explain "Symbol"     # node details / code
    graphify path "A" "B"         # shortest path between symbols

We wrap it as a native-tool arm (like cbmem): the agent gets graphify_search +
graphify_explain, returning the compact subgraph — so its token-reduction claim
is tested END-TO-END, not just at the retrieval stage.

Install (its OWN env — keep off the sentence-transformers stack):
    pipx install graphifyy            # or: uv tool install graphifyy
    # then set GRAPHIFY_BIN if `graphify` isn't on PATH
Selftest:
    python -m eval.backends.graphify --selftest <repo>

Parsing is defensive (graph JSON field names vary by version) — run the selftest
to confirm shapes before a full run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

_PATH_KEYS = {"file", "path", "file_path", "filepath", "location", "rel_path",
              "relative_path", "uri"}
_EXTRACTED: set = set()


def _bin() -> str:
    env = os.environ.get("GRAPHIFY_BIN")
    if env:
        if not Path(env).is_file():
            raise RuntimeError(f"GRAPHIFY_BIN points to a missing file: {env}")
        return env
    found = shutil.which("graphify")
    if found:
        return found
    raise RuntimeError("graphify not found. `pipx install graphifyy` (or "
                       "`uv tool install graphifyy`), or set GRAPHIFY_BIN.")


def _extract_timeout() -> int:
    """How long `graphify extract .` may run for ONE repo. Env-configurable because
    big repos (astropy ≈ 80–100 LLM chunks; via NIM Llama-70B ≈ 20–40 min wall) easily
    exceed the old 600 s. Per-chunk LLM timeout is graphify's own GRAPHIFY_API_TIMEOUT."""
    raw = os.environ.get("GRAPHIFY_EXTRACT_TIMEOUT", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return 3600  # 1 h — covers astropy on NIM Llama-70B; override with env for bigger repos


def _run(args: List[str], cwd: Path, timeout: int = 300) -> str:
    try:
        r = subprocess.run([_bin(), *args], cwd=str(cwd),
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not os.environ.get("SG_EVAL_QUIET"):
            sys.stderr.write(f"[graphify] exit {r.returncode} for `{args[0] if args else '?'}`: "
                             f"{(r.stderr or '').strip()[:200]}\n")
        return r.stdout or ""
    except (subprocess.SubprocessError, OSError) as e:
        if not os.environ.get("SG_EVAL_QUIET"):
            sys.stderr.write(f"[graphify] subprocess failed ({type(e).__name__}: {e})\n")
        return ""


def _extract_args() -> List[str]:
    """`extract .` plus an explicit `--backend` when we've pinned one via env.

    graphify auto-detects its extraction backend (gemini→kimi→claude→openai→…→ollama).
    To route at a self-hosted / NIM OpenAI-compatible endpoint we set OLLAMA_BASE_URL
    (graphify's `ollama` slot is a generic OpenAI-compat path). Passing `--backend
    ollama` EXPLICITLY here prevents a stray OPENAI_API_KEY/GEMINI_API_KEY in the env
    from hijacking extraction to a hosted provider (silent misroute / data leak).
    GRAPHIFY_BACKEND overrides if you want a different slot.
    """
    args = ["extract", "."]
    backend = os.environ.get("GRAPHIFY_BACKEND") or (
        "ollama" if os.environ.get("OLLAMA_BASE_URL") else "")
    if backend:
        args += ["--backend", backend]
    return args


def _ensure_extracted(repo: Path) -> None:
    """Build graphify-out/graph.json once per repo (the one-time extract cost)."""
    repo = Path(repo).resolve()
    gj = repo / "graphify-out" / "graph.json"
    if gj.exists() or str(repo) in _EXTRACTED:
        _EXTRACTED.add(str(repo))
        return
    _run(_extract_args(), cwd=repo, timeout=_extract_timeout())
    _EXTRACTED.add(str(repo))


def _walk_dicts(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_dicts(v)
    elif isinstance(node, list):
        for it in node:
            yield from _walk_dicts(it)


def _extract_symbols(blob: str, repo: Path) -> list:
    """(name, file) pairs from a graphify subgraph JSON, ranked/deduped."""
    try:
        data = json.loads(blob)
    except Exception:
        return []
    out, seen = [], set()
    for d in _walk_dicts(data):
        name = d.get("name") or d.get("symbol") or d.get("qualified_name") or d.get("id")
        fp = next((d[k] for k in _PATH_KEYS if isinstance(d.get(k), str)), "")
        if not name:
            continue
        fp = str(fp).replace("\\", "/")
        key = (str(name), fp)
        if key in seen:
            continue
        seen.add(key)
        out.append((str(name), fp))
    return out


def _extract_files(blob: str, repo: Path) -> List[str]:
    out, seen = [], set()
    for _, fp in _extract_symbols(blob, repo):
        if fp and fp not in seen and (repo / fp).is_file():
            seen.add(fp); out.append(fp)
    return out


def _graph_ready(repo: Path) -> bool:
    """True iff a non-empty prebuilt graph.json exists in this workspace."""
    gj = Path(repo).resolve() / "graphify-out" / "graph.json"
    try:
        return gj.exists() and gj.stat().st_size > 0
    except OSError:
        return False


_NOT_PREBUILT = ("No results. (graphify graph not prebuilt for this repo — run "
                 "`python -m eval.scripts.graphify_prebuild <dataset>` first.)")


def search_native(query: str, repo: Path, k: int = 10) -> str:
    """graphify query → ranked file::symbol (harness-parseable).

    AGENT-TIME path: NEVER extracts. If the graph isn't prebuilt (and copied into
    the workspace by isolation), return fast instead of triggering a ~1h LLM build
    that times out and wedges the whole task. Prebuild graphs separately with
    eval.scripts.graphify_prebuild."""
    repo = Path(repo).resolve()
    if not _graph_ready(repo):
        return _NOT_PREBUILT
    blob = _run(["query", query], cwd=repo, timeout=120)
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
        lines.append(f"{i+1}. {fp}::{short}" + (f"\n   symbol: {name}" if name != short else ""))
    return "Ranked results (file::symbol):\n" + "\n".join(lines)


def explain(symbol: str, repo: Path) -> str:
    """graphify explain → a symbol's node details / code / neighbors."""
    if not _graph_ready(repo):
        return _NOT_PREBUILT
    blob = _run(["explain", symbol], cwd=Path(repo).resolve(), timeout=60)
    try:
        return json.dumps(json.loads(blob), indent=1)[:3000]
    except Exception:
        return (blob or "").strip()[:3000] or "(no output)"


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """File-level retrieve (for the retrieval_eval path). Never extracts at query
    time — graphs must be prebuilt (eval.scripts.graphify_prebuild)."""
    repo = Path(repo).resolve()
    if not _graph_ready(repo):
        return []
    blob = _run(["query", query], cwd=repo, timeout=120)
    return _extract_files(blob, repo)[:k]


def _selftest(repo: str) -> None:
    repo_p = Path(repo).resolve()
    print(f"binary : {_bin()}")
    print(f"repo   : {repo_p}")
    print("\n--- extract (one-time graph build) ---")
    _ensure_extracted(repo_p)
    gj = repo_p / "graphify-out" / "graph.json"
    print(f"graph.json exists: {gj.exists()}  ({gj.stat().st_size if gj.exists() else 0} bytes)")
    print("\n--- NATIVE graphify_search('header fromstring bytes') ---")
    print(search_native("header fromstring bytes", repo_p, k=8)[:1500])
    print("\n--- NATIVE graphify_explain('fromstring') ---")
    print(explain("fromstring", repo_p)[:800])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.graphify --selftest <repo_path>")
