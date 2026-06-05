"""SG's NATIVE agent tools — what makes SG a pipeline, not a bare ranker.

In the systems comparison (final v2) the SG arm exposes these beyond the shared
edit/submit, so the agent uses SG the way the product actually works:

  sg_search   — structural ranked results (heuristic_query)   [already in tools.py]
  read_symbol — read ONE function/class body by `file::symbol` (token-cheap;
                no whole-file dump)
  expand      — graph navigation: the callers + callees of a symbol, with their
                signatures (SG's structural edge — bm25/grep can't do this)

read_symbol uses stdlib `ast` (SWE-bench is all-Python) so it needs no index.
expand uses SGEngine's call graph (built once per workspace, cached).
Wired into the SG arm's tool surface by profiles.py (Phase 2).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _ensure_sg_on_path() -> None:
    here = Path(__file__).resolve()
    for p in (str(here.parent.parent), str(here.parents[2])):
        if p not in sys.path:
            sys.path.insert(0, p)


# ── read_symbol ──────────────────────────────────────────────────────────────

def _symbol_span(src: str, sym: str):
    """(start, end, qualname) of the def/class matching `sym` (full qualname or
    last name token). Exact qualname wins; else innermost matching name."""
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    target = sym.split(".")[-1]
    best = None
    stack: list = []

    def walk(node):
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child.name)
                q = ".".join(stack)
                s = child.lineno
                e = getattr(child, "end_lineno", s)
                if q == sym:
                    best = (s, e, q)
                elif child.name == target and (best is None or best[2] != sym):
                    if best is None or (e - s) < (best[1] - best[0]):
                        best = (s, e, q)
                walk(child)
                stack.pop()
            else:
                walk(child)

    walk(tree)
    return best


def read_symbol(repo: Path, fqn: str) -> str:
    """Return just the body of `file::symbol`, line-numbered. Falls back with a
    clear message if the symbol can't be located (agent can use read_file)."""
    if not fqn or "::" not in fqn:
        return ("ERROR: read_symbol needs a 'file::symbol' id from a search "
                "result, e.g. pkg/mod.py::Class.method")
    repo = Path(repo)
    file, sym = fqn.split("::", 1)
    f = (repo / file).resolve()
    if not str(f).startswith(str(repo.resolve())):
        return "ERROR: path escapes repo"
    if not f.is_file():
        return f"ERROR: not a file: {file}"
    src = f.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    span = _symbol_span(src, sym)
    if not span:
        return (f"ERROR: symbol '{sym}' not found in {file}. Use read_file for "
                f"the whole file.")
    s, e, q = span
    e = min(e, len(lines))
    body = "\n".join(f"{i}: {lines[i-1]}" for i in range(s, e + 1))
    return f"{file}::{q} [{s}-{e} of {len(lines)}]\n{body}"


# ── expand (callers + callees) ───────────────────────────────────────────────

def expand(repo: Path, fqn: str, depth: int = 1) -> str:
    """SG graph navigation: 1-hop callers and callees of `fqn` with signatures.

    This is SG's structural contribution — given a function, show what calls it
    (blast radius) and what it calls (dependency chain), so the agent can follow
    the real control/data flow instead of guessing files.
    """
    _ensure_sg_on_path()
    try:
        from skeletongraph.engine import SGEngine
        from skeletongraph.config import SGConfig
    except Exception as ex:
        return f"ERROR: SG unavailable for expand ({type(ex).__name__})."
    repo = Path(repo)
    cfg = SGConfig()
    cfg.enable_embeddings = False
    cfg.enable_summaries = False
    try:
        store = SGEngine(project_root=repo, config=cfg).get_store()
    except Exception as ex:
        return f"ERROR: expand index build failed ({type(ex).__name__})."

    table = store.skeleton_table
    # resolve fqn (exact, else suffix/name match)
    target = fqn
    if target not in table:
        name = fqn.split("::")[-1]
        cands = [k for k in table if k.endswith(fqn) or k.split("::")[-1].endswith(name)]
        if not cands:
            return f"ERROR: '{fqn}' not in the call graph. Try read_symbol or read_file."
        target = sorted(cands, key=len)[0]

    g = store.graph

    def _fmt(fqns: dict, kind: str) -> list:
        out = []
        for f, dist in sorted(fqns.items(), key=lambda kv: kv[1]):
            if f == target:
                continue
            sk = table.get(f)
            sig = (getattr(sk, "signature", "") or "").strip() if sk else ""
            out.append(f"  [{kind} d{dist}] {f}" + (f"  {sig}" if sig else ""))
        return out[:20]

    try:
        callers = g.blast_radius(target, max_depth=depth)      # who calls target
        callees = g.dependency_chain(target, max_depth=depth)  # what target calls
    except Exception as ex:
        return f"ERROR: graph traversal failed ({type(ex).__name__})."

    lines = [f"Graph neighbors of {target} (depth {depth}):"]
    cl = _fmt(callers, "caller")
    ce = _fmt(callees, "callee")
    lines += (cl or ["  (no callers found)"])
    lines += (ce or ["  (no callees found)"])
    return "\n".join(lines)
