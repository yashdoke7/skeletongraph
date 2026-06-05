"""Aider RepoMap backend — tree-sitter + PageRank centrality.

Aider (https://aider.chat) is the most-used CLI coding agent that uses a
tree-sitter symbol graph + PageRank centrality for repository navigation.
Its `RepoMap` class is the closest published *prior art* to SkeletonGraph,
which makes it the most informative additional baseline: if SG beats
Aider's repo-map, the specific design choices (gated graph expansion +
three-tier summaries + constraint zone) add value over the canonical
PageRank-on-symbol-graph pattern.

Wrapped as a controlled `search_code` backend:
  1. Extract identifier-shaped tokens from the query → `mentioned_idents`
  2. Enumerate every source file under the repo (`other_fnames`)
  3. Call `RepoMap.get_ranked_tags(...)` — Aider's PageRank ranking
  4. Dedupe to file paths in rank order, return top-k

Cache discipline: Aider's default cache is `<repo>/.aider.tags.cache.v3/`,
which would leak into the agent's git diff and pollute the SWE-bench patch
(same failure mode that broke hybrid). We force the cache to a SIBLING of
the workspace (outside the git tree). The workspace `.gitignore` also
lists `.aider*` defensively.

Install:
  pip install aider-chat
Selftest:
  python -m eval.backends.aider_repomap --selftest <repo>
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Set

# ── identifier extraction (same shape as grep_sim's; consistent across arms) ──

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = {
    "the", "this", "that", "with", "from", "into", "when", "then", "should",
    "fix", "add", "error", "issue", "bug", "code", "test", "tests", "function",
    "method", "class", "return", "value", "true", "false", "none", "null",
    "self", "args", "kwargs", "for", "and", "not", "but", "use", "using",
    "what", "how", "why", "where", "which", "would", "could", "must",
}

_SOURCE_EXTS: Set[str] = {".py", ".js", ".ts", ".tsx", ".jsx", ".go",
                          ".java", ".rs", ".rb", ".cpp", ".cc", ".c",
                          ".h", ".cs", ".kt", ".php", ".swift"}
_NOISE_DIRS: Set[str] = {".git", ".venv", "venv", "__pycache__",
                         "node_modules", "build", "dist", ".aider_cache",
                         ".skeletongraph", ".hybrid_index"}


def _extract_idents(query: str) -> Set[str]:
    """Pull identifier-shaped tokens from the query for PageRank boosting."""
    return {
        w for w in _IDENT_RE.findall(query or "")
        if w.lower() not in _STOP and len(w) >= 3
    }


def _walk_source_files(repo: Path) -> List[str]:
    """Enumerate source files under `repo` (absolute paths)."""
    out: List[str] = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SOURCE_EXTS:
            continue
        parts = set(p.relative_to(repo).parts)
        if parts & _NOISE_DIRS:
            continue
        out.append(str(p))
    return out


# ── aider import (lazy — keeps the backend cheap when not used) ──────────────

def _try_import():
    """Return (RepoMap_class, InputOutput_class) or (None, None) if aider missing."""
    try:
        from aider.repomap import RepoMap
        try:
            from aider.io import InputOutput
        except ImportError:
            InputOutput = None
        return RepoMap, InputOutput
    except ImportError:
        return None, None


class _SilentIO:
    """Stub of aider.io.InputOutput that prints nothing.

    Aider's RepoMap calls io.tool_warning / io.tool_error / io.tool_output
    when it hits parse problems. We swallow those so they don't pollute
    the agent eval logs. read_text is needed for the tag scanner.
    """
    encoding = "utf-8"
    pretty = False
    def tool_warning(self, *a, **k): pass
    def tool_error(self, *a, **k): pass
    def tool_output(self, *a, **k): pass
    def read_text(self, fname):
        try:
            return Path(fname).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


class _StubModel:
    """Aider needs `main_model.token_count(text)` to budget the map. We
    just want ranked tags, so a cheap chars→tokens approximation suffices."""
    def token_count(self, text):
        return max(1, len(text or "") // 4)


# ── public retrieve ──────────────────────────────────────────────────────────

def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked repo-relative file paths via Aider RepoMap.

    Raises RuntimeError with a clear install hint if `aider-chat` is missing,
    so the eval doesn't silently report recall=0 across an entire stage.
    """
    RepoMap, _IO_class = _try_import()
    if RepoMap is None:
        raise RuntimeError(
            "aider_repomap backend unavailable. "
            "Install with:  pip install aider-chat\n"
            "Then re-run.  Selftest:  "
            "python -m eval.backends.aider_repomap --selftest <repo>"
        )

    repo = Path(repo).resolve()

    # Cache outside the repo (sibling of the workspace) — same discipline as
    # hybrid/graphify. Prevents .aider.tags.cache.v3 from leaking into the
    # agent's git diff and breaking the SWE-bench patch verifier.
    cache_root = repo.parent / ".aider_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    rm = RepoMap(
        map_tokens=4096,
        root=str(repo),
        main_model=_StubModel(),
        io=_SilentIO(),
        verbose=False,
        max_context_window=131072,
    )
    # Aider versions differ in how cache location is overridden. Try both
    # known knobs; if neither matches, .aider* in the workspace .gitignore
    # is the last line of defence.
    if hasattr(rm, "TAGS_CACHE_DIR"):
        rm.TAGS_CACHE_DIR = str(cache_root)
    if hasattr(rm, "cache_root"):
        rm.cache_root = str(cache_root)

    other_files = _walk_source_files(repo)
    mentioned_idents = _extract_idents(query)

    try:
        ranked_tags = rm.get_ranked_tags(
            chat_fnames=[],
            other_fnames=other_files,
            mentioned_fnames=set(),
            mentioned_idents=mentioned_idents,
        )
    except TypeError:
        # Older aider signatures accept positional args only / differ.
        try:
            ranked_tags = rm.get_ranked_tags(
                [], other_files, set(), mentioned_idents,
            )
        except Exception:
            ranked_tags = []
    except Exception:
        # Tree-sitter parse failures / sqlite corruption / etc. — fall back
        # to empty rather than crash the whole task.
        ranked_tags = []

    # Dedupe tags to file paths in rank order
    out: List[str] = []
    seen: Set[str] = set()
    for tag in ranked_tags:
        # Aider returns either a namedtuple (rel_fname, fname, line, name, kind)
        # or a plain tuple in older versions. Try attribute first, then index.
        fname = getattr(tag, "rel_fname", None)
        if fname is None and isinstance(tag, tuple) and tag:
            fname = tag[0]
        if not fname:
            continue
        fname_norm = str(fname).replace("\\", "/")
        if fname_norm in seen:
            continue
        seen.add(fname_norm)
        out.append(fname_norm)
        if len(out) >= k:
            break

    return out


# ── native aider: the repo-map INJECTED into context (systems comparison) ────

def get_map_text(repo: Path, map_tokens: int = 4096) -> str:
    """Aider's actual mechanism: a PageRank-ranked repo map (signatures from
    across the repo, token-budgeted) that gets INJECTED into the prompt — not a
    search tool. Returns the map text, or '' if aider is unavailable."""
    RepoMap, _ = _try_import()
    if RepoMap is None:
        return ""
    repo = Path(repo).resolve()
    cache_root = repo.parent / ".aider_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rm = RepoMap(map_tokens=map_tokens, root=str(repo), main_model=_StubModel(),
                 io=_SilentIO(), verbose=False, max_context_window=131072)
    if hasattr(rm, "TAGS_CACHE_DIR"):
        rm.TAGS_CACHE_DIR = str(cache_root)
    other = _walk_source_files(repo)
    for call in (lambda: rm.get_repo_map([], other),
                 lambda: rm.get_repo_map(chat_files=[], other_files=other)):
        try:
            m = call()
            if m:
                return str(m)
        except Exception:
            continue
    return ""


# ── selftest CLI ─────────────────────────────────────────────────────────────

def _selftest(repo: str) -> None:
    """Verify aider is importable and RepoMap returns file paths."""
    repo_p = Path(repo).resolve()
    print(f"repo: {repo_p}")
    RepoMap, _ = _try_import()
    if RepoMap is None:
        print("FAIL: aider not installed.  pip install aider-chat")
        raise SystemExit(1)
    print(f"aider RepoMap class: {RepoMap}")
    print()
    for q in ("separability_matrix", "ModelForm field validation",
              "UnicodeDecodeError"):
        print(f"--- retrieve({q!r}, k=10) ---")
        try:
            files = retrieve(q, repo_p, k=10)
        except Exception as e:
            print(f"  RAISED: {type(e).__name__}: {e}")
            continue
        print(f"  {len(files)} files")
        for f in files[:5]:
            print(f"    {f}")
        if len(files) > 5:
            print(f"    ... +{len(files) - 5} more")
        print()
    print("selftest OK" if files else "WARN: retrieve returned 0 paths")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2])
    else:
        print("usage: python -m eval.backends.aider_repomap --selftest <repo>")
