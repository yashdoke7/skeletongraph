"""Aider repo-map baseline.

Uses the published aider-chat package's RepoMap class (PageRank over a
tree-sitter symbol graph) to rank files by relevance to a query.

We give it the same (query + repo) that every other arm gets and extract the
ranked file list it would surface. This is the canonical repo-map baseline.
Document the pinned aider-chat version in eval/REPRODUCIBILITY.md.

Install:
    pip install aider-chat

API notes (tested against aider-chat 0.82.x):
    RepoMap(map_tokens, root, main_model, io, verbose)
    rm.get_repo_map(chat_files, other_files, mentioned_fnames, mentioned_idents)
      → str  (contains file path lines + truncated symbol snippets)

The `main_model` field was renamed in some versions — we accept any model-like
object that exposes .token_count or fall back to a stub.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

# ── supported source extensions (mirrors aider's defaults) ────────────────────
_SOURCE_EXTS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rs", ".rb",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".cs", ".kt", ".php", ".swift",
}

# ── regex to pull file headers from repo-map text ─────────────────────────────
# Aider's format: path/to/file.py:\n  def func...\n
# The header line is the path followed by an optional colon and nothing else.
_FILE_HEADER_RE = re.compile(r"^([\w./ \\-]+\.[A-Za-z]+):?\s*$")

# Query token extractor — identifiers ≥3 chars
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked file paths (forward-slash, relative to repo root).

    Raises RuntimeError if aider-chat is not installed.
    Falls back to an empty list if the repo-map text is unparseable (e.g. the
    repo has no tree-sitter-parseable files).
    """
    try:
        from aider.repomap import RepoMap  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "aider-chat is not installed.  "
            "Run:  pip install aider-chat\n"
            "or:   pip install skeletongraph[eval-strong]"
        ) from exc

    repo = Path(repo).resolve()

    # Collect all source files aider would consider (rglob, filter by extension)
    all_files = [
        str(p) for p in repo.rglob("*")
        if p.is_file() and p.suffix.lower() in _SOURCE_EXTS
        # Skip hidden dirs and common noise dirs
        and not any(part.startswith(".") for part in p.relative_to(repo).parts)
        and not any(part in {"node_modules", "__pycache__", ".venv", "venv",
                             "build", "dist", ".git"}
                    for part in p.relative_to(repo).parts)
    ]

    if not all_files:
        return []

    rm = RepoMap(
        map_tokens=4096,     # token budget for the repo-map text
        root=str(repo),
        main_model=_stub_model(),
        io=_NullIO(),
        verbose=False,
    )

    mentioned_idents = {
        t for t in _IDENT_RE.findall(query) if len(t) >= 3
    }

    try:
        ranked_text = rm.get_repo_map(
            chat_files=[],
            other_files=all_files,
            mentioned_fnames=set(),
            mentioned_idents=mentioned_idents,
        )
    except Exception:
        ranked_text = ""

    return _extract_files_in_order(ranked_text or "", repo)[:k]


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_files_in_order(map_text: str, repo: Path) -> List[str]:
    """Parse file-path headers from Aider's repo-map text output.

    Aider emits the repo map as a sequence of file sections, each starting with:
        path/to/file.py:
            ⋮ symbol snippets

    We pull path lines in the order they appear (= Aider's PageRank order) and
    convert to repo-relative forward-slash paths.
    """
    files: List[str] = []
    seen: Set[str] = set()
    for line in map_text.splitlines():
        m = _FILE_HEADER_RE.match(line.strip())
        if not m:
            continue
        raw = m.group(1).strip()
        # Resolve to an absolute path and check it exists inside the repo
        candidate = (repo / raw).resolve()
        if not candidate.is_file():
            continue
        try:
            rel = str(candidate.relative_to(repo)).replace("\\", "/")
        except ValueError:
            continue
        if rel not in seen:
            seen.add(rel)
            files.append(rel)
    return files


def _stub_model():
    """Minimal model-like object for Aider's RepoMap constructor.

    Aider requires a model that exposes token_count (or similar). We don't call
    a model — we just need the RepoMap to rank files; the token budget controls
    how much output text it produces.
    """
    class _StubModel:
        """Stub that satisfies Aider's RepoMap model interface."""
        name = "stub"
        max_context_tokens = 128_000

        def token_count(self, text: str) -> int:
            return len(text) // 4  # rough 4-chars-per-token estimate

        # Some versions use these instead of token_count
        def count_tokens(self, text: str) -> int:
            return self.token_count(text)

        def info(self) -> dict:
            return {}

    return _StubModel()


class _NullIO:
    """Suppress all aider stdout/stderr chatter during retrieval."""

    def tool_output(self, *a, **k) -> None: pass
    def tool_warning(self, *a, **k) -> None: pass
    def tool_error(self, *a, **k) -> None: pass

    def read_text(self, fname: str) -> str:
        """Aider's RepoMap may call this to read file contents."""
        try:
            return Path(fname).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
