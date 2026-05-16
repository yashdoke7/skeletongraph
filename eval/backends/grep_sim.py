"""Grep-simulation baseline — the naive agent.

Models what an agent with no retrieval tool does: pull identifiers out of the
issue text, grep the repo for them, rank files by match count. Returns file
paths (Stage 0 scores at file granularity, so bare paths match gold_files).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_SRC_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
            ".rs", ".cpp", ".c", ".h", ".cs", ".rb", ".php"}
_STOP = {
    "the", "this", "that", "with", "from", "into", "when", "then", "should",
    "fix", "add", "error", "issue", "bug", "code", "test", "tests", "function",
    "method", "class", "return", "value", "true", "false", "none", "null",
    "self", "args", "kwargs", "for", "and", "not", "but", "use", "using",
}


def _keywords(query: str, limit: int = 20) -> List[str]:
    counts = Counter(
        w.lower() for w in _IDENT_RE.findall(query or "")
        if w.lower() not in _STOP
    )
    return [w for w, _ in counts.most_common(limit)]


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    repo_path = Path(repo_path)
    keywords = _keywords(query)
    if not keywords:
        return []

    scores: Counter = Counter()
    for path in repo_path.rglob("*"):
        if path.suffix.lower() not in _SRC_EXT or not path.is_file():
            continue
        # skip vendored / hidden dirs
        parts = set(path.relative_to(repo_path).parts)
        if parts & {".git", "node_modules", "venv", ".venv", "dist", "build"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            continue
        hits = sum(text.count(kw) for kw in keywords)
        if hits:
            rel = path.relative_to(repo_path).as_posix()
            scores[rel] = hits

    return [f for f, _ in scores.most_common(top_n)]
