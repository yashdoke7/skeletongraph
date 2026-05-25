"""Ripgrep-style lexical baseline.

Models the well-known lexical-search family: exact phrase / identifier search
over raw source, ranked by line-level matches plus filename hits. It does not
use ASTs, embeddings, summaries, or graph edges. Returns file paths.
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
    phrase = " ".join((query or "").lower().split())
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
        rel = path.relative_to(repo_path).as_posix()
        score = 0.0

        # Filename/path match is one of grep's biggest strengths in code.
        rel_l = rel.lower()
        for kw in keywords:
            if kw in rel_l:
                score += 8.0

        if phrase and len(phrase) >= 8 and phrase in text:
            score += 12.0

        # Line-level scoring approximates ripgrep output better than global
        # file counts: many hits on one generated line shouldn't dominate.
        for line in text.splitlines():
            line_score = 0.0
            for kw in keywords:
                if kw in line:
                    line_score += 1.0
                    if re.search(rf"\b{re.escape(kw)}\b", line):
                        line_score += 0.5
            if line_score:
                score += min(line_score, 5.0)

        if score:
            scores[rel] = score

    return [f for f, _ in scores.most_common(top_n)]
