"""Flat BM25 baseline — standard RAG, no graph, no centrality.

Ranks every function in the repo by pure BM25 over its text
(fqn + signature + docstring + body). Uses SG's parser only to *enumerate*
functions and their line spans — the parser is shared infrastructure, not the
contribution; the contribution being compared is the retrieval strategy.

Self-contained BM25 (no rank_bm25 dependency) so the baseline is fully
inspectable for the paper.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class _BM25:
    """Okapi BM25."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = max(1, len(docs))
        self.avgdl = sum(len(d) for d in docs) / self.N
        df: Counter = Counter()
        for d in docs:
            for t in set(d):
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }
        self.tf = [Counter(d) for d in docs]

    def score(self, q: List[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * num / den
        return s


def _functions_with_text(repo_path: Path) -> List[Tuple[str, str]]:
    """Enumerate (fqn, indexable_text) for every function in the repo."""
    from skeletongraph.engine import SGEngine

    engine = SGEngine(project_root=repo_path)   # auto-builds index
    store = engine.get_store()

    by_file: Dict[str, list] = {}
    for fqn, sk in store.skeleton_table.items():
        by_file.setdefault(sk.file_path, []).append((fqn, sk))

    out: List[Tuple[str, str]] = []
    for rel, sks in by_file.items():
        fp = repo_path / rel
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
        for fqn, sk in sks:
            body = ""
            if lines:
                start = max(0, getattr(sk, "line_start", 1) - 1)
                end = min(len(lines), getattr(sk, "line_end", start + 1))
                body = "\n".join(lines[start:end])
            text = "\n".join([
                fqn,
                getattr(sk, "signature", "") or "",
                getattr(sk, "docstring", "") or "",
                body,
            ])
            out.append((fqn, text))
    return out


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    repo_path = Path(repo_path)
    funcs = _functions_with_text(repo_path)
    if not funcs:
        return []
    fqns = [f for f, _ in funcs]
    docs = [_tokenize(t) for _, t in funcs]
    bm = _BM25(docs)
    q = _tokenize(query)
    ranked = sorted(range(len(fqns)), key=lambda i: -bm.score(q, i))
    return [fqns[i] for i in ranked[:top_n]]
