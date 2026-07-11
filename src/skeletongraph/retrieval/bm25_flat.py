"""Flat BM25 over repo functions — the lexical leg of `fusion`.

Ranks every function in the repo by pure BM25 over its text
(fqn + signature + docstring + body). Uses SG's parser only to *enumerate*
functions and their line spans. Self-contained BM25 (no rank_bm25 dependency).

This is the product-side copy of the algorithm the eval benchmark uses
(`eval/backends/bm25_flat.py`) — kept byte-identical so MCP retrieval matches
the paper numbers. It only depends on `skeletongraph.*`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from ..graph.inverted_index import tokenize_identifier

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for word in _TOKEN_RE.findall(text or ""):
        split = tokenize_identifier(word)
        if split:
            tokens.extend(split)
    return tokens


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


# Per-process cache: (resolved repo path) -> (source-tree mtime fingerprint,
# [(fqn, text), ...]). The eval backend this was ported from pays this cost ONCE
# per fresh process (one query = one process), so rebuilding was free there. The
# MCP server is a long-lived session with many sg_search/sg_expand calls — WITHOUT
# this cache, every single call re-parsed the entire repo from scratch (measured:
# 1.3s x 14,873 functions, EVERY call, on top of a redundant second rebuild inside
# retrieve_fusion) — that redundant rebuild tax was the actual cause of the 3x
# wall-clock blowup on the Claude MCP arm, not the retrieval algorithm itself.
_CACHE: dict = {}


def _fingerprint(paths) -> float:
    """Max mtime across a known file list — pure filesystem stat, no engine
    construction, so a cache HIT costs microseconds instead of ~1.2s. A MISS
    (first call for this repo, or a genuinely new file appeared) falls through
    to the full rebuild below, which then seeds the cached file list."""
    latest = 0.0
    for fp in paths:
        try:
            latest = max(latest, fp.stat().st_mtime)
        except OSError:
            pass
    return latest


def known_files(repo_path: Path) -> List[Path]:
    """The file list backing the current cache entry for this repo (builds the
    cache first if needed). Lets other retrieval modules — e.g. fusion.py's
    engine cache — fingerprint the SAME file set without re-deriving it or
    paying their own separate tree-sitter walk."""
    key = str(Path(repo_path).resolve())
    _functions_with_text(repo_path)   # ensure cached
    return list(_CACHE[key][1])


def _functions_with_text(repo_path: Path) -> List[Tuple[str, str]]:
    """Enumerate (fqn, indexable_text) for every function in the repo.
    Cached per-process; invalidated when any previously-seen file's mtime
    advances (i.e., after the agent edits something) or the file SET changes,
    so mid-session edits are still reflected on the next call."""
    from ..engine import SGEngine
    from ..config import SGConfig

    key = str(repo_path.resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        fingerprint, files, out = cached
        if _fingerprint(files) == fingerprint:
            return out

    # BM25 only needs the skeleton table to enumerate functions; embeddings OFF
    # so this lexical leg doesn't pay the sentence-transformers load.
    cfg = SGConfig()
    cfg.enable_embeddings = False
    engine = SGEngine(project_root=repo_path, config=cfg)   # auto-builds index
    store = engine.get_store()

    by_file: Dict[str, list] = {}
    for fqn, sk in store.skeleton_table.items():
        by_file.setdefault(sk.file_path, []).append((fqn, sk))

    files = [repo_path / rel for rel in by_file]
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
    _CACHE[key] = (_fingerprint(files), files, out)
    return out


# Per-process cache of the BUILT index: (resolved repo) -> (fingerprint, fqns,
# _BM25). The corpus enumeration was already cached above, but retrieve() still
# re-tokenized all ~14k docs AND rebuilt the whole _BM25 (df/idf/tf counters) on
# EVERY call — ~3.5s each, measured — even though none of that depends on the
# query, only on repo content. That per-call rebuild was the dominant remaining
# warm-latency cost (bm25 ~3.5s vs heuristic_query ~0.08s vs enum ~0.05s). Caching
# the built index behind the SAME mtime fingerprint makes every warm call just
# tokenize the query + score (~0.1-0.2s). Results are byte-identical — this only
# removes redundant rebuilds — so it does not change any ranking or paper number.
_INDEX_CACHE: dict = {}


def _bm25_index(repo_path: Path):
    """(fqns, _BM25) for the repo, built once per content-fingerprint and reused.
    Invalidated exactly like _functions_with_text — a real edit advances the
    fingerprint and triggers a rebuild on the next call, so it self-heals."""
    key = str(repo_path.resolve())
    funcs = _functions_with_text(repo_path)   # (also refreshes _CACHE fingerprint)
    if not funcs:
        return [], None
    fp = _CACHE[key][0]                        # same fingerprint the enum cached under
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1], cached[2]
    fqns = [f for f, _ in funcs]
    bm = _BM25([_tokenize(t) for _, t in funcs])
    _INDEX_CACHE[key] = (fp, fqns, bm)
    return fqns, bm


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    repo_path = Path(repo_path)
    fqns, bm = _bm25_index(repo_path)
    if not fqns:
        return []
    q = _tokenize(query)
    ranked = sorted(range(len(fqns)), key=lambda i: -bm.score(q, i))
    return [fqns[i] for i in ranked[:top_n]]
