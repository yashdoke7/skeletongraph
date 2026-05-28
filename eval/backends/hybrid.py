"""Hybrid-RAG baseline: BM25 ∪ dense → cross-encoder rerank.

The deployed industry default (Augment, Voyage, Cohere pattern). Same
retrieval recipe that SG's curator chooses among; here we run it
monolithically without query classification — that's the controlled
experiment that shows SG's curator adds value.

Index is built lazily on first call and cached on disk under
<repo>/.hybrid_index/ for subsequent searches in the same workspace.

Stack
-----
BM25     — hand-rolled Okapi BM25 (same as bm25_flat.py, no extra dep)
Dense    — sentence-transformers all-MiniLM-L6-v2  (same model as SG uses)
Reranker — cross-encoder/ms-marco-MiniLM-L-6-v2

Indexing unit: symbol/function chunk when SG's parser is available, raw file as
a fallback. This is the standard "BM25 + dense + reranker over chunks" baseline;
returning FQNs keeps it comparable to SG and BM25 without giving it graph edges.

Install
-------
pip install sentence-transformers numpy
pip install skeletongraph[eval-strong]
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── BM25 (inline, no dependency) ─────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class _BM25:
    """Okapi BM25 over an arbitrary document corpus."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
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
        self._docs = docs

    def score(self, query_tokens: List[str], i: int) -> float:
        tf, dl = self.tf[i], len(self._docs[i])
        s = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * num / den
        return s

    def top_k(self, query_tokens: List[str], k: int) -> List[int]:
        scores = [(self.score(query_tokens, i), i) for i in range(len(self._docs))]
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


# ── model singletons ────────────────────────────────────────────────────────
# The embedder and cross-encoder are process-wide: loading them is expensive
# and they carry no per-repo state.  Without this, every task (= every repo)
# re-instantiates both models — the repeated "Loading weights" bars — which is
# pure waste in a 30-task eval loop.

_EMBED_MODEL = None       # SentenceTransformer (all-MiniLM-L6-v2)
_CROSS_ENCODER = None     # CrossEncoder (ms-marco-MiniLM-L-6-v2)


def _get_embed_model():
    """Load the dense embedder once and reuse it across all repos.

    Reads SG_EMBED_MODEL — the SAME env var SkeletonGraph uses — so the dense
    baseline and SG always share an embedder (controlled comparison). Defaults
    to MiniLM. trust_remote_code allows code-specific embedders to load.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        import os
        from sentence_transformers import SentenceTransformer
        name = os.environ.get("SG_EMBED_MODEL", "all-MiniLM-L6-v2")
        _EMBED_MODEL = SentenceTransformer(name, trust_remote_code=True)
    return _EMBED_MODEL


def _get_cross_encoder():
    """Load the cross-encoder reranker once and reuse it across all calls.

    Reads SG_RERANK_MODEL (defaults to the standard MS-MARCO MiniLM reranker).
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        import os
        from sentence_transformers import CrossEncoder
        name = os.environ.get("SG_RERANK_MODEL",
                              "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CROSS_ENCODER = CrossEncoder(name)
    return _CROSS_ENCODER


# ── per-repo index ─────────────────────────────────────────────────────────────

class _HybridIndex:
    """Lazy per-repo hybrid index (BM25 + dense embeddings).

    CACHE LOCATION: the index is written to `<repo>.parent / .hybrid_index/`,
    i.e. a SIBLING of the repo directory, NOT inside the repo itself.
    Writing inside the repo (the original design) caused `git diff HEAD` to
    capture the binary embeddings.npz and corrupt every SWE-bench
    `model_patch` with a "Binary files differ" hunk that the harness rejects
    (29/30 hybrid runs erred on NIM v2 for exactly this reason). The
    eval/agent/isolation.py workspace `.gitignore` also lists this path as a
    second line of defence — but the proper fix is to never write inside the
    repo to begin with.
    """

    INDEX_DIR = ".hybrid_index"
    FILES_JSON = "doc_ids.json"
    EMBED_NPZ = "embeddings.npz"

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        # SIBLING of the repo — outside the git workspace so it never enters
        # the agent's patch. Cleaned up automatically when the workspace
        # parent directory is rm'd by isolation.cleanup_workspace().
        self._idx_dir = repo.parent / self.INDEX_DIR
        self._file_paths: List[str] = []       # FQNs or relative file paths
        self._file_texts: List[str] = []       # chunk text
        self._bm25: Optional[_BM25] = None
        self._matrix = None                    # np.ndarray (n_files, embed_dim)
        self._embed_model = None               # SentenceTransformer
        self._loaded = False

    # ── load / build ──────────────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._idx_dir.exists() and self._is_fresh():
            self._load_from_disk()
        else:
            self._build_and_save()
        self._loaded = True

    def _is_fresh(self) -> bool:
        """Index is valid if the two required files exist."""
        return (
            (self._idx_dir / self.FILES_JSON).exists()
            and (self._idx_dir / self.EMBED_NPZ).exists()
        )

    def _build_and_save(self) -> None:
        """Build the index from scratch and persist to disk."""
        import numpy as np

        self._file_paths, self._file_texts = _collect_file_texts(self.repo)
        if not self._file_paths:
            return

        # BM25
        self._bm25 = _BM25([_tokenize(t) for t in self._file_texts])

        # Dense embeddings (shared model singleton)
        model = _get_embed_model()
        embeddings = model.encode(
            self._file_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._matrix = embeddings
        self._embed_model = model

        # Save — temp file MUST end in .npz (numpy appends .npz silently otherwise)
        self._idx_dir.mkdir(parents=True, exist_ok=True)
        emb_path = self._idx_dir / self.EMBED_NPZ
        tmp_path = self._idx_dir / "embeddings.tmp.npz"
        np.savez_compressed(str(tmp_path), matrix=embeddings)
        tmp_path.replace(emb_path)   # atomic rename

        (self._idx_dir / self.FILES_JSON).write_text(
            json.dumps(self._file_paths, indent=2), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        """Restore a previously built index from disk."""
        import numpy as np

        self._file_paths = json.loads(
            (self._idx_dir / self.FILES_JSON).read_text(encoding="utf-8")
        )
        data = np.load(str(self._idx_dir / self.EMBED_NPZ))
        self._matrix = data["matrix"]

        # Re-collect texts for BM25 (not persisted — cheap to recompute)
        _, self._file_texts = _collect_file_texts(self.repo)
        self._bm25 = _BM25([_tokenize(t) for t in self._file_texts])
        self._embed_model = _get_embed_model()

    # ── search ────────────────────────────────────────────────────────────────

    def bm25_search(self, query: str, top_k: int) -> List[str]:
        if not self._bm25 or not self._file_paths:
            return []
        idxs = self._bm25.top_k(_tokenize(query), top_k)
        return [self._file_paths[i] for i in idxs]

    def dense_search(self, query: str, top_k: int) -> List[str]:
        if self._matrix is None or not self._file_paths:
            return []
        import numpy as np
        q_emb = self._embed_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = self._matrix @ q_emb[0]           # shape (n_files,)
        idxs = int(scores.shape[0])
        ranked = sorted(range(idxs), key=lambda i: -float(scores[i]))
        return [self._file_paths[i] for i in ranked[:top_k]]

    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """Cross-encoder rerank. Falls back to dense order if CE unavailable."""
        if not candidates:
            return []
        try:
            ce = _get_cross_encoder()       # shared singleton, loaded once
        except Exception:
            # Cross-encoder unavailable → keep dense/BM25 union order
            return candidates[:top_k]

        # Build query-passage pairs
        path_to_text: Dict[str, str] = dict(zip(self._file_paths, self._file_texts))
        pairs = [(query, path_to_text.get(p, p)) for p in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [p for p, _ in ranked[:top_k]]


# ── module-level index cache (one index per repo path) ─────────────────────────

_INDEX_CACHE: Dict[str, _HybridIndex] = {}


def _get_index(repo: Path) -> _HybridIndex:
    key = str(repo.resolve())
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _HybridIndex(repo.resolve())
    idx = _INDEX_CACHE[key]
    idx.ensure_loaded()
    return idx


# ── public API ─────────────────────────────────────────────────────────────────


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked FQNs/chunks, falling back to file paths.

    Pipeline: BM25 top-30 ∪ dense top-30 → cross-encoder rerank → top k.
    """
    idx = _get_index(Path(repo))
    bm25_hits = idx.bm25_search(query, top_k=30)
    dense_hits = idx.dense_search(query, top_k=30)
    # Union in insertion order (preserves approximate quality ordering)
    seen: Set[str] = set()
    candidates: List[str] = []
    for h in bm25_hits + dense_hits:
        if h not in seen:
            seen.add(h)
            candidates.append(h)
    return idx.rerank(query, candidates, top_k=k)


# ── file text collection ───────────────────────────────────────────────────────

_SOURCE_EXTS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rs", ".rb",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".cs", ".kt", ".php",
}

_NOISE_DIRS: Set[str] = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "build", "dist", ".skeletongraph", ".hybrid_index",
}


def _collect_file_texts(repo: Path) -> Tuple[List[str], List[str]]:
    """Walk repo and build (file_paths, file_texts) lists.

    DEFAULT (production-RAG parity): file-level documents collected by walking
    the repo with rglob.  This is the standard "BM25 + dense + cross-encoder
    rerank" baseline (Cohere/Voyage/Pinecone tutorial pattern) — it does NOT
    depend on SkeletonGraph's parser, so the comparison is strictly
    SG-with-graph vs an off-the-shelf RAG stack.

    OPT-IN (legacy / appendix): set SG_HYBRID_USE_SG_CHUNKS=1 to use SG's
    function-level chunks instead.  This produces a stronger hybrid baseline
    (because it gets SG's tree-sitter parsing for free), useful for an
    ablation that asks "how much of SG's edge survives if hybrid sees the
    same chunks?" but NOT a fair vs-industry baseline.  See RESEARCH.md.
    """
    import os
    if os.environ.get("SG_HYBRID_USE_SG_CHUNKS") == "1":
        try:
            return _collect_via_sg(repo)
        except Exception:
            return _collect_raw(repo)
    return _collect_raw(repo)


def _collect_via_sg(repo: Path) -> Tuple[List[str], List[str]]:
    """Collect per-symbol chunk text via SG's existing index (preferred path)."""
    from skeletongraph.engine import SGEngine

    engine = SGEngine(project_root=repo)
    store = engine.get_store()

    paths: List[str] = []
    texts: List[str] = []
    for fqn, sk in sorted(store.skeleton_table.items()):
        fp = repo / sk.file_path
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        parts = [
            fqn,
            getattr(sk, "signature", "") or "",
            getattr(sk, "docstring", "") or "",
            getattr(sk, "file_path", "") or "",
        ]
        if lines:
            s = max(0, getattr(sk, "line_start", 1) - 1)
            e = min(len(lines), getattr(sk, "line_end", s + 30))
            parts.append("\n".join(lines[s:e]))
        paths.append(fqn)
        texts.append("\n".join(parts))
    return paths, texts


def _collect_raw(repo: Path) -> Tuple[List[str], List[str]]:
    """Fallback: read every source file as plain text (no SG index needed)."""
    paths: List[str] = []
    texts: List[str] = []
    for p in sorted(repo.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SOURCE_EXTS:
            continue
        rel_parts = p.relative_to(repo).parts
        if any(part in _NOISE_DIRS or part.startswith(".") for part in rel_parts):
            continue
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        paths.append(rel)
        texts.append(text[:20_000])  # cap per-file to avoid giant files
    return paths, texts
