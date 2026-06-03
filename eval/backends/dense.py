"""Dense embedding retrieval — function-level, all-MiniLM-L6-v2.

Two roles:
  1. `retrieve()` — dense over CODE (fqn + signature + docstring + body). This is
     the control for the summary-search probe: it isolates "dense matching" from
     "what is being embedded". If summary-dense beats code-dense, the win is the
     SUMMARY, not the embedder.
  2. `rank()` — reusable: embed an arbitrary doc list, rank by cosine vs the query.
     `summary_search.py` calls this to rank over summaries.

Chunking reuses bm25_flat's tree-sitter enumeration so the ONLY variable vs the
bm25 baseline is dense-vs-lexical, and vs summary-dense is code-vs-summary.

Doc embeddings are content-hashed and cached to disk (.npy) so re-runs are cheap
and deterministic. The query is encoded fresh each call (microseconds).

Needs: pip install 'sentence-transformers>=3,<4'  (CPU is fine — MiniLM is 22M).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None        # lazily-loaded SentenceTransformer (load once per process)


def _ensure_paths() -> None:
    """Make `skeletongraph` and sibling `backends`/`eval` importable regardless
    of how the harness was launched (python -m eval.* or python eval/...)."""
    here = Path(__file__).resolve()
    eval_dir = str(here.parent.parent)          # .../eval
    repo_root = str(here.parents[2])            # repo root (has skeletongraph/ + eval/)
    for p in (eval_dir, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _encode(texts: List[str]):
    import numpy as np
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    embs = _model().encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,   # cosine == dot product
    )
    return embs.astype("float32")


def _doc_embeddings(fqns: List[str], docs: List[str], cache_dir: Path, tag: str):
    """Encode docs once, cache by (model, tag, content-hash). Returns (N, d) array."""
    import numpy as np
    fp = hashlib.sha1(
        (_MODEL_NAME + "|" + tag + "|" +
         "\n".join(f"{f}\t{d}" for f, d in zip(fqns, docs))).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"emb_{tag}_{fp}.npy"
    if npy.exists():
        try:
            return np.load(npy)
        except Exception:
            pass
    embs = _encode(docs)
    try:
        np.save(npy, embs)
    except Exception:
        pass
    return embs


def rank(query: str, fqns: List[str], docs: List[str], top_n: int,
         cache_dir: Path, tag: str) -> List[str]:
    """Rank `fqns` by cosine similarity of `docs` embeddings to `query`."""
    import numpy as np
    if not fqns:
        return []
    doc_emb = _doc_embeddings(fqns, docs, cache_dir, tag)
    if doc_emb.shape[0] == 0:
        return []
    q = _encode([query])
    if q.shape[0] == 0:
        return []
    sims = doc_emb @ q[0]                      # both normalized → cosine
    order = np.argsort(-sims)[:top_n]
    return [fqns[int(i)] for i in order]


def _code_docs(repo_path: Path) -> Tuple[List[str], List[str]]:
    """(fqns, code_texts) for every function — same enumeration as bm25_flat."""
    _ensure_paths()
    try:
        from eval.backends.bm25_flat import _functions_with_text
    except Exception:
        from backends.bm25_flat import _functions_with_text
    funcs = _functions_with_text(Path(repo_path))
    return [f for f, _ in funcs], [t for _, t in funcs]


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Dense over CODE — the control arm for the summary-search probe."""
    repo_path = Path(repo_path)
    fqns, docs = _code_docs(repo_path)
    cache = repo_path / ".skeletongraph" / "dense_cache"
    return rank(query, fqns, docs, top_n, cache, tag="code")
