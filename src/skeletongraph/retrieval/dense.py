"""Dense embedding retrieval over repo functions — the semantic leg of `fusion`.

Dense over CODE (fqn + signature + docstring + body). Chunking reuses
`bm25_flat`'s enumeration so the only variable vs the lexical leg is
dense-vs-lexical. Doc embeddings are content-hashed and cached to disk (.npy).

Product-side copy of the eval algorithm (`eval/backends/dense.py`), kept
identical so MCP fusion matches the paper numbers. sentence-transformers is a
HARD dependency of skeletongraph (see pyproject.toml — deliberately not
optional, after a past regression where an optional embeddings extra silently
degraded results) — no separate install needed for this module.

Defaults to jina-embeddings-v2-base-code — the embedder `fusion` was validated
with (SWE-bench Pro: recall@1 +50%, MRR +26% over BM25 alone). The eval harness
this was ported from defaults to generic MiniLM instead, because there it's
also used as a deliberate CONTROL for a different ablation (dense-vs-lexical in
isolation) — that reason doesn't apply to the shipped product, which should
just default to the retriever that actually won. Override with SG_DENSE_MODEL.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import List, Tuple


def _model_name() -> str:
    # jina-embeddings-v2-base-code is the validated fusion embedder (see module
    # docstring) — the product default, not the eval's generic-MiniLM control.
    return os.environ.get("SG_DENSE_MODEL", "jinaai/jina-embeddings-v2-base-code")


_MODELS: dict = {}   # name -> lazily-loaded SentenceTransformer
_MODEL_LOCK = threading.Lock()   # guards _MODELS against a double-load race
                                 # between prewarm()'s thread and a real query


def _model():
    name = _model_name()
    # Double-checked locking: the fast path (already loaded) stays lock-free,
    # but two threads racing on a cold cache must not both build the model —
    # loading jina-v2-base twice costs ~25s AND doubles resident memory.
    if name in _MODELS:
        return _MODELS[name]
    with _MODEL_LOCK:
        if name not in _MODELS:
            from sentence_transformers import SentenceTransformer
            try:
                _MODELS[name] = SentenceTransformer(name, trust_remote_code=True)
            except TypeError:
                _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def prewarm(background: bool = True):
    """Start loading the embedding model NOW instead of on the first query.

    Measured: the first dense query costs ~24.7s (model load) vs ~0.4s warm.
    An MCP server that loads lazily pays that 24.7s inside the agent's first
    sg_search, where the user feels it. But an agent typically spends 5-15s
    reading the prompt and planning before it ever searches — so loading in a
    background thread at server startup hides most or all of the cost.

    Safe to call more than once (the lock + dict make it idempotent) and safe
    to call when dense is never used — it just wastes a background thread.
    Returns the Thread when background=True (mostly for tests), else None.
    """
    if _model_name() in _MODELS:
        return None
    if not background:
        _model()
        return None

    def _warm():
        try:
            _model()
        except Exception:
            # Never let a prewarm failure surface: the real query path will hit
            # the same error and report it in context, and a server must not die
            # in a daemon thread over an optional optimization.
            pass

    t = threading.Thread(target=_warm, name="sg-dense-prewarm", daemon=True)
    t.start()
    return t


# Truncate BEFORE the tokenizer to avoid attention-matrix OOM on huge functions.
_MAX_DOC_CHARS = int(os.environ.get("SG_DENSE_MAX_CHARS", "2048"))


def _encode(texts: List[str]):
    import numpy as np
    if not texts:
        m = _model()
        dim = m.get_sentence_embedding_dimension() or 384
        return np.zeros((0, dim), dtype="float32")
    texts = [t[:_MAX_DOC_CHARS] for t in texts]
    m = _model()
    orig_max = getattr(m, "max_seq_length", 8192)
    m.max_seq_length = min(orig_max, 512)
    try:
        embs = m.encode(
            texts, batch_size=int(os.environ.get("SG_DENSE_BATCH_SIZE", "8")),
            show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
    finally:
        m.max_seq_length = orig_max
    return embs.astype("float32")


def _doc_hash(doc: str) -> str:
    """Stable per-FUNCTION content hash (model + truncation + text). Changes iff
    that one function's indexable text changes — the unit of incremental reuse."""
    return hashlib.sha1(
        (_model_name() + f"|maxc={_MAX_DOC_CHARS}|" + doc[:_MAX_DOC_CHARS]).encode("utf-8")
    ).hexdigest()[:20]


# In-process mirror of the on-disk store, so repeated retrieve() calls in one
# server session don't re-read the .npz from disk. Keyed by the store path.
_STORE_CACHE: dict = {}   # store_path -> {hash: np.ndarray}


def _load_store(store_path: Path) -> dict:
    import numpy as np
    cached = _STORE_CACHE.get(str(store_path))
    if cached is not None:
        return cached
    store: dict = {}
    if store_path.exists():
        try:
            data = np.load(store_path, allow_pickle=False)
            keys, vecs = data["keys"], data["vecs"]
            store = {str(k): vecs[i] for i, k in enumerate(keys)}
        except Exception:
            store = {}
    _STORE_CACHE[str(store_path)] = store
    return store


def _save_store(store_path: Path, store: dict) -> None:
    import numpy as np
    import os
    if not store:
        return
    keys = np.array(list(store.keys()))
    vecs = np.stack(list(store.values())).astype("float32")
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp that ALREADY ends in .npz (np.savez auto-appends .npz to
        # any name lacking it — giving it one keeps the path we can os.replace).
        tmp = store_path.with_name(store_path.stem + ".tmp.npz")
        np.savez(tmp, keys=keys, vecs=vecs)
        os.replace(tmp, store_path)   # atomic — a crash mid-write can't corrupt the store
    except Exception:
        pass


def _doc_embeddings(fqns: List[str], docs: List[str], cache_dir: Path, tag: str):
    """Return (N, d) embeddings for docs, INCREMENTALLY.

    Each function's vector is cached by its own content hash in a single per-repo
    store (embcache_<tag>.npz). Only functions whose text changed (or new ones)
    are encoded; everything else is reused. So the first build pays the full
    corpus encode ONCE, but every later update — after the agent edits a file, or
    a `sg build` on a changed repo — only re-encodes the handful of changed
    functions (milliseconds), instead of the whole corpus (minutes). This is what
    makes fusion's dense leg cheap to keep warm and cheap for real users to
    maintain, mirroring how the main index already updates incrementally."""
    import numpy as np
    docs = [d[:_MAX_DOC_CHARS] for d in docs]
    store_path = cache_dir / f"embcache_{tag}.npz"
    store = _load_store(store_path)

    hashes = [_doc_hash(d) for d in docs]
    missing_idx = [i for i, h in enumerate(hashes) if h not in store]
    if missing_idx:
        new_vecs = _encode([docs[i] for i in missing_idx])
        for j, i in enumerate(missing_idx):
            store[hashes[i]] = new_vecs[j]
        # Prune to exactly the current corpus so vectors for deleted/changed
        # functions don't accumulate — `hashes` covers the whole function set on
        # a full retrieve, so this keeps the store size == current #functions.
        store = {h: store[h] for h in hashes}
        _STORE_CACHE[str(store_path)] = store
        _save_store(store_path, store)   # persist so the next process/session reuses it

    if not hashes:
        dim = _model().get_sentence_embedding_dimension() or 384
        return np.zeros((0, dim), dtype="float32")
    return np.stack([store[h] for h in hashes]).astype("float32")


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
    sims = doc_emb @ q[0]                      # both normalized -> cosine
    order = np.argsort(-sims)[:top_n]
    return [fqns[int(i)] for i in order]


def _code_docs(repo_path: Path) -> Tuple[List[str], List[str]]:
    """(fqns, code_texts) for every function — same enumeration as bm25_flat."""
    from .bm25_flat import _functions_with_text
    funcs = _functions_with_text(Path(repo_path))
    return [f for f, _ in funcs], [t for _, t in funcs]


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Dense over CODE."""
    repo_path = Path(repo_path)
    fqns, docs = _code_docs(repo_path)
    cache = repo_path / ".skeletongraph" / "dense_cache"
    return rank(query, fqns, docs, top_n, cache, tag="code")
