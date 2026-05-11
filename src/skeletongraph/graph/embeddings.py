"""
Embedding-based semantic search using sentence-transformers (MiniLM).

Embeds function signatures + docstrings at build time for zero-cost
semantic retrieval at query time. Optional dependency — falls back to
inverted index if sentence-transformers is not installed.

Model: all-MiniLM-L6-v2 (22MB, 384 dimensions, ~14k tokens/sec)
Storage: embeddings.npz (numpy compressed, ~6KB per 100 functions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model name — small, fast, good enough for code retrieval
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model. Returns None if not installed."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
        return _model
    except ImportError:
        logger.debug("sentence-transformers not installed. Embeddings disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        return None


def is_available() -> bool:
    """Check if embedding support is importable.

    Optional dependencies can be installed but unusable (for example a
    transformers/Keras mismatch). Treat any import failure as "not available" so
    deterministic indexing never breaks because optional embeddings are broken.
    """
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception as e:
        logger.debug("sentence-transformers unavailable. Embeddings disabled: %s", e)
        return False


def build_embedding_text(
    fqn: str,
    signature: str,
    docstring: str = "",
    body_keywords: Optional[List[str]] = None,
) -> str:
    """Build the text to embed for a single function.

    Format: "{short_name} {signature} {docstring_first_line} {top_keywords}"
    Keeps it compact — MiniLM works best with short texts.
    """
    # Short name from FQN: "requests/models.py::PreparedRequest.prepare" → "PreparedRequest.prepare"
    short_name = fqn.split("::")[-1] if "::" in fqn else fqn

    # First line of docstring only
    doc_line = ""
    if docstring:
        doc_line = docstring.strip().split("\n")[0].strip()

    # Top 10 body keywords
    kw_text = ""
    if body_keywords:
        kw_text = " ".join(body_keywords[:10])

    parts = [short_name, signature]
    if doc_line:
        parts.append(doc_line)
    if kw_text:
        parts.append(kw_text)

    return " ".join(parts)


@dataclass
class EmbeddingStore:
    """In-memory store of function embeddings for semantic search.

    Built at `skeletongraph build` time. Loaded at server start.
    Provides cosine similarity search against user queries.
    """

    # Ordered list of FQNs (index corresponds to embedding row)
    fqns: List[str] = field(default_factory=list)

    # Embedding matrix: shape (N, 384), L2-normalized
    matrix: Optional[np.ndarray] = None

    # Pre-computed texts used for embedding (for incremental updates)
    texts: Dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.fqns)

    @property
    def is_empty(self) -> bool:
        return self.matrix is None or len(self.fqns) == 0

    def build(
        self,
        entries: List[Tuple[str, str, str, List[str]]],
        on_progress: Optional[callable] = None,
    ) -> None:
        """Build embeddings for all functions.

        Args:
            entries: List of (fqn, signature, docstring, body_keywords) tuples.
            on_progress: Optional callback(current, total) for progress reporting.
        """
        model = _get_model()
        if model is None:
            logger.info("Embedding model not available. Skipping embedding build.")
            return

        if not entries:
            return

        # Build texts
        self.fqns = []
        texts_to_embed = []
        self.texts = {}

        for fqn, signature, docstring, body_kw in entries:
            text = build_embedding_text(fqn, signature, docstring, body_kw)
            self.fqns.append(fqn)
            texts_to_embed.append(text)
            self.texts[fqn] = text

        # Batch embed — MiniLM handles batching internally
        logger.info(f"Embedding {len(texts_to_embed)} functions with {_MODEL_NAME}...")

        # Embed in chunks to support progress reporting
        chunk_size = 256
        all_embeddings = []
        for i in range(0, len(texts_to_embed), chunk_size):
            chunk = texts_to_embed[i:i + chunk_size]
            chunk_emb = model.encode(
                chunk,
                show_progress_bar=False,
                normalize_embeddings=True,  # L2-normalize for cosine sim via dot product
                convert_to_numpy=True,
            )
            all_embeddings.append(chunk_emb)
            if on_progress:
                on_progress(min(i + chunk_size, len(texts_to_embed)), len(texts_to_embed))

        self.matrix = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Embedded {self.matrix.shape[0]} functions. Matrix shape: {self.matrix.shape}")

    def update(
        self,
        changed_entries: List[Tuple[str, str, str, List[str]]],
        removed_fqns: List[str],
    ) -> None:
        """Incrementally update embeddings for changed/removed functions.

        More efficient than full rebuild for small changes.
        """
        model = _get_model()
        if model is None or self.is_empty:
            return

        # Remove deleted FQNs
        if removed_fqns:
            removed_set = set(removed_fqns)
            keep_mask = [fqn not in removed_set for fqn in self.fqns]
            self.fqns = [fqn for fqn, keep in zip(self.fqns, keep_mask) if keep]
            self.matrix = self.matrix[keep_mask]
            for fqn in removed_fqns:
                self.texts.pop(fqn, None)

        # Add/update changed entries
        if changed_entries:
            new_texts = []
            new_fqns = []
            for fqn, signature, docstring, body_kw in changed_entries:
                text = build_embedding_text(fqn, signature, docstring, body_kw)
                self.texts[fqn] = text

                # If already exists, remove old embedding
                if fqn in self.fqns:
                    idx = self.fqns.index(fqn)
                    self.fqns.pop(idx)
                    self.matrix = np.delete(self.matrix, idx, axis=0)

                new_texts.append(text)
                new_fqns.append(fqn)

            if new_texts:
                new_emb = model.encode(
                    new_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)

                self.fqns.extend(new_fqns)
                if self.matrix is not None and len(self.matrix) > 0:
                    self.matrix = np.vstack([self.matrix, new_emb])
                else:
                    self.matrix = new_emb

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for functions semantically similar to the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of (fqn, similarity_score) tuples, sorted by descending score.
            Returns empty list if embeddings are not available.
        """
        if self.is_empty:
            return []

        model = _get_model()
        if model is None:
            return []

        # Embed the query
        query_emb = model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Cosine similarity via dot product (both are L2-normalized)
        similarities = (self.matrix @ query_emb.T).flatten()

        # Get top-k indices
        k = min(top_k, len(self.fqns))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            (self.fqns[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.0  # Skip zero/negative similarities
        ]

    def get_similarity(self, fqn: str, query: str) -> float:
        """Get the similarity score between a specific function and a query.

        Used by confidence scoring to evaluate individual candidate relevance.
        """
        if self.is_empty or fqn not in self.fqns:
            return 0.0

        model = _get_model()
        if model is None:
            return 0.0

        idx = self.fqns.index(fqn)
        query_emb = model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        return float(self.matrix[idx] @ query_emb.T)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, sg_dir: Path) -> None:
        """Save embeddings to .skeletongraph/embeddings.npz."""
        if self.is_empty:
            return

        import json
        path = sg_dir / "embeddings.npz"
        tmp_path = path.with_suffix(".tmp")

        np.savez_compressed(
            tmp_path,
            matrix=self.matrix,
        )
        tmp_path.replace(path)

        # Save FQN ordering separately (JSON for readability/debuggability)
        meta_path = sg_dir / "embeddings_meta.json"
        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(
            json.dumps({"fqns": self.fqns, "model": _MODEL_NAME, "dim": _EMBEDDING_DIM}),
            encoding="utf-8",
        )
        tmp_meta.replace(meta_path)

    @classmethod
    def load(cls, sg_dir: Path) -> EmbeddingStore:
        """Load embeddings from .skeletongraph/embeddings.npz."""
        import json

        store = cls()
        emb_path = sg_dir / "embeddings.npz"
        meta_path = sg_dir / "embeddings_meta.json"

        if not emb_path.exists() or not meta_path.exists():
            return store

        try:
            data = np.load(emb_path)
            store.matrix = data["matrix"].astype(np.float32)

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            store.fqns = meta["fqns"]

            # Validate dimensions match
            if store.matrix.shape[0] != len(store.fqns):
                logger.warning(
                    f"Embedding matrix ({store.matrix.shape[0]}) doesn't match "
                    f"FQN count ({len(store.fqns)}). Discarding embeddings."
                )
                return cls()

            logger.info(f"Loaded {store.count} embeddings from {emb_path}")
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}")
            return cls()

        return store
