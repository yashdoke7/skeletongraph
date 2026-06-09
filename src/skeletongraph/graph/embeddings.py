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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Model name — configurable so the eval can run a controlled embedder
# comparison. The shipped default is the small/fast MiniLM (22 MB, 384-dim).
# Set SG_EMBED_MODEL to a code-specific embedder (e.g.
# "jinaai/jina-embeddings-v2-base-code") for a stronger, fairer dense baseline.
# SG and the hybrid eval backend read the SAME env var, so SG never gets an
# embedder advantage over the dense-RAG baseline. NOTE: changing the model
# changes the vector dimension, so existing indexes (.skeletongraph /
# .hybrid_index) must be rebuilt after switching.
_MODEL_NAME = os.environ.get("SG_EMBED_MODEL", "all-MiniLM-L6-v2")
_EMBEDDING_DIM = int(os.environ.get("SG_EMBED_DIM", "384"))

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model. Returns None if not installed."""
    global _model
    if _model is not None:
        return _model
    try:
        import sys
        import os
        
        # Remove current directory from sys.path to prevent 'requests' repo from poisoning Huggingface's import requests
        cwd = os.getcwd()
        original_sys_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in ('', cwd)]
        
        try:
            import transformers
            if not hasattr(transformers.PretrainedConfig, "is_decoder"):
                transformers.PretrainedConfig.is_decoder = False
            if not hasattr(transformers.PretrainedConfig, "add_cross_attention"):
                transformers.PretrainedConfig.add_cross_attention = False
            from sentence_transformers import SentenceTransformer
            # trust_remote_code lets code-specific embedders (jina, nomic, …) load;
            # harmless for MiniLM.
            _model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
            return _model
        finally:
            sys.path = original_sys_path
    except ImportError:
        logger.debug("sentence-transformers not installed. Embeddings disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        try:
            from rich.console import Console
            Console().print(f"[bold yellow]WARNING:[/bold yellow] Skipping embeddings! Failed to load model '{_MODEL_NAME}': {e}")
        except ImportError:
            print(f"WARNING: Skipping embeddings! Failed to load model '{_MODEL_NAME}': {e}")
        return None


def is_available() -> bool:
    """Check if embedding support is importable.

    Optional dependencies can be installed but unusable (for example a
    transformers/Keras mismatch). Treat any import failure as "not available" so
    deterministic indexing never breaks because optional embeddings are broken.
    """
    try:
        import sys
        import os
        cwd = os.getcwd()
        original_sys_path = sys.path[:]
        sys.path = [p for p in sys.path if p not in ('', cwd)]
        try:
            import sentence_transformers  # noqa: F401
            return True
        finally:
            sys.path = original_sys_path
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
                batch_size=2,
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

        # Dimension guard (see get_similarity): a prebuilt matrix loaded with a
        # mismatched query encoder would raise here. Degrade to "no dense results"
        # rather than crash the arm.
        if query_emb.shape[-1] != self.matrix.shape[-1]:
            logger.warning("Embedding dim mismatch in search (index=%d, query=%d) "
                           "— set SG_EMBED_MODEL to the index's model; returning no "
                           "dense hits.", self.matrix.shape[-1], query_emb.shape[-1])
            return []

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

    def rescore(self, query: str, fqns: List[str]) -> Dict[str, float]:
        """Compute semantic similarities for a specific subset of functions.
        
        Args:
            query: Natural language query string.
            fqns: List of FQNs to rescore.
            
        Returns:
            Dictionary mapping FQN to similarity score. FQNs without embeddings
            are omitted from the returned dictionary.
        """
        if self.is_empty or not fqns:
            return {}

        fqn_set = set(fqns)
        indices = [i for i, f in enumerate(self.fqns) if f in fqn_set]
        if not indices:
            return {}

        model = _get_model()
        if model is None:
            return {}

        query_emb = model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Dot product with the subset matrix
        sub_matrix = self.matrix[indices]
        similarities = (sub_matrix @ query_emb.T).flatten()

        return {self.fqns[idx]: float(sim) for idx, sim in zip(indices, similarities)}

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

        # Dimension guard: a prebuilt matrix (e.g. Jina 768) loaded while the
        # active query encoder is a different model (e.g. MiniLM 384) would make
        # this dot product raise and crash the whole query. Embedding similarity
        # is only a confidence TIEBREAKER, so degrade gracefully (return 0.0) and
        # warn once instead of failing. Fix is to set SG_EMBED_MODEL to the model
        # the index was built with.
        if query_emb.shape[-1] != self.matrix.shape[-1]:
            cls = type(self)
            if not getattr(cls, "_dim_warned", False):
                cls._dim_warned = True
                logger.warning(
                    "Embedding dim mismatch: index=%d, query encoder=%d. "
                    "Skipping embedding similarity (set SG_EMBED_MODEL to the "
                    "model the index was built with). Warning shown once.",
                    self.matrix.shape[-1], query_emb.shape[-1])
            return 0.0

        # query_emb shape is (1, D) — use [0] to get (D,) so the dot product with
        # matrix[idx] (D,) returns a numpy scalar, not a (1,) array.
        return float(self.matrix[idx] @ query_emb[0])

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, sg_dir: Path) -> None:
        """Save embeddings to .skeletongraph/embeddings.npz."""
        if self.is_empty:
            return

        import json
        path = sg_dir / "embeddings.npz"
        # np.savez_compressed APPENDS ".npz" to any path that does not already
        # end in it — so the temp file MUST end in ".npz", otherwise numpy
        # writes "embeddings.tmp.npz" and the rename of "embeddings.tmp" below
        # fails with FileNotFoundError. (This silently disabled SG embeddings.)
        tmp_path = sg_dir / "embeddings.tmp.npz"

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
