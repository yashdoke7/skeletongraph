"""
Confidence scoring system for routing decisions.

5-factor composite score determines:
  - HIGH (≥0.7): Graph alone assembles complete prompt
  - MEDIUM (0.4-0.7): Graph + optional SLM review (Enhanced mode)
  - LOW (0.2-0.4): SLM required for context expansion
  - MISS (<0.2): Pass-through — tell agent to explore freely

Weights are calibratable per-project via calibration.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph.embeddings import EmbeddingStore
    from ..storage.local import IndexStore

logger = logging.getLogger(__name__)


# Default weights — calibrated via benchmark (Day 7)
_DEFAULT_WEIGHTS = {
    "entity_match": 0.35,
    "coverage": 0.25,
    "ambiguity": 0.15,
    "dependency_depth": 0.15,
    "cross_file": 0.10,
}

# Thresholds for confidence levels
_DEFAULT_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
    "low": 0.20,
}


@dataclass
class ConfidenceScore:
    """5-factor confidence score for retrieval quality assessment.

    Each factor is 0.0-1.0. Composite is a weighted sum.
    """

    entity_match: float = 0.0
    """1.0 = exact FQN match from prompt, 0.5 = partial/fuzzy, 0.0 = no match"""

    coverage: float = 0.0
    """Semantic similarity between query and top candidate (via embeddings).
    Falls back to keyword overlap ratio if embeddings unavailable."""

    ambiguity: float = 0.0
    """1.0 = single clear target, 0.5 = 2-3 candidates, 0.1 = 10+ candidates"""

    dependency_depth: float = 0.0
    """1.0 = target has ≤2 hop deps, 0.5 = 3-5 hops, 0.0 = 6+ hops or cyclic"""

    cross_file: float = 0.0
    """1.0 = all targets in 1 file, 0.5 = 2-3 files, 0.0 = 5+ files"""

    # Metadata for debugging
    factors_detail: Dict[str, str] = field(default_factory=dict)

    def composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted composite score."""
        w = weights or _DEFAULT_WEIGHTS
        return (
            self.entity_match * w.get("entity_match", 0.35)
            + self.coverage * w.get("coverage", 0.25)
            + self.ambiguity * w.get("ambiguity", 0.15)
            + self.dependency_depth * w.get("dependency_depth", 0.15)
            + self.cross_file * w.get("cross_file", 0.10)
        )

    def level(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> str:
        """Determine confidence level: HIGH, MEDIUM, LOW, or MISS."""
        t = thresholds or _DEFAULT_THRESHOLDS
        c = self.composite(weights)
        if c >= t.get("high", 0.70):
            return "HIGH"
        if c >= t.get("medium", 0.40):
            return "MEDIUM"
        if c >= t.get("low", 0.20):
            return "LOW"
        return "MISS"

    def to_dict(self) -> dict:
        return {
            "entity_match": round(self.entity_match, 3),
            "coverage": round(self.coverage, 3),
            "ambiguity": round(self.ambiguity, 3),
            "dependency_depth": round(self.dependency_depth, 3),
            "cross_file": round(self.cross_file, 3),
            "composite": round(self.composite(), 3),
            "level": self.level(),
            "detail": self.factors_detail,
        }


def compute_confidence(
    query: str,
    target_fqns: Set[str],
    store: "IndexStore",
    embeddings: Optional["EmbeddingStore"] = None,
    match_source: str = "none",
) -> ConfidenceScore:
    """Compute the 5-factor confidence score for a retrieval result.

    Args:
        query: The user's natural language prompt.
        target_fqns: Set of FQNs that the resolver found.
        store: The loaded index store.
        embeddings: Optional embedding store for semantic coverage.
        match_source: How the targets were found ("entity", "bm25", "keyword", "none").

    Returns:
        ConfidenceScore with all 5 factors computed.
    """
    score = ConfidenceScore()

    if not target_fqns:
        score.factors_detail["entity_match"] = "No targets found"
        return score  # All zeros → MISS

    # ── Factor 1: Entity Match ──────────────────────────────────────────
    if match_source == "entity":
        score.entity_match = 1.0
        score.factors_detail["entity_match"] = "Exact entity match from prompt"
    elif match_source == "bm25":
        score.entity_match = 0.6
        score.factors_detail["entity_match"] = "Semantic BM25 summary match"
    elif match_source == "keyword":
        score.entity_match = 0.4
        score.factors_detail["entity_match"] = "Keyword search match"
    else:
        score.entity_match = 0.0
        score.factors_detail["entity_match"] = "No match source"

    # ── Factor 2: Coverage (semantic similarity) ─────────────────────────
    if embeddings and not embeddings.is_empty:
        # Average similarity between query and all target FQNs
        similarities = []
        for fqn in target_fqns:
            sim = embeddings.get_similarity(fqn, query)
            similarities.append(sim)

        if similarities:
            # Use max similarity (best candidate relevance)
            best_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            # Blend: 70% best candidate, 30% average
            score.coverage = min(1.0, best_sim * 0.7 + avg_sim * 0.3)
            score.factors_detail["coverage"] = (
                f"Embedding similarity: best={best_sim:.3f}, avg={avg_sim:.3f}"
            )
    else:
        # Fallback: keyword overlap ratio
        from ..graph.inverted_index import tokenize_text
        query_tokens = set(tokenize_text(query))
        if query_tokens:
            matched_tokens = set()
            for fqn in target_fqns:
                sk = store.skeleton_table.get(fqn)
                if sk:
                    name_tokens = set(tokenize_text(sk.fqn.split("::")[-1]))
                    sig_tokens = set(tokenize_text(sk.signature))
                    matched_tokens |= (query_tokens & (name_tokens | sig_tokens))

            score.coverage = len(matched_tokens) / len(query_tokens)
            score.factors_detail["coverage"] = (
                f"Keyword overlap: {len(matched_tokens)}/{len(query_tokens)} terms"
            )
        else:
            score.coverage = 0.0

    # ── Factor 3: Ambiguity ──────────────────────────────────────────────
    n_targets = len(target_fqns)
    if n_targets == 1:
        score.ambiguity = 1.0
    elif n_targets <= 3:
        score.ambiguity = 0.7
    elif n_targets <= 5:
        score.ambiguity = 0.5
    elif n_targets <= 10:
        score.ambiguity = 0.3
    else:
        score.ambiguity = 0.1
    score.factors_detail["ambiguity"] = f"{n_targets} target(s) found"

    # ── Factor 4: Dependency Depth ───────────────────────────────────────
    max_depth = 0
    for fqn in list(target_fqns)[:5]:  # Check top 5 to avoid O(n²)
        # Count total reachable nodes within 6 hops
        blast = store.graph.blast_radius(fqn, max_depth=6)
        deps = store.graph.dependency_chain(fqn, max_depth=6)
        total_reachable = len(blast) + len(deps)
        if total_reachable > max_depth:
            max_depth = total_reachable

    if max_depth <= 5:
        score.dependency_depth = 1.0
    elif max_depth <= 15:
        score.dependency_depth = 0.7
    elif max_depth <= 30:
        score.dependency_depth = 0.4
    else:
        score.dependency_depth = 0.2
    score.factors_detail["dependency_depth"] = f"{max_depth} reachable nodes"

    # ── Factor 5: Cross-File ─────────────────────────────────────────────
    files_involved: Set[str] = set()
    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            files_involved.add(sk.file_path)
        # Also count 1-hop neighbor files
        blast = store.graph.blast_radius(fqn, max_depth=1)
        deps = store.graph.dependency_chain(fqn, max_depth=1)
        for neighbor_fqn in list(blast.keys()) + list(deps.keys()):
            n_sk = store.skeleton_table.get(neighbor_fqn)
            if n_sk:
                files_involved.add(n_sk.file_path)

    n_files = len(files_involved)
    if n_files <= 1:
        score.cross_file = 1.0
    elif n_files <= 2:
        score.cross_file = 0.7
    elif n_files <= 3:
        score.cross_file = 0.5
    elif n_files <= 5:
        score.cross_file = 0.3
    else:
        score.cross_file = 0.1
    score.factors_detail["cross_file"] = f"{n_files} files involved"

    return score


@dataclass
class CalibrationConfig:
    """Per-project calibration of confidence weights and thresholds.

    Stored in .skeletongraph/calibration.json.
    Updated after benchmark runs (Day 7).
    """

    weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_THRESHOLDS))
    calibration_runs: int = 0
    last_calibrated: float = 0.0

    def save(self, sg_dir: Path) -> None:
        path = sg_dir / "calibration.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "weights": self.weights,
            "thresholds": self.thresholds,
            "calibration_runs": self.calibration_runs,
            "last_calibrated": self.last_calibrated,
        }, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, sg_dir: Path) -> CalibrationConfig:
        path = sg_dir / "calibration.json"
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                weights=data.get("weights", dict(_DEFAULT_WEIGHTS)),
                thresholds=data.get("thresholds", dict(_DEFAULT_THRESHOLDS)),
                calibration_runs=data.get("calibration_runs", 0),
                last_calibrated=data.get("last_calibrated", 0.0),
            )
        except Exception:
            return cls()
