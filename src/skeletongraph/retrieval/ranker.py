"""
Multi-signal candidate ranking.

Replaces hardcoded score values in resolver.py with a configurable,
normalized ranking system. Each signal contributes a weighted score
that is normalized to [0, 1] before combining.

Signals:
  1. Distance (graph hops from target)
  2. Connectivity (hub score — how many things depend on this)
  3. Complexity (cyclomatic complexity — complex code needs more context)
  4. Recency (recently modified files get priority)
  5. Test coverage (test functions for the target)
  6. Export status (public API > internal helper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..parser.skeleton import SkeletonCore
from ..graph.dependency import DependencyGraph


@dataclass
class RankWeights:
    """Configurable signal weights for ranking."""
    distance: float = 10.0       # Closer = higher (inverse)
    connectivity: float = 3.0    # More connections = higher
    complexity: float = 1.5      # More complex = needs more context
    is_test: float = 4.0         # Tests for target get bonus
    is_exported: float = 2.0     # Public API gets bonus
    same_file: float = 3.0       # Same file as target gets bonus
    is_constructor: float = 5.0  # Constructors auto-included at high priority


class Ranker:
    """Multi-signal candidate ranker with configurable weights."""

    def __init__(
        self,
        graph: DependencyGraph,
        weights: Optional[RankWeights] = None,
        centrality_enabled: bool = True,
    ) -> None:
        self.graph = graph
        self.weights = weights or RankWeights()
        self._hub_scores: Dict[str, float] = {}
        # When False (sg-norerank ablation): hub/centrality signal is silenced.
        # Distance, complexity, test, export, and same-file signals are preserved.
        self._centrality_enabled = centrality_enabled

    def compute_hub_scores(self, all_fqns: Set[str]) -> None:
        """Pre-compute hub scores (in-degree based) for all nodes.

        Hub score = number of reverse edges (things that depend on this node).
        Normalized to [0, 1] range.
        """
        max_in_degree = 1
        for fqn in all_fqns:
            in_degree = len(self.graph.reverse.get(fqn, []))
            self._hub_scores[fqn] = in_degree
            max_in_degree = max(max_in_degree, in_degree)

        # Normalize
        for fqn in self._hub_scores:
            self._hub_scores[fqn] /= max_in_degree

    def score(
        self,
        fqn: str,
        skeleton: SkeletonCore,
        distance: int,
        reason: str,
        target_file: str = "",
    ) -> float:
        """Compute composite relevance score for a candidate.

        Args:
            fqn: The candidate's FQN.
            skeleton: The candidate's SkeletonCore.
            distance: Graph distance from the target entity.
            reason: Why this candidate was included.
            target_file: File path of the primary target (for same-file bonus).

        Returns:
            Composite score (higher = more relevant).
        """
        score = 0.0

        # Signal 1: Distance (inverse — closer = higher)
        if distance == 0:
            score += self.weights.distance  # Direct target
        elif distance > 0:
            score += self.weights.distance / distance

        # Signal 2: Connectivity (hub score — disabled for sg-norerank ablation)
        if self._centrality_enabled:
            hub = self._hub_scores.get(fqn, 0.0)
            score += hub * self.weights.connectivity

        # Signal 3: Complexity (check higher threshold first)
        if skeleton.complexity > 10:
            score += self.weights.complexity * 1.5  # Very complex code
        elif skeleton.complexity > 5:
            score += self.weights.complexity  # Moderately complex code

        # Signal 4: Test bonus
        if "test" in reason.lower() or "test" in fqn.lower():
            score += self.weights.is_test

        # Signal 5: Export status
        if skeleton.is_exported:
            score += self.weights.is_exported

        # Signal 6: Same file bonus
        if target_file and skeleton.file_path == target_file:
            score += self.weights.same_file

        # Signal 7: Constructor bonus
        if skeleton.kind.auto_include_constructor:
            score += self.weights.is_constructor

        return round(score, 2)

    def rank_candidates(
        self,
        candidates: Dict[str, "RankedCandidate"],
        top_n: int = 50,
    ) -> List["RankedCandidate"]:
        """Sort candidates by tier (ascending) then score (descending) and truncate.

        Tier 1 always comes first, then Tier 2, then Tier 3.
        Within each tier, higher scores come first.
        Truncates to top_n total candidates to prevent context bloat.
        """
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: (c.tier.value, -c.score),
        )
        return sorted_candidates[:top_n]
