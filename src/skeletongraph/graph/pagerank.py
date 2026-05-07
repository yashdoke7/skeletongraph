"""
PageRank computation for the dependency graph.

Computes importance scores for every function/class in the codebase.
Hub functions (high PageRank) are:
  - More likely to be relevant to any query
  - Prioritized in the SLM function index
  - Auto-summarized first during `sg build`

Algorithm: Power iteration on the adjacency matrix.
Computed once at build time, stored in pagerank.json.
Zero cost at query time — just a dict lookup.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def compute_pagerank(
    edges: List[Tuple[str, str]],
    nodes: Optional[Set[str]] = None,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Compute PageRank scores for all nodes in the graph.

    Args:
        edges: List of (caller, callee) directed edges.
        nodes: Optional explicit node set. If None, inferred from edges.
        damping: Damping factor (probability of following an edge vs random jump).
        max_iterations: Maximum iterations for convergence.
        tolerance: Convergence threshold (L1 norm of score changes).

    Returns:
        Dict mapping FQN → PageRank score (0-1, sums to ~1.0).
    """
    # Build adjacency lists
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    incoming: Dict[str, Set[str]] = defaultdict(set)
    all_nodes: Set[str] = set()

    for src, dst in edges:
        outgoing[src].add(dst)
        incoming[dst].add(src)
        all_nodes.add(src)
        all_nodes.add(dst)

    if nodes:
        all_nodes.update(nodes)

    n = len(all_nodes)
    if n == 0:
        return {}

    # Initialize uniform scores
    scores: Dict[str, float] = {node: 1.0 / n for node in all_nodes}
    base_score = (1.0 - damping) / n

    for iteration in range(max_iterations):
        new_scores: Dict[str, float] = {}
        delta = 0.0

        # Handle dangling nodes (no outgoing edges): distribute evenly
        dangling_sum = sum(
            scores[node] for node in all_nodes
            if not outgoing.get(node)
        )
        dangling_contrib = damping * dangling_sum / n

        for node in all_nodes:
            # Sum of incoming contributions
            incoming_sum = sum(
                scores[src] / len(outgoing[src])
                for src in incoming.get(node, set())
                if outgoing.get(src)
            )

            new_score = base_score + damping * incoming_sum + dangling_contrib
            new_scores[node] = new_score
            delta += abs(new_score - scores[node])

        scores = new_scores

        if delta < tolerance:
            logger.debug("PageRank converged in %d iterations (delta=%.2e)", iteration + 1, delta)
            break
    else:
        logger.debug("PageRank did not converge after %d iterations (delta=%.2e)", max_iterations, delta)

    return scores


def top_n(scores: Dict[str, float], n: int = 100) -> List[str]:
    """Return the top N FQNs by PageRank score."""
    return sorted(scores, key=scores.get, reverse=True)[:n]


def percentile_threshold(scores: Dict[str, float], percentile: float = 0.80) -> float:
    """Return the PageRank score at a given percentile.

    Used to determine the auto-summarize threshold:
    top 20% = percentile_threshold(scores, 0.80)
    """
    if not scores:
        return 0.0
    sorted_scores = sorted(scores.values())
    idx = int(len(sorted_scores) * percentile)
    idx = min(idx, len(sorted_scores) - 1)
    return sorted_scores[idx]


def get_hub_functions(
    scores: Dict[str, float],
    top_percent: float = 0.20,
) -> List[str]:
    """Return FQNs in the top N% by PageRank (hub functions).

    These are auto-summarized during `sg build` because they're
    most likely to appear in retrieval results.
    """
    if not scores:
        return []
    threshold = percentile_threshold(scores, 1.0 - top_percent)
    return [fqn for fqn, score in scores.items() if score >= threshold]
