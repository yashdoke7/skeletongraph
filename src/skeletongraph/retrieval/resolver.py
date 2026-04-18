"""
Context resolver: skeleton-first retrieval with page-fault expansion.

The core retrieval algorithm:
  1. Parse user intent → identify target entities
  2. Resolve entities to FQNs (Bloom filter → skeleton table)
  3. Expand via graph traversal (blast_radius / dependency_chain)
  4. Rank candidates by relevance
  5. Assign tiers (Tier 1: full body, Tier 2: skeleton + summary, Tier 3: FQN only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..graph.dependency import DependencyGraph, EdgeType
from ..graph.inverted_index import InvertedIndex
from ..parser.skeleton import SkeletonCore
from ..storage.local import IndexStore
from ..summary.summary_store import SummaryStore
from .intent import Intent, TaskType, analyze_intent


class Tier(Enum):
    """Retrieval tier determines how much detail is included."""
    TIER1 = 1  # Full function body (target of the edit)
    TIER2 = 2  # Skeleton + summary (1-hop neighbors)
    TIER3 = 3  # FQN + return type only (2-hop periphery)


@dataclass
class RankedCandidate:
    """A skeleton candidate with computed relevance score and assigned tier."""
    skeleton: SkeletonCore
    tier: Tier
    distance: int = 0      # Hops from the target entity
    score: float = 0.0     # Composite relevance score
    reason: str = ""       # Why this was included (for debugging)


@dataclass
class ResolverResult:
    """Output of the resolver: ranked candidates ready for zone assembly."""
    candidates: List[RankedCandidate]
    intent: Intent
    confidence: str = "HIGH"     # HIGH, MEDIUM, LOW
    confidence_reason: str = ""
    entities_matched: List[str] = field(default_factory=list)


def resolve_context(
    prompt: str,
    store: IndexStore,
    max_depth: int = 2,
) -> ResolverResult:
    """Main entry point: prompt → ranked candidates.

    Args:
        prompt: User's natural language request.
        store: The loaded index.
        max_depth: Maximum graph traversal depth.

    Returns:
        ResolverResult with ranked candidates and confidence.
    """
    # Step 1: Analyze intent
    known_files = set(store.file_skeletons.keys())
    known_fqns = set(store.skeleton_table.keys())
    intent = analyze_intent(prompt, known_files, known_fqns)

    # Step 2: Resolve entities to FQNs
    target_fqns = _resolve_entities(intent, store)

    # Step 3: Determine confidence
    if target_fqns:
        confidence = "HIGH"
        confidence_reason = f"Exact entity match: {', '.join(list(target_fqns)[:3])}"
    else:
        # Fallback: search inverted index
        search_results = store.inverted_index.search(prompt, top_k=5)
        if search_results:
            target_fqns = {fqn for fqn, _ in search_results}
            confidence = "MEDIUM"
            confidence_reason = "Matched via keyword search"
        else:
            confidence = "LOW"
            confidence_reason = "No entity or keyword matches found"

    # Step 4: Expand via graph traversal
    candidates: Dict[str, RankedCandidate] = {}

    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            candidates[fqn] = RankedCandidate(
                skeleton=sk, tier=Tier.TIER1, distance=0,
                score=10.0, reason="Direct target",
            )

    # Expand based on task type
    if intent.task_type in (TaskType.DEBUG, TaskType.EDIT, TaskType.REFACTOR):
        # For edit/debug: show what CALLS the target (blast radius)
        # and what the target DEPENDS on
        for fqn in list(target_fqns):
            # Blast radius (who calls this?)
            affected = store.graph.blast_radius(fqn, max_depth=max_depth)
            for affected_fqn, dist in affected.items():
                if affected_fqn not in candidates:
                    sk = store.skeleton_table.get(affected_fqn)
                    if sk:
                        tier = Tier.TIER2 if dist == 1 else Tier.TIER3
                        candidates[affected_fqn] = RankedCandidate(
                            skeleton=sk, tier=tier, distance=dist,
                            score=5.0 / dist,
                            reason=f"Blast radius (depth {dist})",
                        )

            # Dependency chain (what does this call?)
            deps = store.graph.dependency_chain(fqn, max_depth=max_depth)
            for dep_fqn, dist in deps.items():
                if dep_fqn not in candidates:
                    sk = store.skeleton_table.get(dep_fqn)
                    if sk:
                        tier = Tier.TIER2 if dist == 1 else Tier.TIER3
                        candidates[dep_fqn] = RankedCandidate(
                            skeleton=sk, tier=tier, distance=dist,
                            score=4.0 / dist,
                            reason=f"Dependency (depth {dist})",
                        )

        # Include related tests
        for fqn in list(target_fqns):
            tests = store.graph.test_coverage(fqn)
            for test_fqn in tests:
                if test_fqn not in candidates:
                    sk = store.skeleton_table.get(test_fqn)
                    if sk:
                        candidates[test_fqn] = RankedCandidate(
                            skeleton=sk, tier=Tier.TIER2, distance=1,
                            score=3.0, reason="Test coverage",
                        )

    elif intent.task_type == TaskType.EXPLAIN:
        # For explain: show dependencies (what does it depend on?)
        for fqn in list(target_fqns):
            deps = store.graph.dependency_chain(fqn, max_depth=max_depth + 1)
            for dep_fqn, dist in deps.items():
                if dep_fqn not in candidates:
                    sk = store.skeleton_table.get(dep_fqn)
                    if sk:
                        tier = Tier.TIER2 if dist <= 2 else Tier.TIER3
                        candidates[dep_fqn] = RankedCandidate(
                            skeleton=sk, tier=tier, distance=dist,
                            score=4.0 / dist,
                            reason=f"Dependency for explanation (depth {dist})",
                        )

    elif intent.task_type == TaskType.CREATE:
        # For create: show similar functions and interfaces
        for fqn in list(target_fqns):
            # Show what's already in the file
            file_path = store.skeleton_table[fqn].file_path if fqn in store.skeleton_table else ""
            if file_path and file_path in store.file_skeletons:
                for sk in store.file_skeletons[file_path].all_skeletons:
                    if sk.fqn not in candidates:
                        candidates[sk.fqn] = RankedCandidate(
                            skeleton=sk, tier=Tier.TIER2, distance=1,
                            score=2.0, reason="Same file context",
                        )

    # Step 5: Auto-include constructors for any class that's in context
    _auto_include_constructors(candidates, store)

    # Step 6: Rank and sort
    ranked = sorted(candidates.values(), key=lambda c: (-c.tier.value, -c.score))
    # Re-sort: Tier 1 first, then Tier 2, then Tier 3, each by score descending
    ranked = sorted(ranked, key=lambda c: (c.tier.value, -c.score))

    return ResolverResult(
        candidates=ranked,
        intent=intent,
        confidence=confidence,
        confidence_reason=confidence_reason,
        entities_matched=[e.value for e in intent.entities],
    )


def _resolve_entities(intent: Intent, store: IndexStore) -> Set[str]:
    """Resolve intent entities to concrete FQNs."""
    fqns: Set[str] = set()

    # File path mentions → include all functions in that file
    for file_path in intent.file_paths:
        if file_path in store.file_skeletons:
            # For specific function mentions, only include those
            if intent.function_names:
                for sk in store.file_skeletons[file_path].all_skeletons:
                    name = sk.fqn.split("::")[-1] if "::" in sk.fqn else sk.fqn
                    short = name.split(".")[-1] if "." in name else name
                    if short in intent.function_names or name in intent.function_names:
                        fqns.add(sk.fqn)
            else:
                # No specific function → include top-level functions only
                for fn in store.file_skeletons[file_path].functions:
                    fqns.add(fn.fqn)

    # Function name mentions not tied to a file
    if not fqns and intent.function_names:
        for name in intent.function_names:
            for fqn, sk in store.skeleton_table.items():
                short = fqn.split("::")[-1] if "::" in fqn else fqn
                if short == name or short.endswith(f".{name}"):
                    fqns.add(fqn)

    # File path only, no function → all exported/public functions
    if not fqns and intent.file_paths:
        for file_path in intent.file_paths:
            if file_path in store.file_skeletons:
                for sk in store.file_skeletons[file_path].all_skeletons:
                    if sk.is_exported:
                        fqns.add(sk.fqn)

    return fqns


def _auto_include_constructors(
    candidates: Dict[str, RankedCandidate],
    store: IndexStore,
) -> None:
    """If a class is in context, auto-include its constructor."""
    class_fqns = [
        c.skeleton.fqn for c in candidates.values()
        if c.skeleton.kind.auto_include_constructor
    ]
    for class_fqn in class_fqns:
        # Constructor FQN = class_fqn + ".__init__" (Python) or ".constructor" (TS)
        for suffix in (".__init__", ".constructor"):
            ctor_fqn = class_fqn + suffix
            if ctor_fqn in store.skeleton_table and ctor_fqn not in candidates:
                sk = store.skeleton_table[ctor_fqn]
                candidates[ctor_fqn] = RankedCandidate(
                    skeleton=sk, tier=Tier.TIER2, distance=0,
                    score=6.0, reason="Auto-included constructor",
                )
