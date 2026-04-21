"""
Context resolver: skeleton-first retrieval with page-fault expansion.

The core retrieval algorithm:
  1. Parse user intent → identify target entities
  2. Resolve entities to FQNs (Bloom filter → skeleton table)
  3. Expand via graph traversal (blast_radius / dependency_chain)
  4. Rank candidates using multi-signal ranker
  5. Assign tiers (Tier 1: full body, Tier 2: skeleton + summary, Tier 3: FQN only)
  6. Session-aware deduplication (skip bodies agent already has)
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
from .ranker import Ranker, RankWeights
from .session import Session


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
    session_cached: bool = False  # True if body was sent in a previous turn


@dataclass
class ResolverResult:
    """Output of the resolver: ranked candidates ready for zone assembly."""
    candidates: List[RankedCandidate]
    intent: Intent
    confidence: str = "HIGH"     # HIGH, MEDIUM, LOW
    confidence_reason: str = ""
    entities_matched: List[str] = field(default_factory=list)
    session_dedup_count: int = 0  # How many Zone 2 bodies were skipped


def resolve_context(
    prompt: str,
    store: IndexStore,
    max_depth: int = 2,
    session: Optional[Session] = None,
) -> ResolverResult:
    """Main entry point: prompt → ranked candidates.

    Args:
        prompt: User's natural language request.
        store: The loaded index.
        max_depth: Maximum graph traversal depth.
        session: Optional session for cross-turn deduplication.

    Returns:
        ResolverResult with ranked candidates and confidence.
    """
    # Step 0: Session-based anaphora resolution
    if session and _has_anaphora(prompt):
        last_targets = session.get_last_target_fqns()
        if last_targets:
            # "it", "that", "this" → resolve to last turn's targets
            prompt = _resolve_anaphora(prompt, last_targets, store)

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
        # Fallback 1: Semantic BM25 search over LLM Summaries (if any exist)
        if len(store.summaries._store) > 0:
            from ..graph.bm25 import BM25Model
            bm25 = BM25Model()
            bm25.fit(store.summaries._store)
            bm25_results = bm25.search(prompt, top_k=5)
            if bm25_results:
                target_fqns = {fqn for fqn, _ in bm25_results}
                confidence = "MEDIUM"
                confidence_reason = "Matched via semantic BM25 summary search"

        # Fallback 2: Keyword search on Inverted Index (if BM25 fails or no summaries)
        if not target_fqns:
            search_results = store.inverted_index.search(prompt, top_k=5)
            if search_results:
                target_fqns = {fqn for fqn, _ in search_results}
                confidence = "MEDIUM"
                confidence_reason = "Matched via keyword search"
            else:
                confidence = "LOW"
                confidence_reason = "No entity, semantic, or keyword matches found"

    # Step 4: Build ranker with hub scores
    ranker = Ranker(store.graph)
    ranker.compute_hub_scores(known_fqns)

    # Determine target file for same-file bonus
    target_file = ""
    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            target_file = sk.file_path
            break

    # Step 5: Expand via graph traversal
    candidates: Dict[str, RankedCandidate] = {}
    session_dedup_count = 0

    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            is_cached = session.should_skip_body(fqn) if session else False
            if is_cached:
                session_dedup_count += 1
            candidates[fqn] = RankedCandidate(
                skeleton=sk, tier=Tier.TIER1, distance=0,
                score=ranker.score(fqn, sk, 0, "Direct target", target_file),
                reason="Direct target",
                session_cached=is_cached,
            )

    # Expand based on task type
    if intent.task_type in (TaskType.DEBUG, TaskType.EDIT, TaskType.REFACTOR):
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
                            score=ranker.score(
                                affected_fqn, sk, dist,
                                f"Blast radius (depth {dist})", target_file,
                            ),
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
                            score=ranker.score(
                                dep_fqn, sk, dist,
                                f"Dependency (depth {dist})", target_file,
                            ),
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
                            score=ranker.score(
                                test_fqn, sk, 1, "Test coverage", target_file,
                            ),
                            reason="Test coverage",
                        )

    elif intent.task_type == TaskType.EXPLAIN:
        for fqn in list(target_fqns):
            deps = store.graph.dependency_chain(fqn, max_depth=max_depth + 1)
            for dep_fqn, dist in deps.items():
                if dep_fqn not in candidates:
                    sk = store.skeleton_table.get(dep_fqn)
                    if sk:
                        tier = Tier.TIER2 if dist <= 2 else Tier.TIER3
                        candidates[dep_fqn] = RankedCandidate(
                            skeleton=sk, tier=tier, distance=dist,
                            score=ranker.score(
                                dep_fqn, sk, dist,
                                f"Dependency for explanation (depth {dist})",
                                target_file,
                            ),
                            reason=f"Dependency for explanation (depth {dist})",
                        )

    elif intent.task_type == TaskType.CREATE:
        for fqn in list(target_fqns):
            file_path = store.skeleton_table[fqn].file_path if fqn in store.skeleton_table else ""
            if file_path and file_path in store.file_skeletons:
                for sk in store.file_skeletons[file_path].all_skeletons:
                    if sk.fqn not in candidates:
                        candidates[sk.fqn] = RankedCandidate(
                            skeleton=sk, tier=Tier.TIER2, distance=1,
                            score=ranker.score(
                                sk.fqn, sk, 1, "Same file context", target_file,
                            ),
                            reason="Same file context",
                        )

    elif intent.task_type == TaskType.REVIEW:
        # For review tasks, include all changed functions and their blast radius
        for fqn in list(target_fqns):
            affected = store.graph.blast_radius(fqn, max_depth=max_depth)
            for affected_fqn, dist in affected.items():
                if affected_fqn not in candidates:
                    sk = store.skeleton_table.get(affected_fqn)
                    if sk:
                        tier = Tier.TIER2 if dist == 1 else Tier.TIER3
                        candidates[affected_fqn] = RankedCandidate(
                            skeleton=sk, tier=tier, distance=dist,
                            score=ranker.score(
                                affected_fqn, sk, dist,
                                f"Review blast radius (depth {dist})",
                                target_file,
                            ),
                            reason=f"Review blast radius (depth {dist})",
                        )

    # Step 6: Auto-include constructors for any class that's in context
    _auto_include_constructors(candidates, store, ranker, target_file)

    # Step 7: Rank and sort using the ranker
    ranked = ranker.rank_candidates(candidates)

    return ResolverResult(
        candidates=ranked,
        intent=intent,
        confidence=confidence,
        confidence_reason=confidence_reason,
        entities_matched=[e.value for e in intent.entities],
        session_dedup_count=session_dedup_count,
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
    ranker: Ranker,
    target_file: str,
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
                    score=ranker.score(
                        ctor_fqn, sk, 0, "Auto-included constructor",
                        target_file,
                    ),
                    reason="Auto-included constructor",
                )


def _has_anaphora(prompt: str) -> bool:
    """Check if prompt contains anaphoric references (it, that, this, etc.)."""
    anaphora = {"it", "that", "this", "the same", "those"}
    words = set(prompt.lower().split())
    return bool(words & anaphora)


def _resolve_anaphora(
    prompt: str,
    last_targets: Set[str],
    store: IndexStore,
) -> str:
    """Resolve anaphoric references by appending target context.

    Doesn't modify the original prompt — appends a hint for intent parsing.
    """
    if not last_targets:
        return prompt

    # Get short names of last targets
    short_names = []
    for fqn in last_targets:
        sk = store.skeleton_table.get(fqn)
        if sk:
            name = fqn.split("::")[-1] if "::" in fqn else fqn
            short = name.split(".")[-1] if "." in name else name
            short_names.append(short)

    if short_names:
        context_hint = f" [context: {', '.join(short_names[:3])}]"
        return prompt + context_hint

    return prompt
