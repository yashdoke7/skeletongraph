"""
Context resolver: skeleton-first retrieval with page-fault expansion.

The core retrieval algorithm:
  1. Parse user intent → identify target entities
  2. Resolve entities to FQNs (Bloom filter → skeleton table)
  3. Expand via graph traversal (blast_radius / dependency_chain)
  4. Rank candidates using multi-signal ranker
  5. Assign tiers (Tier 1: full body, Tier 2: skeleton + summary, Tier 3: FQN only)
  6. Session-aware deduplication (skip bodies agent already has)

v4/v5 additions:
  - ModeSpec-driven expansion: graph_direction, blast_depth, dep_depth, load_tests
    read directly from classifier ModeSpec instead of switching on TaskType
  - SLM entity resolution: resolves SLM-extracted FQNs via fuzzy matching
  - Uses tokenize_query() instead of tokenize_text() for query-time matching
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .classifier import ModeSpec

from ..graph.dependency import DependencyGraph, EdgeType
from ..graph.inverted_index import InvertedIndex, tokenize_query
from ..parser.skeleton import SkeletonCore
from ..storage.local import IndexStore
from ..summary.summary_store import SummaryStore
from .confidence import ConfidenceScore, compute_confidence
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
    confidence: str = "HIGH"     # HIGH, MEDIUM, LOW, MISS
    confidence_reason: str = ""
    confidence_score: Optional[ConfidenceScore] = None  # 5-factor score
    entities_matched: List[str] = field(default_factory=list)
    session_dedup_count: int = 0  # How many Zone 2 bodies were skipped


# Above this many resolved entity FQNs, an entity match is treated as AMBIGUOUS
# (likely a common token matching many same-named functions) by the gated
# weak-entity fallback. Precise references resolve to 1-3 FQNs.
_WEAK_ENTITY_THRESHOLD = 6


def resolve_context(
    prompt: str,
    store: IndexStore,
    max_depth: int = 2,
    session: Optional[Session] = None,
    top_n: int = 50,
    seed_fqns: Optional[Set[str]] = None,
    mode_spec: Optional["ModeSpec"] = None,
    enable_keyword_fallback: bool = False,
    enable_bm25_fallback: bool = False,
    enable_graph_expansion: bool = True,
    graph_expansion_policy: str = "gated",
    enable_centrality_rerank: bool = True,
    enable_weak_entity_fallback: bool = False,
    bm25_primary: bool = False,
    enable_dense_fallback: bool = False,
    dense_primary: bool = False,
    enable_hybrid_fusion: bool = False,
    dense_rerank: bool = False,
    keyword_embedded_dense: bool = False,
) -> ResolverResult:
    """Main entry point: prompt → ranked candidates.

    Args:
        prompt: User's natural language request.
        store: The loaded index.
        max_depth: Maximum graph traversal depth (used as fallback if no mode_spec).
        session: Optional session for cross-turn deduplication.
        top_n: Max number of candidates to return.
        seed_fqns: Optional exact or fuzzy FQN seeds supplied by the caller.
        mode_spec: Optional ModeSpec from classifier — drives graph expansion.
            When provided, graph_direction/blast_depth/dep_depth/load_tests
            are read from this spec instead of switching on TaskType.
        enable_graph_expansion: When False (sg-nograph ablation), skip all
            graph traversal — blast_radius and dependency_chain. Only direct
            entity matches are returned (Tier 1 only).
        enable_centrality_rerank: When False (sg-norerank ablation), disable
            the hub/PageRank connectivity signal in the Ranker.

    Returns:
        ResolverResult with ranked candidates and confidence.
    """
    # Step 0: Session-based anaphora resolution
    # Only trigger when session has previous turns (avoids phantom injection on Turn 1)
    if session and session.turn_count > 0 and _has_anaphora(prompt):
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
    explicit_seed_fqns = _resolve_seed_fqns(seed_fqns or set(), store)
    if explicit_seed_fqns:
        target_fqns.update(explicit_seed_fqns)

    # Track how targets were found (for confidence scoring)
    match_source = "none"

    # Normalized [0,1] lexical relevance scores from BM25/keyword fallback.
    # Preserved here so the ranker can inject them as a bonus instead of
    # re-ordering purely by graph centrality (which discards lexical signal).
    _lexical_scores: Dict[str, float] = {}

    # Step 3: Determine confidence
    if target_fqns:
        confidence = "HIGH"
        confidence_reason = f"Exact entity match: {', '.join(list(target_fqns)[:3])}"
        match_source = (
            "slm"
            if any(getattr(e, "entity_type", "") == "slm_entity" for e in intent.entities)
            else "entity"
        )
        # Weak-entity fallback (ablation, gated). An AMBIGUOUS entity match — a
        # common token that short-name-matched many FQNs — is likely coincidental
        # (e.g. "format" matching every *.format). Only THEN pull full-corpus BM25
        # and add its top hits as seeds. Gated on ambiguity so precise matches
        # (the common case, where SG's precision lives) are never diluted — this
        # is the disciplined version of the reverted blanket blend.
        if (enable_weak_entity_fallback and enable_bm25_fallback
                and len(target_fqns) > _WEAK_ENTITY_THRESHOLD):
            _bm = _bm25_fallback(prompt, store, top_k=15)
            if _bm:
                _lex_max = max(s for _, s in _bm) or 1.0
                _lexical_scores = {fqn: s / _lex_max for fqn, s in _bm}
                for _f, _s in _bm[:3]:        # only the top-3 lexical hits
                    target_fqns.add(_f)
    else:
        # Optional BM25 fallback (disabled by default)
        if enable_bm25_fallback:
            bm25_results = _bm25_fallback(prompt, store, top_k=15)
            if bm25_results:
                target_fqns = {fqn for fqn, _ in bm25_results}
                _lex_max = max(s for _, s in bm25_results) or 1.0
                _lexical_scores = {fqn: s / _lex_max for fqn, s in bm25_results}
                confidence = "MEDIUM" if len(bm25_results) > 3 else "LOW"
                confidence_reason = (
                    f"Matched {len(bm25_results)} entities via BM25 fallback"
                )
                match_source = "bm25"
            elif enable_keyword_fallback:
                search_results = store.inverted_index.search(prompt, top_k=15)
                if search_results:
                    target_fqns = {fqn for fqn, _ in search_results}
                    _lex_max = max(s for _, s in search_results) or 1.0
                    _lexical_scores = {fqn: s / _lex_max for fqn, s in search_results}
                    confidence = "MEDIUM" if len(search_results) > 3 else "LOW"
                    confidence_reason = (
                        f"Matched {len(search_results)} entities via keyword search"
                    )
                    match_source = "keyword"
                else:
                    confidence = "LOW"
                    confidence_reason = "No entity, BM25, or keyword matches found"
            else:
                confidence = "LOW"
                confidence_reason = "No entity or BM25 matches found"
        else:
            # Optional keyword fallback (disabled by default)
            if enable_keyword_fallback:
                search_results = store.inverted_index.search(prompt, top_k=15)
                if search_results:
                    target_fqns = {fqn for fqn, _ in search_results}
                    _lex_max = max(s for _, s in search_results) or 1.0
                    _lexical_scores = {fqn: s / _lex_max for fqn, s in search_results}
                    confidence = "MEDIUM" if len(search_results) > 3 else "LOW"
                    confidence_reason = (
                        f"Matched {len(search_results)} entities via keyword search"
                    )
                    match_source = "keyword"
                else:
                    confidence = "LOW"
                    confidence_reason = "No entity or keyword matches found"
            else:
                confidence = "LOW"
                confidence_reason = "No entity matches; keyword fallback disabled"

    # sg-rerank (the WINNING composition, default product retrieval): BM25 supplies
    # the wide RECALL pool — always, not just as a fallback — and the structural
    # ranker below (centrality + entity bonus + normalized lexical score) REORDERS
    # it. This is "generate-then-rerank": bm25's recall with SG's rank, at SG's
    # token cost (read_symbol fetches only the chosen function bodies). Entity
    # matches resolved above are kept and get the highest structural score; the
    # BM25 pool widens recall around them. Gated by `bm25_primary` so the lean
    # entity-first `sg` path and the ablations keep their original behavior.
    if enable_hybrid_fusion:
        _bm25_pool = _bm25_fallback(prompt, store, top_k=max(top_n, 25))
        _dense_pool = _dense_fallback(prompt, store, top_k=max(top_n, 25), keyword_embedded=keyword_embedded_dense)
        
        combined_scores = {}
        if _bm25_pool:
            _pmax = max(s for _, s in _bm25_pool) or 1.0
            for _f, _s in _bm25_pool:
                combined_scores[_f] = combined_scores.get(_f, 0.0) + (_s / _pmax)
                
        if _dense_pool:
            _dmax = max(s for _, s in _dense_pool) or 1.0
            for _f, _s in _dense_pool:
                combined_scores[_f] = combined_scores.get(_f, 0.0) + (_s / _dmax)
                
        sorted_fusion = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:max(top_n, 25)]
        for _f, _s in sorted_fusion:
            _lexical_scores[_f] = _s
            target_fqns.add(_f)
            
        if match_source in ("none", ""):
            match_source = "hybrid"

    elif dense_rerank:
        _bm25_pool = _bm25_fallback(prompt, store, top_k=max(top_n, 25))
        if _bm25_pool:
            pool_fqns = [_f for _f, _ in _bm25_pool]
            if hasattr(store, "embeddings") and not store.embeddings.is_empty:
                search_query = prompt
                if keyword_embedded_dense:
                    tokens = tokenize_query(prompt)
                    if tokens:
                        search_query = " ".join(tokens)
                dense_scores = store.embeddings.rescore(search_query, pool_fqns)
                if dense_scores:
                    _dmax = max(dense_scores.values()) or 1.0
                    for _f in pool_fqns:
                        _s = dense_scores.get(_f, 0.0)
                        _lexical_scores[_f] = _s / _dmax
                        target_fqns.add(_f)
                else:
                    _pmax = max(s for _, s in _bm25_pool) or 1.0
                    for _f, _s in _bm25_pool:
                        _lexical_scores[_f] = _s / _pmax
                        target_fqns.add(_f)
            else:
                _pmax = max(s for _, s in _bm25_pool) or 1.0
                for _f, _s in _bm25_pool:
                    _lexical_scores[_f] = _s / _pmax
                    target_fqns.add(_f)
                    
        if match_source in ("none", ""):
            match_source = "dense_rerank"

    else:
        if bm25_primary:
            _pool = _bm25_fallback(prompt, store, top_k=max(top_n, 25))
            if _pool:
                _pmax = max(s for _, s in _pool) or 1.0
                for _f, _s in _pool:
                    _lexical_scores.setdefault(_f, _s / _pmax)
                target_fqns.update(_f for _f, _ in _pool)
                if match_source in ("none", ""):
                    match_source = "bm25"
                    
        if dense_primary:
            _dense_pool = _dense_fallback(prompt, store, top_k=max(top_n, 25), keyword_embedded=keyword_embedded_dense)
            if _dense_pool:
                _dmax = max(s for _, s in _dense_pool) or 1.0
                for _f, _s in _dense_pool:
                    _lexical_scores.setdefault(_f, _s / _dmax)
                target_fqns.update(_f for _f, _ in _dense_pool)
                if match_source in ("none", ""):
                    match_source = "dense"
                    
        # Optional dense fallback (if entities and bm25 failed/were skipped)
        if enable_dense_fallback and not target_fqns and not dense_primary:
            dense_results = _dense_fallback(prompt, store, top_k=15, keyword_embedded=keyword_embedded_dense)
            if dense_results:
                target_fqns.update(fqn for fqn, _ in dense_results)
                _dmax = max(s for _, s in dense_results) or 1.0
                for fqn, s in dense_results:
                    _lexical_scores[fqn] = s / _dmax
                confidence = "MEDIUM" if len(dense_results) > 3 else "LOW"
                confidence_reason = (
                    f"Matched {len(dense_results)} entities via Dense Vector fallback"
                )
                match_source = "dense"

    # Step 4: Build ranker with hub scores
    ranker = Ranker(store.graph, centrality_enabled=enable_centrality_rerank)
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
    _LEX_WEIGHT = 8.0  # lexical bonus scale; comparable to distance=1 (10/1) weight

    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            is_cached = session.should_skip_body_hash(fqn, sk.sha256) if session else False
            if is_cached:
                session_dedup_count += 1
            base_score = ranker.score(fqn, sk, 0, "Direct target", target_file)
            lex_bonus = _lexical_scores.get(fqn, 0.0) * _LEX_WEIGHT
            candidates[fqn] = RankedCandidate(
                skeleton=sk, tier=Tier.TIER1, distance=0,
                score=base_score + lex_bonus,
                reason="Direct target",
                session_cached=is_cached,
            )

    # ── Graph expansion: ModeSpec-driven (v5) or TaskType-based (legacy) ──
    # sg-nograph ablation: skip entirely — only direct entity hits are returned.
    if enable_graph_expansion:
        expansion_spec = mode_spec
        expansion_seeds = set(target_fqns)
        policy = (graph_expansion_policy or "gated").lower()
        if policy == "off":
            expansion_spec = None
            expansion_seeds = set()
        elif policy == "gated":
            expansion_spec, expansion_seeds = _gate_graph_expansion(
                prompt=prompt,
                intent=intent,
                mode_spec=mode_spec,
                target_fqns=target_fqns,
                match_source=match_source,
                lexical_scores=_lexical_scores,
            )

        if expansion_spec is not None and expansion_seeds:
            _expand_from_mode_spec(
                expansion_seeds, expansion_spec, store, ranker, candidates, target_file,
            )
        elif policy == "always" and mode_spec is None:
            _expand_from_task_type(
                target_fqns, intent, max_depth, store, ranker, candidates, target_file,
            )

    # Step 6: Auto-include constructors for any class that's in context
    _auto_include_constructors(candidates, store, ranker, target_file)

    # Step 7: Rank and sort using the ranker
    ranked = ranker.rank_candidates(candidates, top_n=top_n)

    # Step 8: Compute 5-factor confidence score
    conf_score = compute_confidence(
        query=prompt,
        target_fqns=target_fqns,
        store=store,
        embeddings=store.embeddings if hasattr(store, 'embeddings') else None,
        match_source=match_source,
    )
    # Override string confidence with the computed level
    confidence = conf_score.level()
    confidence_reason = (
        f"{conf_score.level()} (composite={conf_score.composite():.2f}) "
        f"| {confidence_reason}"
    )

    return ResolverResult(
        candidates=ranked,
        intent=intent,
        confidence=confidence,
        confidence_reason=confidence_reason,
        confidence_score=conf_score,
        entities_matched=[e.value for e in intent.entities] + sorted(explicit_seed_fqns),
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
                # No specific function → include all classes and functions in that file
                for sk in store.file_skeletons[file_path].all_skeletons:
                    fqns.add(sk.fqn)

    # Function name mentions not tied to a file
    if not fqns and intent.function_names:
        for name in intent.function_names:
            if name in store.skeleton_table:
                fqns.add(name)
                continue
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


def _resolve_seed_fqns(seed_fqns: Set[str], store: IndexStore) -> Set[str]:
    """Resolve caller-supplied entity seeds to indexed FQNs.

    Agents can pass exact FQNs from a function index, while humans and wrappers
    often pass partial names. Keep the matching conservative and let ranking
    handle any remaining ambiguity.
    """
    resolved: Set[str] = set()
    for raw in seed_fqns:
        seed = raw.strip()
        if not seed:
            continue

        if seed in store.skeleton_table:
            resolved.add(seed)
            continue

        alt_seed = f"src/{seed}" if not seed.startswith("src/") else seed.replace("src/", "", 1)
        if alt_seed in store.skeleton_table:
            resolved.add(alt_seed)
            continue
        
        # Support Go/TS prefixes
        pfx_found = False
        for pfx in ("cmd/", "pkg/", "internal/", "lib/", "dist/", "packages/"):
            if seed.startswith(pfx):
                s2 = seed.replace(pfx, "", 1)
                if s2 in store.skeleton_table:
                    resolved.add(s2)
                    pfx_found = True
            else:
                s2 = f"{pfx}{seed}"
                if s2 in store.skeleton_table:
                    resolved.add(s2)
                    pfx_found = True
        if pfx_found:
            continue

        seed_name = seed.split("::")[-1]
        matches: List[str] = []
        for fqn in store.skeleton_table:
            short = fqn.split("::")[-1] if "::" in fqn else fqn
            if fqn.endswith(seed) or short == seed_name or short.endswith(f".{seed_name}"):
                matches.append(fqn)

        resolved.update(sorted(matches, key=len)[:5])

    return resolved


def _bm25_fallback(prompt: str, store: IndexStore, top_k: int = 10) -> List[Tuple[str, float]]:
    """Run BM25 search against the indexed token corpus."""
    from ..graph.bm25 import BM25Model

    if store.inverted_index.entry_count == 0:
        return []

    bm25 = store.bm25_model
    if bm25 is None or not bm25.is_fitted:
        corpus = store.inverted_index.build_bm25_corpus()
        if not corpus:
            return []
        bm25 = BM25Model()
        bm25.fit(corpus)
        store.bm25_model = bm25

    return bm25.search(prompt, top_k=top_k)


def _dense_fallback(prompt: str, store: IndexStore, top_k: int = 15, keyword_embedded: bool = False) -> List[Tuple[str, float]]:
    """Run Dense Vector semantic search using sentence-transformers."""
    if not hasattr(store, 'embeddings') or store.embeddings is None or store.embeddings.is_empty:
        return []
        
    search_query = prompt
    if keyword_embedded:
        tokens = tokenize_query(prompt)
        if tokens:
            search_query = " ".join(tokens)
            
    return store.embeddings.search(search_query, top_k=top_k)


# ── Graph expansion policy ──────────────────────────────────────────────

_GRAPH_REQUEST_TERMS = (
    "caller", "callers", "callee", "callees", "call graph", "dependency",
    "dependencies", "dependents", "blast radius", "impact", "affected",
    "uses of", "usage", "flow", "trace", "refactor", "migration", "migrate",
    "architecture", "design", "review",
)


def _gate_graph_expansion(
    prompt: str,
    intent: Intent,
    mode_spec: Optional["ModeSpec"],
    target_fqns: Set[str],
    match_source: str,
    lexical_scores: Dict[str, float],
) -> Tuple[Optional["ModeSpec"], Set[str]]:
    """Decide whether graph neighbors should enter the first packet.

    The old policy expanded from every seed whenever the mode allowed graph
    context. That is helpful for refactors and impact analysis, but noisy for
    bug-localization: a BM25 fallback can produce 15 seeds, and expanding all of
    them drowns the actual edit target. The gated policy keeps first-search
    packets precise and lets broad graph context appear only when the task asks
    for it or when we have a small, exact seed set.
    """
    if not target_fqns or mode_spec is None:
        return None, set()

    from .classifier import QueryMode

    prompt_l = prompt.lower()
    explicit_graph = any(term in prompt_l for term in _GRAPH_REQUEST_TERMS)
    graph_modes = {
        QueryMode.REFACTOR,
        QueryMode.REVIEW,
        QueryMode.MIGRATE,
        QueryMode.ARCHITECTURE,
    }
    precise_entity = match_source in {"entity", "slm"} and len(target_fqns) <= 3

    if mode_spec.mode == QueryMode.RETRIEVAL_FAST:
        return None, set()

    if precise_entity and mode_spec.mode in {
        QueryMode.DEBUG_TARGETED,
        QueryMode.TEST,
        QueryMode.DOCUMENT,
    }:
        # Exact bug/test/doc targets benefit from immediate 1-hop context, but
        # deeper traversal is where precision collapses.
        return replace(mode_spec, blast_depth=min(mode_spec.blast_depth, 1),
                       dep_depth=min(mode_spec.dep_depth, 1)), set(target_fqns)

    if explicit_graph or mode_spec.mode in graph_modes:
        seeds = _top_graph_seeds(target_fqns, lexical_scores, limit=4)
        return replace(
            mode_spec,
            blast_depth=min(mode_spec.blast_depth, 2),
            dep_depth=min(mode_spec.dep_depth, 2),
        ), seeds

    # BM25/keyword/no-entity fallback for normal fix/build prompts: keep the
    # retrieved candidates selective. The agent can request graph via intent or
    # sg_expand after seeing the top target.
    return None, set()


def _top_graph_seeds(
    target_fqns: Set[str],
    lexical_scores: Dict[str, float],
    limit: int,
) -> Set[str]:
    if not target_fqns:
        return set()
    if lexical_scores:
        ranked = sorted(target_fqns, key=lambda f: -lexical_scores.get(f, 0.0))
    else:
        ranked = sorted(target_fqns)
    return set(ranked[:limit])


# ── ModeSpec-driven expansion (v5) ──────────────────────────────────────


def _expand_from_mode_spec(
    target_fqns: Set[str],
    mode_spec: "ModeSpec",
    store: IndexStore,
    ranker: Ranker,
    candidates: Dict[str, RankedCandidate],
    target_file: str,
) -> None:
    """Expand graph based on ModeSpec fields. Replaces TaskType switching.

    Reads graph_direction, blast_depth, dep_depth, and load_tests directly
    from the classifier ModeSpec, making the 12-mode taxonomy functional.
    """
    direction = mode_spec.graph_direction
    blast_depth = mode_spec.blast_depth
    dep_depth = mode_spec.dep_depth

    # Skip expansion entirely for "none" direction modes (e.g. RETRIEVAL_FAST)
    if direction == "none" and not mode_spec.load_tests:
        return

    for fqn in list(target_fqns):
        # Reverse BFS (blast radius): "who calls this?"
        if direction in ("reverse", "both") and blast_depth > 0:
            affected = store.graph.blast_radius(fqn, max_depth=blast_depth)
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

        # Forward BFS (dependency chain): "what does this call?"
        if direction in ("forward", "both") and dep_depth > 0:
            deps = store.graph.dependency_chain(fqn, max_depth=dep_depth)
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

        # Test coverage (controlled by load_tests flag)
        if mode_spec.load_tests:
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


# ── Legacy TaskType-based expansion (v3 compat) ────────────────────────


def _expand_from_task_type(
    target_fqns: Set[str],
    intent: Intent,
    max_depth: int,
    store: IndexStore,
    ranker: Ranker,
    candidates: Dict[str, RankedCandidate],
    target_file: str,
) -> None:
    """Legacy expansion based on TaskType. Used when no ModeSpec is provided.

    Preserved for backward compatibility with v3 callers (e.g. MCP v3 fallback).
    """
    if intent.task_type in (TaskType.DEBUG, TaskType.EDIT, TaskType.REFACTOR):
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
                                f"Blast radius (depth {dist})", target_file,
                            ),
                            reason=f"Blast radius (depth {dist})",
                        )

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

            affected = store.graph.blast_radius(fqn, max_depth=1)
            for affected_fqn, dist in affected.items():
                if affected_fqn not in candidates:
                    sk = store.skeleton_table.get(affected_fqn)
                    if sk:
                        candidates[affected_fqn] = RankedCandidate(
                            skeleton=sk, tier=Tier.TIER2, distance=dist,
                            score=ranker.score(
                                affected_fqn, sk, dist,
                                f"Explaining callers (depth {dist})",
                                target_file,
                            ),
                            reason=f"Explaining callers (depth {dist})",
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
        # Python/TS/JS instance constructors
        for suffix in (".__init__", ".constructor"):
            ctor_fqn = class_fqn + suffix
            if ctor_fqn in store.skeleton_table and ctor_fqn not in candidates:
                sk = store.skeleton_table[ctor_fqn]
                candidates[ctor_fqn] = RankedCandidate(
                    skeleton=sk, tier=Tier.TIER2, distance=0,
                    score=ranker.score(
                        ctor_fqn, sk, 0, "Auto-included constructor", target_file,
                    ),
                    reason="Auto-included constructor",
                )

        # Go convention: NewStructName() or newStructName() standalone functions
        # Rust convention: StructName::new()
        class_name = class_fqn.split("::")[-1].split(".")[-1]
        go_new1 = f"New{class_name}"
        go_new2 = f"new{class_name}"
        file_path = class_fqn.split("::")[0]
        
        for name in (go_new1, go_new2, "new"):
            # Rust uses file::Struct.new or file::new
            # Go uses file::NewStruct
            for possible_fqn in (f"{file_path}::{name}", f"{class_fqn}.{name}"):
                if possible_fqn in store.skeleton_table and possible_fqn not in candidates:
                    sk = store.skeleton_table[possible_fqn]
                    candidates[possible_fqn] = RankedCandidate(
                        skeleton=sk, tier=Tier.TIER2, distance=0,
                        score=ranker.score(
                            possible_fqn, sk, 0, "Auto-included constructor", target_file,
                        ),
                        reason="Auto-included constructor",
                    )


def _has_anaphora(prompt: str) -> bool:
    """Check if prompt contains anaphoric references (it, that, this, etc.).

    Only matches standalone references, not words like 'with', 'within',
    'itself'. Checks that the word appears at word boundaries.
    """
    import re
    # Match standalone anaphoric words at word boundaries
    anaphora_pattern = re.compile(r'\b(it|that|this|those|the same)\b', re.IGNORECASE)
    matches = anaphora_pattern.findall(prompt)
    if not matches:
        return False
    # Filter out "it" when part of common phrases like "Fix it" at sentence start
    # Only consider anaphora if the word appears as a subject/reference
    # Simple heuristic: "it" at the start of a sentence or after "fix/debug/update"
    prompt_lower = prompt.lower().strip()
    # If entire prompt is like "fix it" or "debug this", that's anaphora
    if re.match(r'^(fix|debug|update|refactor|explain|test)\s+(it|this|that)\b', prompt_lower):
        return True
    # If "it" appears mid-sentence as a subject, likely anaphora
    if re.search(r'\b(it|this|that)\s+(is|was|has|should|doesn|does|needs|fails)', prompt_lower):
        return True
    return False


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
