"""
Query classifier and mode selector.

Takes an Intent (from intent.py) and maps it to:
  - QueryType (7 types: CODE_FIX, NEW_FEATURE, REFACTOR, DEBUG_INVESTIGATE, PLANNING, SUMMARY, GENERAL)
  - ContextMode (5 modes: FAST, STANDARD, DEEP, PLANNING, REVIEW + PASS_THROUGH)

Pure Python, no LLM, ~1ms. This is the routing brain of the pipeline.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .confidence import ConfidenceScore
    from .intent import Intent

from .intent import TaskType


class QueryType(Enum):
    """7-type query classification for mode routing."""
    CODE_FIX = "code_fix"                   # Fix a bug, resolve an error
    NEW_FEATURE = "new_feature"             # Add/implement something new
    REFACTOR = "refactor"                   # Restructure without changing behavior
    DEBUG_INVESTIGATE = "debug_investigate"  # Why/what's causing/investigate
    PLANNING = "planning"                   # How should/design/approach (no code target)
    SUMMARY = "summary"                     # What did we/progress/summarize
    GENERAL = "general"                     # Doesn't match above


class ContextMode(Enum):
    """5 context modes + pass-through."""
    FAST = "fast"               # ~950 tokens: L0 + L3(current) + L4(target + direct caller)
    STANDARD = "standard"       # ~3500 tokens: L0-L4, 1-hop
    DEEP = "deep"               # ~6500 tokens: L0-L4, 2-hop, full session
    PLANNING = "planning"       # ~1980 tokens: L0-L3, NO code bodies
    REVIEW = "review"           # ~1200 tokens: L0 + L3(all), NO code
    PASS_THROUGH = "pass_through"  # ~50 tokens: graph found nothing, get out of the way


# Token budget caps per mode
MODE_BUDGETS = {
    ContextMode.FAST: 1200,
    ContextMode.STANDARD: 4000,
    ContextMode.DEEP: 8000,
    ContextMode.PLANNING: 3500,
    ContextMode.REVIEW: 2000,
    ContextMode.PASS_THROUGH: 100,
}


@dataclass
class ClassificationResult:
    """Output of the classifier: query type + mode + modifiers."""
    query_type: QueryType
    mode: ContextMode
    modifiers: List[str] = field(default_factory=list)
    extended_thinking: bool = False
    reason: str = ""


# ── Hard-signal keywords for query type override ────────────────────────

_SUMMARY_SIGNALS = frozenset({
    "summarize", "summary", "what did we", "what have we",
    "progress", "what changed", "what's done", "what we built",
    "recap", "status update",
})

_PLANNING_SIGNALS = frozenset({
    "how should", "should we", "what's the best", "design",
    "approach", "plan", "architecture", "strategy",
    "propose", "recommend", "evaluate options",
})

_DEBUG_SIGNALS = frozenset({
    "why", "what's causing", "investigate", "trace",
    "slow", "not working", "root cause", "diagnose",
})

_MINIMAL_SIGNALS = frozenset({
    "quick", "quickly", "just", "small change", "minor",
    "tiny", "simple fix",
})


def classify_query(
    intent: "Intent",
    confidence: Optional["ConfidenceScore"] = None,
    target_fqns: Optional[Set[str]] = None,
    n_files_involved: int = 0,
) -> ClassificationResult:
    """Classify a query into QueryType + ContextMode + modifiers.

    Args:
        intent: Parsed intent from analyze_intent().
        confidence: Optional 5-factor confidence score.
        target_fqns: Set of FQNs found by resolver (for scope detection).
        n_files_involved: Number of files touched by targets + neighbors.

    Returns:
        ClassificationResult with query_type, mode, modifiers, and reason.
    """
    prompt = intent.raw_prompt.lower()
    has_entities = bool(intent.entities)
    conf_level = confidence.level() if confidence else "MEDIUM"
    conf_composite = confidence.composite() if confidence else 0.5

    # ── Step 1: Determine QueryType ──────────────────────────────────

    query_type = _classify_type(intent, prompt, has_entities)

    # ── Step 2: Determine ContextMode ────────────────────────────────

    mode = _select_mode(query_type, conf_level, has_entities, n_files_involved)

    # ── Step 3: Select Modifiers ─────────────────────────────────────

    modifiers = _select_modifiers(query_type, mode, conf_level, prompt, confidence)

    # ── Step 4: Extended Thinking flag ───────────────────────────────

    extended_thinking = False
    if confidence and query_type in (QueryType.PLANNING, QueryType.DEBUG_INVESTIGATE):
        cross_file = confidence.cross_file
        dep_depth = confidence.dependency_depth
        # Trigger: complex task with wide impact
        if cross_file < 0.3 or dep_depth < 0.4:  # many files or deep deps
            extended_thinking = True

    reason = (
        f"{query_type.value} → {mode.value}"
        f" (confidence={conf_level}, composite={conf_composite:.2f})"
        f" modifiers={modifiers}"
    )

    return ClassificationResult(
        query_type=query_type,
        mode=mode,
        modifiers=modifiers,
        extended_thinking=extended_thinking,
        reason=reason,
    )


def _classify_type(intent: "Intent", prompt: str, has_entities: bool) -> QueryType:
    """Map Intent → QueryType using hard signals first, then TaskType fallback."""

    # Hard signal 1: Summary queries (override everything)
    if any(sig in prompt for sig in _SUMMARY_SIGNALS):
        return QueryType.SUMMARY

    # Hard signal 2: Planning queries with no code target
    if any(sig in prompt for sig in _PLANNING_SIGNALS):
        if not has_entities:
            return QueryType.PLANNING

    # Hard signal 3: Debug/investigate with "why" even without entities
    if any(sig in prompt for sig in _DEBUG_SIGNALS) and not has_entities:
        return QueryType.DEBUG_INVESTIGATE

    # Map from existing TaskType
    task = intent.task_type

    if task == TaskType.DEBUG:
        # Distinguish CODE_FIX vs DEBUG_INVESTIGATE
        if has_entities:
            return QueryType.CODE_FIX
        if any(sig in prompt for sig in _DEBUG_SIGNALS):
            return QueryType.DEBUG_INVESTIGATE
        return QueryType.CODE_FIX  # Default: if it smells like debug, fix it

    if task == TaskType.EDIT:
        if has_entities:
            return QueryType.CODE_FIX
        return QueryType.GENERAL

    if task == TaskType.CREATE:
        # "add X to Y module" → NEW_FEATURE even without exact entity match
        # The resolver will do keyword/semantic search to find targets
        return QueryType.NEW_FEATURE

    if task == TaskType.REFACTOR:
        return QueryType.REFACTOR

    if task == TaskType.EXPLAIN:
        if has_entities:
            return QueryType.DEBUG_INVESTIGATE  # "how does X work" with a target
        return QueryType.PLANNING  # "how should we..." without a target

    if task == TaskType.REVIEW:
        # Check if this is a summary request vs actual code review
        if any(sig in prompt for sig in _SUMMARY_SIGNALS):
            return QueryType.SUMMARY
        if has_entities:
            return QueryType.CODE_FIX  # Review of specific code → treat as edit
        return QueryType.SUMMARY

    return QueryType.GENERAL


def _select_mode(
    query_type: QueryType,
    conf_level: str,
    has_entities: bool,
    n_files: int,
) -> ContextMode:
    """Select ContextMode from QueryType + confidence + scope."""

    # MISS → always pass through
    if conf_level == "MISS":
        return ContextMode.PASS_THROUGH

    if query_type == QueryType.CODE_FIX:
        if conf_level == "HIGH":
            return ContextMode.FAST
        return ContextMode.STANDARD

    if query_type == QueryType.NEW_FEATURE:
        if n_files > 3:
            return ContextMode.DEEP
        return ContextMode.STANDARD

    if query_type == QueryType.REFACTOR:
        if n_files > 3:
            return ContextMode.DEEP
        return ContextMode.STANDARD

    if query_type == QueryType.DEBUG_INVESTIGATE:
        return ContextMode.STANDARD

    if query_type == QueryType.PLANNING:
        return ContextMode.PLANNING

    if query_type == QueryType.SUMMARY:
        return ContextMode.REVIEW

    # GENERAL
    if has_entities:
        return ContextMode.STANDARD
    return ContextMode.PLANNING


def _select_modifiers(
    query_type: QueryType,
    mode: ContextMode,
    conf_level: str,
    prompt: str,
    confidence: Optional["ConfidenceScore"],
) -> List[str]:
    """Select reasoning modifiers (max 2 instruction-level)."""

    # MINIMAL overrides everything
    if any(sig in prompt for sig in _MINIMAL_SIGNALS):
        return ["MINIMAL"]

    modifiers: List[str] = []

    if query_type == QueryType.PLANNING and mode in (ContextMode.STANDARD, ContextMode.DEEP, ContextMode.PLANNING):
        modifiers.append("BRAINSTORM")

    if query_type == QueryType.REFACTOR:
        modifiers.append("BLAST_FIRST")

    if query_type == QueryType.DEBUG_INVESTIGATE:
        modifiers.append("VERIFY_ASSUMPTIONS")

    if query_type == QueryType.NEW_FEATURE and mode == ContextMode.DEEP:
        modifiers.append("STEP_COMMIT")

    if conf_level == "LOW":
        modifiers.append("THINK_ALOUD")

    # Cap at 2 instruction-level modifiers
    return modifiers[:2]
