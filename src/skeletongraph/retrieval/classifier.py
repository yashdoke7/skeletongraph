"""
Query classifier and mode selector — v4 (12-mode taxonomy).

Takes an Intent (from intent.py) and/or SLM result and maps to:
  - QueryMode (12 modes: RETRIEVAL_FAST, DEBUG_TARGETED, DEBUG_INVESTIGATE,
    BUILD_GUIDED, BUILD_GREENFIELD, REFACTOR, EXPLAIN, ARCHITECTURE,
    REVIEW, TEST, DOCUMENT, MIGRATE)

Each mode carries:
  - Token budget
  - Model tier (SLM / MLM / LLM)
  - Graph direction and depth
  - Which layers to load
  - Which modifiers to apply

v4 changes:
  - Replaces 7 QueryTypes + 5 ContextModes with 12 unified QueryModes
  - Per-mode spec dataclass with all retrieval parameters
  - Supports SLM mode override (SLM wins if disagreement with local classifier)
  - Hybrid detection: hard signals → MiniLM → SLM fallback
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .confidence import ConfidenceScore
    from .intent import Intent

from .intent import TaskType


# ── 12-Mode Enum ────────────────────────────────────────────────────────


class QueryMode(Enum):
    """12 unified query modes for v4."""
    RETRIEVAL_FAST = "retrieval_fast"        # Quick lookup: "where is X?"
    DEBUG_TARGETED = "debug_targeted"        # Fix specific bug with known entity
    DEBUG_INVESTIGATE = "debug_investigate"  # Diagnose issue from description
    BUILD_GUIDED = "build_guided"            # Add feature to existing code
    BUILD_GREENFIELD = "build_greenfield"    # Create new module/feature
    REFACTOR = "refactor"                    # Restructure without behavior change
    EXPLAIN = "explain"                      # Understand how code works
    ARCHITECTURE = "architecture"            # System design, high-level planning
    REVIEW = "review"                        # Code review, audit
    TEST = "test"                            # Write tests, add coverage
    DOCUMENT = "document"                    # Write docs, docstrings
    MIGRATE = "migrate"                      # Upgrade APIs, replace deprecated code


class ModelTier(Enum):
    """Which model tier handles this mode's execution."""
    SLM = "slm"    # SLM handles directly (no main LLM needed)
    MLM = "mlm"    # Standard coding model (Sonnet, GPT-5.5, etc.)
    LLM = "llm"    # Strongest reasoning model (Opus, GPT-5.5-o1, etc.)


# ── Per-Mode Specification ──────────────────────────────────────────────


@dataclass(frozen=True)
class ModeSpec:
    """Complete specification for a query mode.

    Defines everything the pipeline needs: budget, layers, graph traversal,
    model tier, and modifiers. No mode-specific if/else elsewhere.
    """
    mode: QueryMode
    tier: ModelTier
    budget: int                                     # Max output tokens for assembly
    # Graph expansion
    graph_direction: str = "none"                   # "none" | "forward" | "reverse" | "both"
    blast_depth: int = 0                            # Reverse BFS depth (callers)
    dep_depth: int = 0                              # Forward BFS depth (deps)
    # Layer loading flags
    load_project: bool = True                       # project.md
    load_architecture: bool = False                 # architecture.md
    load_constraints: bool = False                  # constraints.md (scoped)
    load_current_session: bool = True               # session/current.md
    load_recent_session: bool = False               # session/recent.md
    load_project_log: bool = False                  # session/project_log.md
    load_domain: bool = False                       # domain/*.md
    load_target_bodies: bool = True                 # Full function bodies (Tier 1)
    load_neighbor_sigs: bool = True                 # Neighbor signatures + summaries (Tier 2)
    load_tests: bool = False                        # Test function references
    test_detail: str = "names"                      # "names" | "bodies" | "patterns"
    # Modifiers (applied to main LLM prompt)
    modifiers: List[str] = field(default_factory=list)


# ── Mode Specs Table ────────────────────────────────────────────────────


MODE_SPECS: Dict[QueryMode, ModeSpec] = {
    QueryMode.RETRIEVAL_FAST: ModeSpec(
        mode=QueryMode.RETRIEVAL_FAST,
        tier=ModelTier.SLM,
        budget=400,
        graph_direction="none",
        load_target_bodies=False,
        load_neighbor_sigs=False,
        load_current_session=False,
    ),
    QueryMode.DEBUG_TARGETED: ModeSpec(
        mode=QueryMode.DEBUG_TARGETED,
        tier=ModelTier.MLM,
        budget=1500,
        graph_direction="both",
        blast_depth=1,
        dep_depth=1,
        load_constraints=True,
        load_tests=True,
        test_detail="names",
        modifiers=["MINIMAL"],
    ),
    QueryMode.DEBUG_INVESTIGATE: ModeSpec(
        mode=QueryMode.DEBUG_INVESTIGATE,
        tier=ModelTier.MLM,
        budget=4000,
        graph_direction="both",
        blast_depth=2,
        dep_depth=2,
        load_constraints=True,
        load_recent_session=True,
        load_domain=True,
        load_tests=True,
        test_detail="bodies",
        modifiers=["VERIFY_ASSUMPTIONS"],
    ),
    QueryMode.BUILD_GUIDED: ModeSpec(
        mode=QueryMode.BUILD_GUIDED,
        tier=ModelTier.MLM,
        budget=3500,
        graph_direction="forward",
        dep_depth=1,
        load_target_bodies=False,  # Show outlines, not bodies
        load_architecture=True,
        load_constraints=True,
        load_recent_session=True,
        load_domain=True,
        modifiers=["STEP_COMMIT"],
    ),
    QueryMode.BUILD_GREENFIELD: ModeSpec(
        mode=QueryMode.BUILD_GREENFIELD,
        tier=ModelTier.LLM,
        budget=5000,
        graph_direction="none",
        load_target_bodies=False,
        load_architecture=True,
        load_recent_session=True,
        load_project_log=True,
        load_domain=True,
        modifiers=["BRAINSTORM", "STEP_COMMIT"],
    ),
    QueryMode.REFACTOR: ModeSpec(
        mode=QueryMode.REFACTOR,
        tier=ModelTier.MLM,
        budget=5000,
        graph_direction="reverse",
        blast_depth=3,
        load_constraints=True,
        load_tests=True,
        test_detail="names",
        modifiers=["BLAST_FIRST"],
    ),
    QueryMode.EXPLAIN: ModeSpec(
        mode=QueryMode.EXPLAIN,
        tier=ModelTier.MLM,
        budget=3000,
        graph_direction="forward",
        dep_depth=3,
        load_architecture=True,
        load_domain=True,
        modifiers=["THINK_ALOUD"],
    ),
    QueryMode.ARCHITECTURE: ModeSpec(
        mode=QueryMode.ARCHITECTURE,
        tier=ModelTier.LLM,
        budget=6000,
        graph_direction="none",
        load_target_bodies=False,
        load_neighbor_sigs=True,  # Module interfaces
        load_architecture=True,
        load_recent_session=True,
        load_project_log=True,
        load_domain=True,
        modifiers=["BRAINSTORM"],
    ),
    QueryMode.REVIEW: ModeSpec(
        mode=QueryMode.REVIEW,
        tier=ModelTier.MLM,
        budget=4000,
        graph_direction="reverse",
        blast_depth=2,
        load_constraints=True,
        load_tests=True,
        test_detail="names",
        modifiers=["BLAST_FIRST"],
    ),
    QueryMode.TEST: ModeSpec(
        mode=QueryMode.TEST,
        tier=ModelTier.MLM,
        budget=3000,
        graph_direction="forward",
        dep_depth=1,
        load_constraints=True,
        load_tests=True,
        test_detail="patterns",
        modifiers=[],
    ),
    QueryMode.DOCUMENT: ModeSpec(
        mode=QueryMode.DOCUMENT,
        tier=ModelTier.SLM,
        budget=1500,
        graph_direction="reverse",
        blast_depth=1,
        load_tests=False,
        modifiers=[],
    ),
    QueryMode.MIGRATE: ModeSpec(
        mode=QueryMode.MIGRATE,
        tier=ModelTier.MLM,
        budget=6000,
        graph_direction="reverse",
        blast_depth=3,
        load_constraints=True,
        load_recent_session=True,
        load_tests=True,
        test_detail="names",
        modifiers=["BLAST_FIRST", "STEP_COMMIT"],
    ),
}


# ── Mode Budgets (for backward compatibility) ───────────────────────────

MODE_BUDGETS = {mode: spec.budget for mode, spec in MODE_SPECS.items()}


# ── Backward-Compatible Aliases ─────────────────────────────────────────

# v3 code uses these — keep them as aliases pointing to closest v4 mode
class ContextMode(Enum):
    """Legacy 5-mode system. Each maps to a v4 QueryMode."""
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
    PLANNING = "planning"
    REVIEW = "review"
    PASS_THROUGH = "pass_through"


# Mapping from legacy ContextMode to v4 QueryMode
_LEGACY_MODE_MAP = {
    ContextMode.FAST: QueryMode.DEBUG_TARGETED,
    ContextMode.STANDARD: QueryMode.BUILD_GUIDED,
    ContextMode.DEEP: QueryMode.DEBUG_INVESTIGATE,
    ContextMode.PLANNING: QueryMode.ARCHITECTURE,
    ContextMode.REVIEW: QueryMode.REVIEW,
    ContextMode.PASS_THROUGH: QueryMode.RETRIEVAL_FAST,
}


# Also keep legacy QueryType for gradual migration
class QueryType(Enum):
    """Legacy 7-type classification. Maps to v4 QueryModes."""
    CODE_FIX = "code_fix"
    NEW_FEATURE = "new_feature"
    REFACTOR = "refactor"
    DEBUG_INVESTIGATE = "debug_investigate"
    PLANNING = "planning"
    SUMMARY = "summary"
    GENERAL = "general"


# ── Classification Result ───────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Output of the classifier."""
    query_type: QueryType          # Legacy field for backward compat
    mode: ContextMode              # Legacy field for backward compat
    query_mode: QueryMode = QueryMode.BUILD_GUIDED  # v4 primary field
    mode_spec: Optional[ModeSpec] = None
    modifiers: List[str] = field(default_factory=list)
    extended_thinking: bool = False
    reason: str = ""


# ── Hard Signals (unambiguous, no ML needed) ────────────────────────────

_HARD_SIGNALS = {
    # Git diff present → REVIEW
    "review_signals": frozenset({
        "diff --git", "@@", "review this", "code review", "pr review",
        "look at my changes",
    }),
    # Migration signals
    "migrate_signals": frozenset({
        "migrate", "deprecated", "replace all", "upgrade",
        "breaking change", "switch from", "move from",
    }),
    # Architecture signals (no specific entity)
    "architecture_signals": frozenset({
        "architecture", "system design", "how should we design",
        "component design", "module design",
    }),
    # Test signals
    "test_signals": frozenset({
        "write test", "add test", "test coverage", "add spec",
        "unit test", "integration test",
    }),
    # Document signals
    "document_signals": frozenset({
        "docstring", "add documentation", "write docs",
        "api documentation", "jsdoc", "type hints",
    }),
    # Navigation signals
    "navigation_signals": frozenset({
        "where is", "find", "locate", "show me where",
        "which file", "where does",
    }),
}


# ── Main Classifier ─────────────────────────────────────────────────────


def classify_query(
    intent: "Intent",
    confidence: Optional["ConfidenceScore"] = None,
    target_fqns: Optional[Set[str]] = None,
    n_files_involved: int = 0,
    intent_override: Optional[str] = None,
) -> ClassificationResult:
    """Classify a query into QueryMode + spec + modifiers.

    Classification order:
    1. Hard signals (unambiguous keywords/patterns)
    2. Explicit intent override (caller-declared, wins over all inference)
    3. TaskType-based mapping (from intent analysis)

    Args:
        intent: Parsed intent from analyze_intent().
        confidence: Optional 5-factor confidence score.
        target_fqns: Set of FQNs found by resolver.
        n_files_involved: Number of files touched by targets + neighbors.

    Returns:
        ClassificationResult with query_mode, mode_spec, and modifiers.
    """
    prompt = intent.raw_prompt.lower()
    has_entities = bool(intent.entities)
    conf_level = confidence.level() if confidence else "MEDIUM"

    # ── Step 0: explicit intent override ─────────────────────────────
    # Pull-model: the caller (the main model in an IDE, or the SLM in CLI
    # mode) declares intent as a tool argument. An explicit, valid override
    # wins over all inference — no extra classification call needed.
    query_mode = _parse_slm_mode(intent_override) if intent_override else None

    # ── Step 1: Hard signal detection ────────────────────────────────
    if query_mode is None:
        query_mode = _detect_hard_signal(prompt, has_entities)

    # ── Step 2: TaskType-based fallback ──────────────────────────────
    if query_mode is None:
        query_mode = _map_from_task_type(intent, has_entities, conf_level)

    # ── Step 4: Resolve spec and modifiers ───────────────────────────
    mode_spec = MODE_SPECS[query_mode]

    # Override modifiers with confidence-based adjustments
    modifiers = list(mode_spec.modifiers)
    if conf_level == "LOW" and "THINK_ALOUD" not in modifiers:
        modifiers.append("THINK_ALOUD")
    if any(sig in prompt for sig in ("quick", "quickly", "just", "simple")):
        modifiers = ["MINIMAL"]

    # Cap at 2
    modifiers = modifiers[:2]

    # Extended thinking for complex tasks
    extended_thinking = False
    if confidence and query_mode in (QueryMode.ARCHITECTURE, QueryMode.DEBUG_INVESTIGATE, QueryMode.MIGRATE):
        if hasattr(confidence, "cross_file") and confidence.cross_file < 0.3:
            extended_thinking = True

    # Map to legacy types for backward compat
    legacy_query_type = _to_legacy_query_type(query_mode)
    legacy_mode = _to_legacy_context_mode(query_mode, conf_level)

    reason = (
        f"{query_mode.value} → tier={mode_spec.tier.value}"
        f" (budget={mode_spec.budget}, confidence={conf_level})"
        f" modifiers={modifiers}"
    )

    return ClassificationResult(
        query_type=legacy_query_type,
        mode=legacy_mode,
        query_mode=query_mode,
        mode_spec=mode_spec,
        modifiers=modifiers,
        extended_thinking=extended_thinking,
        reason=reason,
    )


# ── Detection Functions ─────────────────────────────────────────────────


def _detect_hard_signal(prompt: str, has_entities: bool) -> Optional[QueryMode]:
    """Detect unambiguous mode from hard signals. Returns None if ambiguous."""

    # Git diff → REVIEW (strongest signal)
    if any(sig in prompt for sig in _HARD_SIGNALS["review_signals"]):
        return QueryMode.REVIEW

    # Migration
    if any(sig in prompt for sig in _HARD_SIGNALS["migrate_signals"]):
        return QueryMode.MIGRATE

    # Architecture (only if no specific entity)
    if not has_entities and any(sig in prompt for sig in _HARD_SIGNALS["architecture_signals"]):
        return QueryMode.ARCHITECTURE

    # Test
    if any(sig in prompt for sig in _HARD_SIGNALS["test_signals"]):
        return QueryMode.TEST

    # Document
    if any(sig in prompt for sig in _HARD_SIGNALS["document_signals"]):
        return QueryMode.DOCUMENT

    # Navigation
    if any(sig in prompt for sig in _HARD_SIGNALS["navigation_signals"]):
        return QueryMode.RETRIEVAL_FAST

    return None


def _parse_slm_mode(mode_str: str) -> Optional[QueryMode]:
    """Parse the SLM's mode string into a QueryMode enum."""
    mode_str = mode_str.lower().strip()
    for qm in QueryMode:
        if qm.value == mode_str:
            return qm
    return None


def _map_from_task_type(
    intent: "Intent",
    has_entities: bool,
    conf_level: str,
) -> QueryMode:
    """Map TaskType → QueryMode (fallback when no hard signal or SLM)."""
    task = intent.task_type
    prompt = intent.raw_prompt.lower()

    if task == TaskType.DEBUG:
        if has_entities:
            return QueryMode.DEBUG_TARGETED
        return QueryMode.DEBUG_INVESTIGATE

    if task == TaskType.CREATE:
        if has_entities:
            return QueryMode.BUILD_GUIDED
        return QueryMode.BUILD_GREENFIELD

    if task == TaskType.EDIT:
        if has_entities:
            return QueryMode.DEBUG_TARGETED  # Edit with entity = targeted fix
        return QueryMode.BUILD_GUIDED

    if task == TaskType.REFACTOR:
        return QueryMode.REFACTOR

    if task == TaskType.EXPLAIN:
        if has_entities:
            return QueryMode.EXPLAIN
        # No entity + explain = might be architecture question
        if any(w in prompt for w in ("architecture", "design", "system")):
            return QueryMode.ARCHITECTURE
        return QueryMode.EXPLAIN

    if task == TaskType.REVIEW:
        return QueryMode.REVIEW

    if task == TaskType.TEST:
        return QueryMode.TEST

    if task == TaskType.DOCUMENT:
        return QueryMode.DOCUMENT

    if task == TaskType.MIGRATE:
        return QueryMode.MIGRATE

    if task == TaskType.ARCHITECTURE:
        return QueryMode.ARCHITECTURE

    # Default: if entities found, debug it; otherwise, build
    if has_entities:
        return QueryMode.DEBUG_TARGETED
    return QueryMode.BUILD_GUIDED


# ── Legacy Mapping ──────────────────────────────────────────────────────


def _to_legacy_query_type(mode: QueryMode) -> QueryType:
    """Map v4 QueryMode → legacy QueryType for backward compat."""
    mapping = {
        QueryMode.RETRIEVAL_FAST: QueryType.GENERAL,
        QueryMode.DEBUG_TARGETED: QueryType.CODE_FIX,
        QueryMode.DEBUG_INVESTIGATE: QueryType.DEBUG_INVESTIGATE,
        QueryMode.BUILD_GUIDED: QueryType.NEW_FEATURE,
        QueryMode.BUILD_GREENFIELD: QueryType.NEW_FEATURE,
        QueryMode.REFACTOR: QueryType.REFACTOR,
        QueryMode.EXPLAIN: QueryType.DEBUG_INVESTIGATE,
        QueryMode.ARCHITECTURE: QueryType.PLANNING,
        QueryMode.REVIEW: QueryType.SUMMARY,
        QueryMode.TEST: QueryType.NEW_FEATURE,
        QueryMode.DOCUMENT: QueryType.GENERAL,
        QueryMode.MIGRATE: QueryType.REFACTOR,
    }
    return mapping.get(mode, QueryType.GENERAL)


def _to_legacy_context_mode(mode: QueryMode, conf_level: str) -> ContextMode:
    """Map v4 QueryMode → legacy ContextMode for backward compat."""
    if conf_level == "MISS":
        return ContextMode.PASS_THROUGH

    mapping = {
        QueryMode.RETRIEVAL_FAST: ContextMode.FAST,
        QueryMode.DEBUG_TARGETED: ContextMode.FAST,
        QueryMode.DEBUG_INVESTIGATE: ContextMode.STANDARD,
        QueryMode.BUILD_GUIDED: ContextMode.STANDARD,
        QueryMode.BUILD_GREENFIELD: ContextMode.DEEP,
        QueryMode.REFACTOR: ContextMode.STANDARD,
        QueryMode.EXPLAIN: ContextMode.STANDARD,
        QueryMode.ARCHITECTURE: ContextMode.PLANNING,
        QueryMode.REVIEW: ContextMode.REVIEW,
        QueryMode.TEST: ContextMode.STANDARD,
        QueryMode.DOCUMENT: ContextMode.FAST,
        QueryMode.MIGRATE: ContextMode.DEEP,
    }
    return mapping.get(mode, ContextMode.STANDARD)
