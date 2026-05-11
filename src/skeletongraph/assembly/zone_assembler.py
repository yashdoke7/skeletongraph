"""Backward-compatible context assembler.

The v5 direction is to route all assembly through ``prompt_builder`` and the
unified engine. Some older CLI paths and tests still import ``assemble_context``
from this module, so this wrapper preserves the old surface while delegating to
the current assembler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..retrieval.classifier import classify_query
from ..retrieval.resolver import ResolverResult
from ..retrieval.session import Session
from ..storage.local import IndexStore
from .prompt_builder import assemble


@dataclass
class AttentionZone:
    """Minimal compatibility shape used by older callers."""

    zone_name: str
    token_count: int
    attention_level: str
    bar: str


@dataclass
class AssembledContext:
    """Compatibility result for the old zone assembler API."""

    text: str
    token_count: int
    confidence: str
    confidence_reason: str
    entities_matched: List[str] = field(default_factory=list)
    zone_breakdown: Dict[str, int] = field(default_factory=dict)
    attention_map: List[AttentionZone] = field(default_factory=list)
    reduction_ratio: float = 0.0
    session_dedup_count: int = 0
    session_tokens_saved: int = 0
    warning: str = ""


def assemble_context(
    resolver_result: ResolverResult,
    store: IndexStore,
    project_root: Path,
    model_context_limit: int = 128_000,
    detail_level: str = "compact",
    session: Optional[Session] = None,
    constraints: str = "",
    **_: Any,
) -> AssembledContext:
    """Assemble context using the current prompt builder.

    Args are intentionally compatible with the deleted v2/v3 zone assembler.
    ``model_context_limit`` and ``detail_level`` are accepted for old callers;
    budgeting is now controlled by the classifier/prompt builder.
    """

    n_files = len({c.skeleton.file_path for c in resolver_result.candidates})
    target_fqns = {c.skeleton.fqn for c in resolver_result.candidates}
    classification = classify_query(
        intent=resolver_result.intent,
        confidence=resolver_result.confidence_score,
        target_fqns=target_fqns,
        n_files_involved=n_files,
    )

    assembled = assemble(
        classification=classification,
        resolver_result=resolver_result,
        store=store,
        project_root=project_root,
        session=session,
    )

    text = assembled.text
    if constraints:
        text = f"## CONSTRAINTS\n{constraints.strip()}\n\n---\n\n{text}"
    if "TASK" not in text:
        text = f"## TASK\n{resolver_result.intent.raw_prompt}\n\n---\n\n{text}"

    zone_breakdown = {
        "zone1_constraints": _est_tokens(constraints) + assembled.layer_breakdown.get("L0_project", 0),
        "zone2_target_code": assembled.layer_breakdown.get("L4_target", 0),
        "zone3_context": (
            assembled.layer_breakdown.get("L4_callers", 0)
            + assembled.layer_breakdown.get("L4_blast", 0)
            + assembled.layer_breakdown.get("L1_architecture", 0)
        ),
        "zone4_prompt": _est_tokens(resolver_result.intent.raw_prompt),
    }

    token_count = _est_tokens(text)
    attention_map = [
        AttentionZone("zone1_constraints", zone_breakdown["zone1_constraints"], "peak", "████"),
        AttentionZone("zone2_target_code", zone_breakdown["zone2_target_code"], "high", "███"),
        AttentionZone("zone3_context", zone_breakdown["zone3_context"], "moderate", "██"),
        AttentionZone("zone4_prompt", zone_breakdown["zone4_prompt"], "peak", "████"),
    ]

    return AssembledContext(
        text=text,
        token_count=token_count,
        confidence=resolver_result.confidence,
        confidence_reason=resolver_result.confidence_reason,
        entities_matched=resolver_result.entities_matched,
        zone_breakdown=zone_breakdown,
        attention_map=attention_map,
        reduction_ratio=assembled.reduction_ratio,
        session_dedup_count=assembled.session_dedup_count,
        session_tokens_saved=0,
        warning=assembled.warning,
    )


def _est_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
