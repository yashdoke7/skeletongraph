"""
4-zone context assembly — adapted from HierMem's attention-aware architecture.

Zone placement optimizes LLM attention patterns:
  Zone 1 (top):    Constraints + output mode rules     → high attention (primacy)
  Zone 2 (upper):  Target code (full bodies)            → highest attention
  Zone 3 (middle): Structural context (skeletons/graph) → moderate attention
  Zone 4 (bottom): Current prompt + task                → high attention (recency)

Constraints and prompt at boundaries → never lost in the middle.
Target code near recency boundary → receives strong attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..retrieval.budget import Allocation, TokenBudget, Zone3Mode
from ..retrieval.resolver import RankedCandidate, ResolverResult, Tier
from ..storage.local import IndexStore
from ..summary.summary_store import SummaryStore
from ..parser.skeleton import SkeletonCore


@dataclass
class AssembledContext:
    """The final assembled context string ready for LLM consumption."""
    text: str
    token_count: int
    zone_breakdown: Dict[str, int]  # zone_name → token count
    confidence: str
    confidence_reason: str
    entities_matched: List[str]
    warning: str = ""
    reduction_ratio: float = 0.0  # vs raw file reading


def assemble_context(
    resolver_result: ResolverResult,
    store: IndexStore,
    project_root: Path,
    constraints: Optional[str] = None,
    model_context_limit: int = 128_000,
) -> AssembledContext:
    """Assemble the 4-zone context from resolver results.

    Args:
        resolver_result: Tiered candidates from the resolver.
        store: The loaded index.
        project_root: For reading full function bodies.
        constraints: Optional project-level constraints text.
        model_context_limit: Model's maximum context window.

    Returns:
        AssembledContext with assembled text and metadata.
    """
    candidates = resolver_result.candidates
    intent = resolver_result.intent

    # ── Build Zone 1: Constraints ──────────────────────────────────────
    zone1_parts = []
    if constraints:
        zone1_parts.append("=== PROJECT CONSTRAINTS ===")
        zone1_parts.append(constraints)

    zone1_text = "\n".join(zone1_parts) if zone1_parts else ""
    zone1_tokens = _estimate_tokens(zone1_text)

    # ── Build Zone 4: Prompt ───────────────────────────────────────────
    zone4_text = f"=== TASK ===\n{intent.raw_prompt}"
    zone4_tokens = _estimate_tokens(zone4_text)

    # ── Compute Zone 2: Target code bodies ─────────────────────────────
    tier1 = [c for c in candidates if c.tier == Tier.TIER1]
    zone2_parts = []
    zone2_tokens = 0

    for candidate in tier1:
        sk = candidate.skeleton
        body = _read_function_body(sk, project_root)
        if body:
            header = f"# {sk.file_display} — {sk.fqn.split('::')[-1]}"
            zone2_parts.append(f"{header}\n{body}")
            zone2_tokens += _estimate_tokens(body) + 5  # header overhead

    zone2_text = "\n\n".join(zone2_parts) if zone2_parts else ""

    # ── Budget allocation ──────────────────────────────────────────────
    budget = TokenBudget(model_context_limit)
    tier2_and_3 = [c for c in candidates if c.tier in (Tier.TIER2, Tier.TIER3)]
    allocation = budget.allocate(
        zone1_tokens=zone1_tokens,
        zone2_tokens=zone2_tokens,
        zone3_candidates_count=len(tier2_and_3),
        zone4_tokens=zone4_tokens,
    )

    # ── Build Zone 3: Structural context ───────────────────────────────
    zone3_parts = []
    zone3_tokens_used = 0

    if allocation.zone3_mode != Zone3Mode.NONE:
        # Get summaries for Tier 2 candidates
        tier2 = [c for c in candidates if c.tier == Tier.TIER2]
        tier3 = [c for c in candidates if c.tier == Tier.TIER3]

        summaries = store.summaries.batch_get(
            [c.skeleton.fqn for c in tier2]
        )

        # Add file structure header (which files are involved)
        involved_files: Set[str] = set()
        for c in candidates:
            involved_files.add(c.skeleton.file_path)

        if involved_files:
            zone3_parts.append("=== FILE STRUCTURE ===")
            for fp in sorted(involved_files):
                fs = store.file_skeletons.get(fp)
                if fs:
                    func_count = len(fs.all_skeletons)
                    zone3_parts.append(f"  {fp} ({func_count} functions)")

        # Add Tier 2 skeletons (signature + summary)
        if tier2 and allocation.zone3_mode in (Zone3Mode.FULL, Zone3Mode.COMPACT):
            zone3_parts.append("\n=== CONTEXT (neighbors) ===")
            for c in sorted(tier2, key=lambda x: -x.score):
                sk = c.skeleton
                summary = summaries.get(sk.fqn, "")
                if allocation.zone3_mode == Zone3Mode.FULL:
                    entry = sk.to_tier2_str(summary)
                else:
                    entry = sk.to_tier2_str("")  # No summary in compact mode

                entry_tokens = _estimate_tokens(entry)
                if zone3_tokens_used + entry_tokens > allocation.zone3_budget:
                    break
                zone3_parts.append(entry)
                zone3_tokens_used += entry_tokens

        # Add Tier 3 FQNs (minimal)
        if tier3 and allocation.zone3_mode in (Zone3Mode.FULL, Zone3Mode.COMPACT, Zone3Mode.MINIMAL):
            remaining = allocation.zone3_budget - zone3_tokens_used
            if remaining > 20:
                zone3_parts.append("\n=== PERIPHERY ===")
                for c in sorted(tier3, key=lambda x: -x.score):
                    entry = c.skeleton.to_tier3_str()
                    entry_tokens = _estimate_tokens(entry)
                    if zone3_tokens_used + entry_tokens > allocation.zone3_budget:
                        break
                    zone3_parts.append(f"  {entry}")
                    zone3_tokens_used += entry_tokens

        # Add edge summary
        if len(candidates) > 1:
            fqn_set = {c.skeleton.fqn for c in candidates}
            edge_summary = store.graph.edge_summary(fqn_set, max_edges=15)
            if edge_summary:
                edge_tokens = _estimate_tokens(edge_summary)
                if zone3_tokens_used + edge_tokens <= allocation.zone3_budget:
                    zone3_parts.append(f"\n=== RELATIONSHIPS ===\n{edge_summary}")
                    zone3_tokens_used += edge_tokens

    zone3_text = "\n".join(zone3_parts) if zone3_parts else ""

    # ── Final Assembly (Zone order: 1 → 3 → 2 → 4) ───────────────────
    # Note: Zone 2 is placed AFTER Zone 3 so target code is closer to
    # the prompt (Zone 4) at the bottom — stronger recency attention.
    sections = []
    if zone1_text:
        sections.append(zone1_text)
    if zone3_text:
        sections.append(zone3_text)
    if zone2_text:
        sections.append(f"=== TARGET CODE ===\n{zone2_text}")
    sections.append(zone4_text)

    assembled = "\n\n".join(sections)
    total_tokens = _estimate_tokens(assembled)

    # Estimate reduction ratio
    raw_tokens = _estimate_raw_reading_tokens(candidates, project_root)
    reduction = raw_tokens / max(total_tokens, 1) if raw_tokens > 0 else 0

    return AssembledContext(
        text=assembled,
        token_count=total_tokens,
        zone_breakdown={
            "zone1_constraints": zone1_tokens,
            "zone2_target_code": zone2_tokens,
            "zone3_structural": zone3_tokens_used,
            "zone4_prompt": zone4_tokens,
        },
        confidence=resolver_result.confidence,
        confidence_reason=resolver_result.confidence_reason,
        entities_matched=resolver_result.entities_matched,
        warning=allocation.warning,
        reduction_ratio=round(reduction, 1),
    )


def _read_function_body(sk: SkeletonCore, project_root: Path) -> str:
    """Read the full function body from disk using line numbers."""
    file_path = project_root / sk.file_path
    if not file_path.exists():
        return ""

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        # line_start and line_end are 1-indexed
        body_lines = lines[sk.line_start - 1:sk.line_end]
        return "\n".join(body_lines)
    except Exception:
        return ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(0, len(text) // 4)


def _estimate_raw_reading_tokens(
    candidates: List[RankedCandidate],
    project_root: Path,
) -> int:
    """Estimate tokens if the agent had read all involved files fully."""
    files_seen: Set[str] = set()
    total = 0
    for c in candidates:
        fp = c.skeleton.file_path
        if fp not in files_seen:
            files_seen.add(fp)
            full_path = project_root / fp
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    total += len(content) // 4
                except Exception:
                    pass
    return total
