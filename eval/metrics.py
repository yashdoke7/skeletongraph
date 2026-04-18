"""
Evaluation metrics for SkeletonGraph context quality.

Measures:
  1. Token efficiency: tokens used vs raw file reading
  2. Constraint preservation: do constraints survive in output?
  3. Context coverage: are relevant functions included?
  4. Edge coverage: are dependency relationships captured?
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from skeletongraph.build import build_index
from skeletongraph.retrieval.resolver import resolve_context, Tier
from skeletongraph.assembly.zone_assembler import assemble_context
from skeletongraph.storage.local import IndexStore


@dataclass
class EvalCase:
    """A single evaluation case."""
    prompt: str
    expected_fqns: List[str]                # FQNs that MUST be in context
    expected_absent_fqns: List[str] = field(default_factory=list)  # Must NOT be included
    constraints: str = ""
    description: str = ""


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    case: EvalCase
    # Token metrics
    skeleton_tokens: int = 0
    raw_tokens: int = 0
    reduction_ratio: float = 0.0
    # Coverage metrics
    expected_found: int = 0
    expected_total: int = 0
    coverage_score: float = 0.0
    # False inclusion metrics
    false_inclusions: int = 0
    # Constraint preservation
    constraint_preserved: bool = True
    # Confidence
    confidence: str = "LOW"
    # Tier distribution
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    # Timing
    resolve_ms: float = 0.0
    assemble_ms: float = 0.0


@dataclass
class EvalSummary:
    """Aggregate evaluation results."""
    total_cases: int = 0
    avg_reduction_ratio: float = 0.0
    avg_coverage_score: float = 0.0
    avg_resolve_ms: float = 0.0
    constraint_preservation_rate: float = 0.0
    high_confidence_rate: float = 0.0
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "avg_reduction_ratio": round(self.avg_reduction_ratio, 2),
            "avg_coverage_score": round(self.avg_coverage_score, 3),
            "avg_resolve_ms": round(self.avg_resolve_ms, 2),
            "constraint_preservation_rate": round(self.constraint_preservation_rate, 3),
            "high_confidence_rate": round(self.high_confidence_rate, 3),
        }


def evaluate(
    project_root: Path,
    cases: List[EvalCase],
    store: Optional[IndexStore] = None,
    model_context_limit: int = 128_000,
) -> EvalSummary:
    """Run evaluation cases against a project.

    Args:
        project_root: Root of the project to evaluate against.
        cases: List of evaluation cases.
        store: Pre-built index (built fresh if None).
        model_context_limit: Context window size.

    Returns:
        EvalSummary with aggregate metrics.
    """
    if store is None:
        store = build_index(project_root)

    results = []

    for case in cases:
        result = _evaluate_case(case, store, project_root, model_context_limit)
        results.append(result)

    # Aggregate
    summary = EvalSummary(
        total_cases=len(results),
        results=results,
    )

    if results:
        summary.avg_reduction_ratio = sum(r.reduction_ratio for r in results) / len(results)
        summary.avg_coverage_score = sum(r.coverage_score for r in results) / len(results)
        summary.avg_resolve_ms = sum(r.resolve_ms for r in results) / len(results)
        summary.constraint_preservation_rate = (
            sum(1 for r in results if r.constraint_preserved) / len(results)
        )
        summary.high_confidence_rate = (
            sum(1 for r in results if r.confidence == "HIGH") / len(results)
        )

    return summary


def _evaluate_case(
    case: EvalCase,
    store: IndexStore,
    project_root: Path,
    model_context_limit: int,
) -> EvalResult:
    """Evaluate a single case."""
    result = EvalResult(case=case)

    # Resolve
    t0 = time.perf_counter()
    resolver_result = resolve_context(case.prompt, store)
    t1 = time.perf_counter()
    result.resolve_ms = (t1 - t0) * 1000

    # Assemble
    t2 = time.perf_counter()
    assembled = assemble_context(
        resolver_result, store, project_root,
        constraints=case.constraints,
        model_context_limit=model_context_limit,
    )
    t3 = time.perf_counter()
    result.assemble_ms = (t3 - t2) * 1000

    # Token metrics
    result.skeleton_tokens = assembled.token_count
    result.raw_tokens = _estimate_raw_tokens(resolver_result.candidates, project_root)
    result.reduction_ratio = (
        result.raw_tokens / max(result.skeleton_tokens, 1)
        if result.raw_tokens > 0 else 0
    )

    # Coverage: check which expected FQNs are in the candidate set
    candidate_fqns = {c.skeleton.fqn for c in resolver_result.candidates}
    result.expected_total = len(case.expected_fqns)
    result.expected_found = sum(
        1 for fqn in case.expected_fqns if fqn in candidate_fqns
    )
    result.coverage_score = (
        result.expected_found / result.expected_total
        if result.expected_total > 0 else 1.0
    )

    # False inclusions
    result.false_inclusions = sum(
        1 for fqn in case.expected_absent_fqns if fqn in candidate_fqns
    )

    # Constraint preservation
    if case.constraints:
        result.constraint_preserved = case.constraints in assembled.text

    # Confidence
    result.confidence = assembled.confidence

    # Tier distribution
    result.tier1_count = sum(1 for c in resolver_result.candidates if c.tier == Tier.TIER1)
    result.tier2_count = sum(1 for c in resolver_result.candidates if c.tier == Tier.TIER2)
    result.tier3_count = sum(1 for c in resolver_result.candidates if c.tier == Tier.TIER3)

    return result


def _estimate_raw_tokens(candidates, project_root: Path) -> int:
    """Estimate tokens if files were read in full."""
    files_seen: Set[str] = set()
    total = 0
    for c in candidates:
        fp = c.skeleton.file_path
        if fp not in files_seen:
            files_seen.add(fp)
            path = project_root / fp
            if path.exists():
                try:
                    total += len(path.read_text(encoding="utf-8", errors="replace")) // 4
                except Exception:
                    pass
    return total


def format_report(summary: EvalSummary) -> str:
    """Format evaluation results as a readable report."""
    lines = [
        "=" * 60,
        "  SkeletonGraph Evaluation Report",
        "=" * 60,
        f"  Cases:                     {summary.total_cases}",
        f"  Avg Token Reduction:       {summary.avg_reduction_ratio:.1f}x",
        f"  Avg Coverage Score:        {summary.avg_coverage_score:.1%}",
        f"  Constraint Preservation:   {summary.constraint_preservation_rate:.1%}",
        f"  High Confidence Rate:      {summary.high_confidence_rate:.1%}",
        f"  Avg Resolve Time:          {summary.avg_resolve_ms:.1f}ms",
        "=" * 60,
    ]

    for i, r in enumerate(summary.results):
        lines.append(f"\n  Case {i+1}: {r.case.description or r.case.prompt[:50]}")
        lines.append(f"    Coverage: {r.coverage_score:.0%} ({r.expected_found}/{r.expected_total})")
        lines.append(f"    Tokens:   {r.skeleton_tokens} (raw: {r.raw_tokens}, {r.reduction_ratio:.1f}x reduction)")
        lines.append(f"    Tiers:    T1={r.tier1_count}, T2={r.tier2_count}, T3={r.tier3_count}")
        lines.append(f"    Confidence: {r.confidence}")
        if r.false_inclusions:
            lines.append(f"    False inclusions: {r.false_inclusions}")

    return "\n".join(lines)
