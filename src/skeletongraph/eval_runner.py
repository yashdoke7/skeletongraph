"""
Evaluation runner — runs the golden dataset against SkeletonGraph.

Provides both a programmatic API and CLI integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .build import build_index
from .retrieval.resolver import resolve_context
from .assembly.zone_assembler import assemble_context


@dataclass
class EvalCase:
    """A single evaluation case."""
    task_id: str
    prompt: str
    expected_fqns: List[str]    # FQNs that should appear in context
    task_type: str = ""
    description: str = ""


@dataclass
class EvalResult:
    """Result of running a single eval case."""
    task_id: str
    prompt: str
    token_count: int
    reduction_ratio: float
    confidence: str
    expected_fqns: List[str]
    found_fqns: List[str]
    missing_fqns: List[str]
    precision: float
    recall: float
    mrr: float
    success: bool              # All expected FQNs found


@dataclass
class EvalSummary:
    """Summary of all evaluation cases."""
    total_cases: int
    success_count: int
    success_rate: float
    avg_reduction_ratio: float
    avg_tokens: float
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    results: List[EvalResult]
    duration_seconds: float


# Built-in eval cases for python_small fixture
_BUILTIN_CASES = [
    EvalCase(
        task_id="auth-01",
        prompt="fix validate_token in auth/middleware.py",
        expected_fqns=[
            "auth/middleware.py::validate_token",
            "auth/middleware.py::decode_jwt",
        ],
        task_type="debug",
        description="Debug JWT validation — should find validate_token and its dep decode_jwt",
    ),
    EvalCase(
        task_id="auth-02",
        prompt="explain how AuthMiddleware works",
        expected_fqns=[
            "auth/middleware.py::AuthMiddleware",
            "auth/middleware.py::AuthMiddleware.__init__",
        ],
        task_type="explain",
        description="Explain class — should find class and constructor",
    ),
    EvalCase(
        task_id="auth-03",
        prompt="add rate limiting to the auth middleware",
        expected_fqns=[
            "auth/middleware.py::AuthMiddleware",
        ],
        task_type="create",
        description="Create feature — should find existing middleware for context",
    ),
    EvalCase(
        task_id="auth-04",
        prompt="refactor decode_jwt to support multiple algorithms",
        expected_fqns=[
            "auth/middleware.py::decode_jwt",
            "auth/middleware.py::validate_token",
        ],
        task_type="refactor",
        description="Refactor — should find target and its caller (blast radius)",
    ),
    EvalCase(
        task_id="auth-05",
        prompt="what does get_user do and what calls it",
        expected_fqns=[
            "auth/middleware.py::get_user",
            "auth/middleware.py::validate_token",
        ],
        task_type="explain",
        description="Explain with callers — should find function and blast radius",
    ),
]


def run_evaluation(
    project_root: Path,
    cases: Optional[List[EvalCase]] = None,
    fixture_path: Optional[str] = None,
) -> EvalSummary:
    """Run evaluation cases against the index.

    Args:
        project_root: Root of the project to evaluate against.
        cases: Optional list of eval cases. Uses built-in cases if None.
        fixture_path: Optional path to fixture directory within project_root.

    Returns:
        EvalSummary with all results.
    """
    start = time.time()
    eval_cases = cases or _BUILTIN_CASES

    # Build index
    target = project_root
    if fixture_path:
        target = project_root / fixture_path

    store = build_index(target)

    results: List[EvalResult] = []
    success_count = 0

    for case in eval_cases:
        result = resolve_context(case.prompt, store)
        assembled = assemble_context(result, store, target)

        # Check which expected FQNs are in the candidates
        returned_fqns = {c.skeleton.fqn for c in result.candidates}
        found = [fqn for fqn in case.expected_fqns if fqn in returned_fqns]
        missing = [fqn for fqn in case.expected_fqns if fqn not in returned_fqns]

        mrr = 0.0
        for i, cand in enumerate(result.candidates):
            if cand.skeleton.fqn in case.expected_fqns:
                mrr = 1.0 / (i + 1)
                break

        precision = len(found) / max(len(returned_fqns), 1)
        recall = len(found) / max(len(case.expected_fqns), 1)
        success = len(missing) == 0

        if success:
            success_count += 1

        results.append(EvalResult(
            task_id=case.task_id,
            prompt=case.prompt,
            token_count=assembled.token_count,
            reduction_ratio=assembled.reduction_ratio,
            confidence=assembled.confidence,
            expected_fqns=case.expected_fqns,
            found_fqns=found,
            missing_fqns=missing,
            precision=precision,
            recall=recall,
            mrr=mrr,
            success=success,
        ))

    duration = time.time() - start
    total = len(results)

    return EvalSummary(
        total_cases=total,
        success_count=success_count,
        success_rate=success_count / max(total, 1),
        avg_reduction_ratio=sum(r.reduction_ratio for r in results) / max(total, 1),
        avg_tokens=sum(r.token_count for r in results) / max(total, 1),
        avg_precision=sum(r.precision for r in results) / max(total, 1),
        avg_recall=sum(r.recall for r in results) / max(total, 1),
        avg_mrr=sum(r.mrr for r in results) / max(total, 1),
        results=results,
        duration_seconds=duration,
    )
