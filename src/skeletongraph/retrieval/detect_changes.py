"""
PR/diff-aware blast radius analysis with risk scoring.

Parses git diff output, identifies changed functions, computes the
blast radius for each, and risk-scores affected files. Assembles
everything into a single 4-zone review context.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..storage.local import IndexStore
from ..graph.dependency import DependencyGraph


@dataclass
class ChangedFunction:
    """A function that was modified in a diff."""
    fqn: str
    file_path: str
    change_type: str  # "modified", "added", "deleted"
    lines_changed: int = 0


@dataclass
class AffectedFile:
    """A file affected by changes (via blast radius)."""
    file_path: str
    affected_fqns: List[str]
    risk_score: float          # 0.0 - 1.0
    risk_reason: str
    distance: int = 1          # Hops from changed code


@dataclass
class ChangeAnalysis:
    """Complete change impact analysis result."""
    changed_functions: List[ChangedFunction]
    affected_files: List[AffectedFile]
    total_blast_radius: int     # Total affected functions
    risk_summary: str
    files_to_review: List[str]  # Sorted by risk (highest first)


def detect_changes(
    project_root: Path,
    store: IndexStore,
    diff_target: str = "HEAD",
    max_depth: int = 2,
) -> ChangeAnalysis:
    """Analyze the blast radius of recent changes.

    Args:
        project_root: Project root directory.
        store: Loaded index.
        diff_target: Git diff target (e.g., "HEAD", "main", "HEAD~3").
        max_depth: Maximum blast-radius depth.

    Returns:
        ChangeAnalysis with risk-scored results.
    """
    # Get changed files from git
    changed_files = _get_git_diff_files(project_root, diff_target)
    if not changed_files:
        return ChangeAnalysis(
            changed_functions=[], affected_files=[],
            total_blast_radius=0,
            risk_summary="No changes detected.",
            files_to_review=[],
        )

    # Identify which functions were changed
    changed_fns: List[ChangedFunction] = []
    for file_path, change_type in changed_files.items():
        if file_path in store.file_skeletons:
            for sk in store.file_skeletons[file_path].all_skeletons:
                changed_fns.append(ChangedFunction(
                    fqn=sk.fqn,
                    file_path=file_path,
                    change_type=change_type,
                ))

    # Compute blast radius for each changed function
    all_affected: Dict[str, Tuple[int, str]] = {}  # fqn → (distance, reason)
    for fn in changed_fns:
        radius = store.graph.blast_radius(fn.fqn, max_depth=max_depth)
        for affected_fqn, dist in radius.items():
            if affected_fqn not in all_affected or all_affected[affected_fqn][0] > dist:
                all_affected[affected_fqn] = (dist, f"depends on {fn.fqn}")

    # Group by file and compute risk scores
    file_impact: Dict[str, List[Tuple[str, int, str]]] = {}
    for fqn, (dist, reason) in all_affected.items():
        sk = store.skeleton_table.get(fqn)
        if sk:
            fp = sk.file_path
            if fp not in file_impact:
                file_impact[fp] = []
            file_impact[fp].append((fqn, dist, reason))

    affected_files: List[AffectedFile] = []
    for fp, impacts in file_impact.items():
        risk = _compute_risk_score(fp, impacts, store)
        affected_files.append(AffectedFile(
            file_path=fp,
            affected_fqns=[fqn for fqn, _, _ in impacts],
            risk_score=risk,
            risk_reason=_risk_reason(risk),
            distance=min(d for _, d, _ in impacts),
        ))

    # Sort by risk
    affected_files.sort(key=lambda f: -f.risk_score)
    files_to_review = [f.file_path for f in affected_files]

    # Summary
    high_risk = sum(1 for f in affected_files if f.risk_score >= 0.7)
    med_risk = sum(1 for f in affected_files if 0.4 <= f.risk_score < 0.7)

    risk_summary = (
        f"{len(changed_fns)} functions changed → "
        f"{len(all_affected)} functions affected across "
        f"{len(affected_files)} files "
        f"({high_risk} high risk, {med_risk} medium risk)"
    )

    return ChangeAnalysis(
        changed_functions=changed_fns,
        affected_files=affected_files,
        total_blast_radius=len(all_affected),
        risk_summary=risk_summary,
        files_to_review=files_to_review,
    )


def _get_git_diff_files(
    project_root: Path, target: str
) -> Dict[str, str]:
    """Get changed files from git diff.

    Returns:
        Dict of relative_path → change_type ("modified", "added", "deleted").
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", target],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Try unstaged changes
            result = subprocess.run(
                ["git", "diff", "--name-status"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    files: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            status, path = parts
            change_map = {"M": "modified", "A": "added", "D": "deleted"}
            files[path] = change_map.get(status[0], "modified")

    return files


def _compute_risk_score(
    file_path: str,
    impacts: List[Tuple[str, int, str]],
    store: IndexStore,
) -> float:
    """Compute risk score for an affected file.

    Factors:
      - Number of affected functions in this file
      - Average complexity of affected functions
      - Proximity to changed code (distance)
      - Whether affected functions are exported (public API)
    """
    score = 0.0
    count = len(impacts)

    # More affected functions = higher risk
    score += min(count / 10.0, 0.3)  # Cap at 0.3

    # Proximity (closer = higher risk)
    min_distance = min(d for _, d, _ in impacts)
    score += max(0, (3 - min_distance)) * 0.15  # 0.15 per hop closer

    # Complexity
    total_complexity = 0
    exported_count = 0
    for fqn, _, _ in impacts:
        sk = store.skeleton_table.get(fqn)
        if sk:
            total_complexity += sk.complexity
            if sk.is_exported:
                exported_count += 1

    avg_complexity = total_complexity / max(count, 1)
    if avg_complexity > 5:
        score += 0.2

    # Exported functions = public API breakage risk
    if exported_count > 0:
        score += 0.15

    return min(score, 1.0)


def _risk_reason(score: float) -> str:
    if score >= 0.7:
        return "HIGH — multiple public functions affected at close proximity"
    elif score >= 0.4:
        return "MEDIUM — some functions affected, review recommended"
    else:
        return "LOW — peripheral impact only"
