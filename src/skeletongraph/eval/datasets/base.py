"""
Base schema for all evaluation datasets.

Every dataset loader (SWE-bench, CRG-compat, custom) must produce
a list of EvalTask objects conforming to this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class EvalTask:
    """A single evaluation task representing a coding problem to solve.
    
    Fields are designed to be a superset of SWE-bench Verified schema,
    CRG's YAML configs, and custom user-defined tasks.
    """
    # ── Identity ────────────────────────────────────────────────
    task_id: str                        # e.g. "django__django-16527"
    repo: str                           # e.g. "django/django"
    repo_url: str = ""                  # clone URL
    base_commit: str = "HEAD"           # checkout target (repo state before fix)
    
    # ── Problem ─────────────────────────────────────────────────
    problem_statement: str = ""         # GitHub issue text / task description
    
    # ── Ground Truth ────────────────────────────────────────────
    gold_patch: str = ""                # The human-authored solution diff
    ground_truth_files: List[str] = field(default_factory=list)  # Files modified in gold patch
    
    # ── Test Verification ───────────────────────────────────────
    fail_to_pass: List[str] = field(default_factory=list)  # Tests that must flip fail→pass
    pass_to_pass: List[str] = field(default_factory=list)  # Tests that must stay passing
    test_cmd: str = "pytest"            # Command to run tests
    test_patch: str = ""                # Patch containing test additions
    
    # ── Metadata ────────────────────────────────────────────────
    language: str = "python"
    size_category: str = ""             # "small", "medium", "large"
    description: str = ""               # Human-readable task summary
    source: str = ""                    # "swe-bench-verified", "crg-compat", "custom"
    version: str = ""                   # Repo version for env setup

    def __post_init__(self):
        """Extract ground truth files from gold patch if not provided."""
        if self.gold_patch and not self.ground_truth_files:
            self.ground_truth_files = extract_files_from_patch(self.gold_patch)


def extract_files_from_patch(patch: str) -> List[str]:
    """Extract modified file paths from a unified diff patch.
    
    Parses diff headers like:
        --- a/django/db/models/query.py
        +++ b/django/db/models/query.py
        diff --git a/foo.py b/foo.py
    
    Returns deduplicated list of file paths.
    """
    files = set()
    
    # Match "diff --git a/path b/path" headers
    for match in re.finditer(r'diff --git a/(.+?) b/(.+)', patch):
        files.add(match.group(2).strip())
    
    # Fallback: match "+++ b/path" headers
    if not files:
        for match in re.finditer(r'\+\+\+ b/(.+)', patch):
            path = match.group(1).strip()
            if path and path != '/dev/null':
                files.add(path)
    
    return sorted(files)


@dataclass
class EvalResult:
    """Result of running one benchmark on one task."""
    task_id: str
    benchmark: str                      # "token_efficiency", "retrieval_quality"
    metrics: dict = field(default_factory=dict)
    
    # Trace references (paths to the actual session logs used)
    sg_trace_path: str = ""
    native_trace_path: str = ""


@dataclass
class BenchmarkSummary:
    """Aggregated results across all tasks for one benchmark."""
    benchmark: str
    task_count: int
    results: List[EvalResult] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)  # mean, median, std, etc.
