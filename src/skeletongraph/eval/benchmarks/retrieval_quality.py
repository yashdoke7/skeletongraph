"""
Retrieval quality benchmark.

For each EvalTask, measures how well the agent's file retrieval
matches the ground-truth files from the human-authored PR patch.

Metrics computed:
  - Precision: % of retrieved files that were actually relevant
  - Recall: % of relevant files that were actually retrieved
  - F1: harmonic mean of precision and recall
  - MRR: Mean Reciprocal Rank (how early the first correct file appears)
  - Hit@k: binary — was at least one correct file in first k retrievals

Ground truth source: SWE-bench gold_patch diff headers.
Retrieved files source: Real agent session traces (tool_calls with view_file targets).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

from ..datasets.base import EvalTask
from ..scorer import RetrievalQualityScore
from ..schema import AgentTrace

logger = logging.getLogger(__name__)


def extract_retrieved_files(trace: AgentTrace) -> List[str]:
    """Extract the ordered list of files the agent retrieved during the session.
    
    Examines tool_calls for view_file, grep_search, query_context targets.
    Returns file paths in the order they were first accessed.
    """
    seen: set = set()
    ordered: List[str] = []
    
    for tc in trace.tool_calls:
        # Extract file path from the tool call target
        target = tc.target.strip()
        if not target:
            continue
        
        # Normalize: remove line ranges (file.py#L10-L20 -> file.py)
        if '#' in target:
            target = target.split('#')[0]
        
        # Normalize: convert backslashes to forward slashes
        target = target.replace('\\', '/')
        
        # Skip non-file targets (search queries, FQNs without file extensions)
        # A file target typically contains a dot for the extension
        basename = Path(target).name
        if '.' not in basename:
            # Could be a grep query or function name — skip
            continue
        
        # Normalize to relative path (strip leading "./" or absolute path prefixes)
        if target.startswith('./'):
            target = target[2:]
        
        # Deduplicate while preserving order
        if target not in seen:
            seen.add(target)
            ordered.append(target)
    
    return ordered


def normalize_path(path: str) -> str:
    """Normalize a file path for comparison.
    
    Strips leading repo prefixes, normalizes separators.
    """
    path = path.replace('\\', '/').strip()
    if path.startswith('./'):
        path = path[2:]
    # Remove common absolute path prefixes
    if '/' in path:
        # Keep only the relative path within the repo
        parts = path.split('/')
        # Find where the actual repo content starts
        # (skip things like /home/user/repos/django/)
        for i, part in enumerate(parts):
            if part in ('src', 'lib', 'tests', 'docs'):
                return '/'.join(parts[i:])
    return path


def match_files(retrieved: List[str], ground_truth: List[str]) -> tuple:
    """Match retrieved files against ground truth using flexible matching.
    
    Handles path normalization issues:
    - "django/db/models/query.py" matches "django/db/models/query.py"
    - "query.py" matches "django/db/models/query.py" (basename match)
    - Full paths match relative paths
    
    Returns (matched_retrieved, matched_truth, unmatched_retrieved, unmatched_truth)
    """
    # Build lookup sets
    truth_basenames = {Path(f).name: f for f in ground_truth}
    truth_paths = set(ground_truth)
    
    matched_r: List[str] = []
    matched_t: Set[str] = set()
    unmatched_r: List[str] = []
    
    for r in retrieved:
        r_norm = normalize_path(r)
        r_basename = Path(r).name
        
        # Try exact match first
        if r_norm in truth_paths:
            matched_r.append(r)
            matched_t.add(r_norm)
            continue
        
        # Try suffix match (retrieved path ends with truth path or vice versa)
        found = False
        for t in ground_truth:
            if t in matched_t:
                continue
            t_norm = normalize_path(t)
            if r_norm.endswith(t_norm) or t_norm.endswith(r_norm):
                matched_r.append(r)
                matched_t.add(t)
                found = True
                break
        
        if not found:
            # Try basename match as last resort
            if r_basename in truth_basenames and truth_basenames[r_basename] not in matched_t:
                matched_r.append(r)
                matched_t.add(truth_basenames[r_basename])
            else:
                unmatched_r.append(r)
    
    unmatched_t = [f for f in ground_truth if f not in matched_t]
    
    return matched_r, list(matched_t), unmatched_r, unmatched_t


def run_retrieval_quality(
    task: EvalTask,
    trace: AgentTrace,
) -> RetrievalQualityScore:
    """Run retrieval quality benchmark for a single task against one trace.
    
    Args:
        task: The evaluation task with ground_truth_files.
        trace: The agent trace (either SG or native).
        
    Returns:
        RetrievalQualityScore with P/R/F1/MRR computed.
    """
    ground_truth = task.ground_truth_files
    retrieved = extract_retrieved_files(trace)
    
    # Use flexible matching to handle path normalization
    matched_r, matched_t, _, _ = match_files(retrieved, ground_truth)
    
    # For the score, we need the retrieved list with paths normalized to match GT
    # So the MRR calculation uses the right ordering
    normalized_retrieved = []
    for r in retrieved:
        r_basename = Path(r).name
        r_norm = normalize_path(r)
        # Map to ground truth path if possible
        mapped = False
        for gt in ground_truth:
            gt_norm = normalize_path(gt)
            if r_norm.endswith(gt_norm) or gt_norm.endswith(r_norm) or Path(gt).name == r_basename:
                normalized_retrieved.append(gt)
                mapped = True
                break
        if not mapped:
            normalized_retrieved.append(r_norm)
    
    return RetrievalQualityScore(
        task_id=task.task_id,
        ground_truth_files=ground_truth,
        retrieved_files=normalized_retrieved,
    )


def run_retrieval_quality_batch(
    tasks: List[EvalTask],
    sg_traces: dict[str, AgentTrace],
    native_traces: dict[str, AgentTrace],
) -> dict[str, List[RetrievalQualityScore]]:
    """Run retrieval quality for all tasks, returning both SG and native scores.
    
    Returns:
        Dict with keys "sg" and "native", each mapping to list of scores.
    """
    sg_scores: List[RetrievalQualityScore] = []
    native_scores: List[RetrievalQualityScore] = []
    
    for task in tasks:
        if not task.ground_truth_files:
            logger.warning("Skipping %s: no ground truth files", task.task_id)
            continue
        
        sg_trace = sg_traces.get(task.task_id)
        native_trace = native_traces.get(task.task_id)
        
        if sg_trace:
            sg_score = run_retrieval_quality(task, sg_trace)
            sg_scores.append(sg_score)
            logger.info(
                "  %s [SG]: P=%.2f R=%.2f F1=%.2f MRR=%.2f",
                task.task_id, sg_score.precision, sg_score.recall,
                sg_score.f1, sg_score.mrr,
            )
        
        if native_trace:
            native_score = run_retrieval_quality(task, native_trace)
            native_scores.append(native_score)
            logger.info(
                "  %s [Native]: P=%.2f R=%.2f F1=%.2f MRR=%.2f",
                task.task_id, native_score.precision, native_score.recall,
                native_score.f1, native_score.mrr,
            )
    
    return {"sg": sg_scores, "native": native_scores}
