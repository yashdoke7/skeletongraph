"""
Token efficiency benchmark.

For each EvalTask, measures:
  1. Native agent retrieval tokens (from real session logs)
  2. SkeletonGraph retrieval tokens (from SG session data)
  3. Whole codebase token ceiling
  4. All reduction ratios and cost estimates

This benchmark requires REAL agent session traces — 
it does not simulate or estimate what an agent "would" read.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..datasets.base import EvalTask
from ..scorer import TokenEfficiencyScore
from ..token_counter import measure_text_tokens, measure_codebase_tokens
from ..schema import AgentTrace

logger = logging.getLogger(__name__)


def run_token_efficiency(
    task: EvalTask,
    sg_trace: AgentTrace,
    native_trace: AgentTrace,
    repo_path: Optional[Path] = None,
) -> TokenEfficiencyScore:
    """Run token efficiency benchmark for a single task.
    
    Args:
        task: The evaluation task definition.
        sg_trace: Parsed trace from SG-augmented agent session.
        native_trace: Parsed trace from native agent session.
        repo_path: Path to the cloned repository (for codebase measurement).
        
    Returns:
        TokenEfficiencyScore with all metrics computed.
    """
    # Measure codebase ceiling if repo is available
    codebase_tokens = 0
    if repo_path and repo_path.exists():
        codebase_tokens = measure_codebase_tokens(repo_path)
    
    return TokenEfficiencyScore(
        task_id=task.task_id,
        native_retrieval_tokens=native_trace.total_tool_output_tokens,
        sg_retrieval_tokens=sg_trace.total_tool_output_tokens,
        native_total_tokens=native_trace.total_conversation_tokens,
        sg_total_tokens=sg_trace.total_conversation_tokens,
        codebase_tokens=codebase_tokens,
    )


def run_token_efficiency_batch(
    tasks: List[EvalTask],
    sg_traces: dict[str, AgentTrace],
    native_traces: dict[str, AgentTrace],
    repos_dir: Optional[Path] = None,
) -> List[TokenEfficiencyScore]:
    """Run token efficiency benchmark across multiple tasks.
    
    Args:
        tasks: List of evaluation tasks.
        sg_traces: Dict mapping task_id -> SG AgentTrace.
        native_traces: Dict mapping task_id -> native AgentTrace.
        repos_dir: Directory containing cloned repos.
        
    Returns:
        List of TokenEfficiencyScore objects.
    """
    results: List[TokenEfficiencyScore] = []
    
    for task in tasks:
        sg_trace = sg_traces.get(task.task_id)
        native_trace = native_traces.get(task.task_id)
        
        if not sg_trace or not native_trace:
            logger.warning(
                "Skipping %s: missing %s trace",
                task.task_id,
                "SG" if not sg_trace else "native",
            )
            continue
        
        repo_path = None
        if repos_dir:
            # Try common naming patterns
            repo_name = task.repo.split("/")[-1]
            for candidate in [repos_dir / repo_name, repos_dir / task.repo.replace("/", "__")]:
                if candidate.exists():
                    repo_path = candidate
                    break
        
        score = run_token_efficiency(task, sg_trace, native_trace, repo_path)
        results.append(score)
        
        logger.info(
            "  %s: native=%d sg=%d ratio=%.1fx",
            task.task_id,
            score.native_retrieval_tokens,
            score.sg_retrieval_tokens,
            score.retrieval_reduction_ratio,
        )
    
    return results
