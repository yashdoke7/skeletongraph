"""
CRG-compatible dataset loader.

Loads evaluation tasks matching code-review-graph's exact YAML configs
(same repos, same commit SHAs) for direct head-to-head comparison.

This allows us to re-run CRG's benchmarks with:
  - tiktoken BPE (instead of CRG's len(text)//4)
  - Real agent traces (instead of static file reads)
  - Independent ground truth (instead of self-referential graph)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .base import EvalTask

logger = logging.getLogger(__name__)


# CRG's exact test configuration, extracted from their YAML config files
# Source: code-review-graph/code_review_graph/eval/configs/*.yaml
CRG_CONFIGS = {
    "express": {
        "url": "https://github.com/expressjs/express.git",
        "language": "javascript",
        "size_category": "small",
        "test_commits": [
            {
                "sha": "HEAD~1",
                "description": "Express commit 1",
                "changed_files": 1,
            },
            {
                "sha": "HEAD~2",
                "description": "Express commit 2",
                "changed_files": 1,
            },
        ],
    },
    "fastapi": {
        "url": "https://github.com/tiangolo/fastapi.git",
        "language": "python",
        "size_category": "medium",
        "test_commits": [
            {
                "sha": "fa3588c38c7473aca7536b12d686102de4b0f407",
                "description": "Fix typo for client_secret in OAuth2 form docstrings",
                "changed_files": 1,
            },
            {
                "sha": "0227991a01e61bf5cdd93cc00e9e243f52b47a4a",
                "description": "Exclude spam comments from statistics in scripts/people.py",
                "changed_files": 1,
            },
        ],
    },
    "flask": {
        "url": "https://github.com/pallets/flask.git",
        "language": "python",
        "size_category": "small",
        "test_commits": [
            {
                "sha": "HEAD~1",
                "description": "Flask commit 1",
                "changed_files": 2,
            },
            {
                "sha": "HEAD~2",
                "description": "Flask commit 2",
                "changed_files": 1,
            },
        ],
    },
    "gin": {
        "url": "https://github.com/gin-gonic/gin.git",
        "language": "go",
        "size_category": "small",
        "test_commits": [
            {"sha": "HEAD~1", "description": "Gin commit 1", "changed_files": 1},
            {"sha": "HEAD~2", "description": "Gin commit 2", "changed_files": 1},
            {"sha": "HEAD~3", "description": "Gin commit 3", "changed_files": 2},
        ],
    },
    "httpx": {
        "url": "https://github.com/encode/httpx.git",
        "language": "python",
        "size_category": "small",
        "test_commits": [
            {"sha": "HEAD~1", "description": "httpx commit 1", "changed_files": 1},
            {"sha": "HEAD~2", "description": "httpx commit 2", "changed_files": 1},
        ],
    },
    "nextjs": {
        "url": "https://github.com/vercel/next.js.git",
        "language": "typescript",
        "size_category": "large",
        "test_commits": [
            {"sha": "HEAD~1", "description": "Next.js commit 1", "changed_files": 2},
            {"sha": "HEAD~2", "description": "Next.js commit 2", "changed_files": 3},
        ],
    },
}

# CRG's published benchmark results for comparison
CRG_PUBLISHED_RESULTS = {
    "express": {"avg_naive_tokens": 693, "avg_graph_tokens": 983, "reduction": 0.7},
    "fastapi": {"avg_naive_tokens": 4944, "avg_graph_tokens": 614, "reduction": 8.1},
    "flask": {"avg_naive_tokens": 44751, "avg_graph_tokens": 4252, "reduction": 9.1},
    "gin": {"avg_naive_tokens": 21972, "avg_graph_tokens": 1153, "reduction": 16.4},
    "httpx": {"avg_naive_tokens": 12044, "avg_graph_tokens": 1728, "reduction": 6.9},
    "nextjs": {"avg_naive_tokens": 9882, "avg_graph_tokens": 1249, "reduction": 8.0},
}


def load_crg_compat(
    repos: Optional[List[str]] = None,
) -> List[EvalTask]:
    """Load CRG-compatible evaluation tasks.
    
    Args:
        repos: Filter to specific repos (e.g. ["fastapi", "flask"]).
        
    Returns:
        List of EvalTask objects matching CRG's exact test configuration.
    """
    tasks: List[EvalTask] = []
    
    for repo_name, config in CRG_CONFIGS.items():
        if repos and repo_name not in repos:
            continue
        
        for i, commit in enumerate(config["test_commits"]):
            task = EvalTask(
                task_id=f"crg_{repo_name}_{i+1}",
                repo=repo_name,
                repo_url=config["url"],
                base_commit=commit["sha"],
                problem_statement=commit["description"],
                language=config["language"],
                size_category=config["size_category"],
                description=commit["description"],
                source="crg-compat",
            )
            tasks.append(task)
    
    logger.info("Loaded %d CRG-compatible tasks", len(tasks))
    return tasks


def get_crg_published_results() -> dict:
    """Return CRG's published benchmark results for comparison reporting."""
    return CRG_PUBLISHED_RESULTS
