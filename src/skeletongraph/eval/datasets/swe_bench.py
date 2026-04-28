"""
SWE-bench Verified dataset loader.

Loads tasks from the princeton-nlp/SWE-bench_Verified dataset 
(or a local JSONL cache) and converts them to EvalTask objects.

SWE-bench Verified contains 500 human-validated issues across 12 Python repos:
  - django/django          (~4600 files, web framework)
  - scikit-learn/scikit-learn (~1200 files, ML library)
  - matplotlib/matplotlib  (~2000 files, plotting)
  - pandas-dev/pandas      (~800 files, data analysis)
  - pytest-dev/pytest      (~400 files, testing framework)
  - psf/requests           (~150 files, HTTP library)
  - astropy/astropy        (~2500 files, astronomy)
  - marshmallow-code/marshmallow (~100 files, serialization)
  - mwaskom/seaborn        (~200 files, visualization)
  - pylint-dev/pylint      (~600 files, linting)
  - pallets/flask          (~100 files, web micro-framework)
  - getpelican/pelican     (~150 files, static site generator)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .base import EvalTask

logger = logging.getLogger(__name__)

# Default local cache location
CACHE_DIR = Path.home() / ".skeletongraph" / "datasets"
SWE_BENCH_CACHE = CACHE_DIR / "swe_bench_verified.jsonl"

# HuggingFace dataset identifier
HF_DATASET = "princeton-nlp/SWE-bench_Verified"

# Repo size categories based on typical file counts
REPO_SIZES = {
    "django/django": "large",
    "scikit-learn/scikit-learn": "large",
    "matplotlib/matplotlib": "large",
    "astropy/astropy": "large",
    "pandas-dev/pandas": "medium",
    "pylint-dev/pylint": "medium",
    "pytest-dev/pytest": "medium",
    "mwaskom/seaborn": "small",
    "psf/requests": "small",
    "pallets/flask": "small",
    "marshmallow-code/marshmallow": "small",
    "getpelican/pelican": "small",
}

REPO_URLS = {
    "django/django": "https://github.com/django/django.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "pandas-dev/pandas": "https://github.com/pandas-dev/pandas.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "marshmallow-code/marshmallow": "https://github.com/marshmallow-code/marshmallow.git",
    "getpelican/pelican": "https://github.com/getpelican/pelican.git",
}


def download_swe_bench(cache_path: Path = SWE_BENCH_CACHE) -> Path:
    """Download SWE-bench Verified dataset from HuggingFace.
    
    Uses the datasets library if available, otherwise falls back 
    to direct HTTP download of the parquet/JSONL file.
    
    Returns path to the cached JSONL file.
    """
    if cache_path.exists():
        logger.info("Using cached SWE-bench dataset: %s", cache_path)
        return cache_path
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        logger.info("Downloading SWE-bench Verified from HuggingFace...")
        ds = load_dataset(HF_DATASET, split="test")
        
        # Write to JSONL cache
        with open(cache_path, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, default=str) + "\n")
        
        logger.info("Cached %d tasks to %s", len(ds), cache_path)
        return cache_path
        
    except ImportError:
        logger.error(
            "Install 'datasets' package to download SWE-bench:\n"
            "  pip install datasets\n"
            "Or manually download and place the JSONL at: %s",
            cache_path,
        )
        raise


def load_swe_bench(
    cache_path: Path = SWE_BENCH_CACHE,
    repos: Optional[List[str]] = None,
    limit: Optional[int] = None,
    size_category: Optional[str] = None,
) -> List[EvalTask]:
    """Load SWE-bench Verified tasks as EvalTask objects.
    
    Args:
        cache_path: Path to local JSONL cache.
        repos: Filter to specific repos (e.g. ["django/django", "psf/requests"]).
        limit: Max number of tasks to return.
        size_category: Filter by repo size ("small", "medium", "large").
        
    Returns:
        List of EvalTask objects ready for evaluation.
    """
    if not cache_path.exists():
        cache_path = download_swe_bench(cache_path)
    
    tasks: List[EvalTask] = []
    
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            raw = json.loads(line)
            repo = raw.get("repo", "")
            
            # Apply filters
            if repos and repo not in repos:
                continue
            
            repo_size = REPO_SIZES.get(repo, "unknown")
            if size_category and repo_size != size_category:
                continue
            
            # Parse FAIL_TO_PASS and PASS_TO_PASS (stored as JSON strings in SWE-bench)
            fail_to_pass = _parse_test_list(raw.get("FAIL_TO_PASS", "[]"))
            pass_to_pass = _parse_test_list(raw.get("PASS_TO_PASS", "[]"))
            
            task = EvalTask(
                task_id=raw.get("instance_id", ""),
                repo=repo,
                repo_url=REPO_URLS.get(repo, f"https://github.com/{repo}.git"),
                base_commit=raw.get("base_commit", "HEAD"),
                problem_statement=raw.get("problem_statement", ""),
                gold_patch=raw.get("patch", ""),
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                test_cmd="pytest",
                test_patch=raw.get("test_patch", ""),
                language="python",
                size_category=repo_size,
                description=raw.get("problem_statement", "")[:200],
                source="swe-bench-verified",
                version=raw.get("version", ""),
            )
            tasks.append(task)
            
            if limit and len(tasks) >= limit:
                break
    
    logger.info(
        "Loaded %d SWE-bench tasks%s",
        len(tasks),
        f" (filtered to {repos})" if repos else "",
    )
    return tasks


def list_available_repos(cache_path: Path = SWE_BENCH_CACHE) -> dict:
    """List repos in the dataset with task counts.
    
    Returns dict mapping repo name -> task count.
    """
    if not cache_path.exists():
        return {repo: "?" for repo in REPO_URLS}
    
    counts: dict[str, int] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            repo = raw.get("repo", "unknown")
            counts[repo] = counts.get(repo, 0) + 1
    
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _parse_test_list(raw_value) -> List[str]:
    """Parse test list from SWE-bench format (can be JSON string or list)."""
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Might be a single test name
        if raw_value.strip():
            return [raw_value.strip()]
    return []
