"""Stage 0 dataset builder.

Pulls SWE-bench Verified, samples a stratified subset, clones each repo at its
base commit (git worktrees — shared .git, low disk), extracts retrieval ground
truth from the gold patch, and writes eval/datasets/stage0.jsonl.

Usage:
    pip install datasets
    python eval/make_dataset.py --n 30
    python eval/make_dataset.py --smoke              # 2 tasks, fast sanity check
    python eval/make_dataset.py --n 50 --seed 7

Output line schema (one JSON per line):
    {
      "task_id": "...", "repo": "owner/name", "base_commit": "...",
      "repo_path": "eval/datasets/repos/<task_id>",
      "query": "<problem statement>",
      "gold_files": ["pkg/mod.py", ...],          # exact, from patch
      "gold_fqns":  ["pkg/mod.py::func", ...]      # best-effort, from hunk headers
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

EVAL_DIR = Path(__file__).resolve().parent
# Heavy IO (clones + worktrees) lives under SG_EVAL_DATA_ROOT when set, so the
# GBs of repos stay OUT of the SG repo. config.py honors the same env for the
# per-run workspaces — set it once and both agree. Default = legacy in-repo.
_DATA_ROOT = (Path(os.environ["SG_EVAL_DATA_ROOT"]) if os.environ.get("SG_EVAL_DATA_ROOT")
              else EVAL_DIR / "datasets")
REPOS_DIR = _DATA_ROOT / "repos"
CACHE_DIR = _DATA_ROOT / "_repo_cache"
OUT_PATH = EVAL_DIR / "datasets" / "stage0.jsonl"

_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_]\w*)")


# ── patch parsing ─────────────────────────────────────────────────────────


def parse_patch(patch: str) -> Tuple[List[str], List[str]]:
    """Return (gold_files, gold_fqns) from a unified-diff patch.

    Files are exact. FQNs are best-effort: the text after the second @@ in a
    hunk header usually names the enclosing def/class.
    """
    files: List[str] = []
    fqns: List[str] = []
    current: str | None = None

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].strip()
            if current and current != "/dev/null":
                files.append(current)
        elif line.startswith("@@") and current:
            # @@ -a,b +c,d @@ <context>
            ctx = line.split("@@", 2)[-1] if line.count("@@") >= 2 else ""
            m = _DEF_RE.search(ctx)
            if m:
                fqns.append(f"{current}::{m.group(1)}")

    return list(dict.fromkeys(files)), list(dict.fromkeys(fqns))


# ── git ───────────────────────────────────────────────────────────────────


def _run(cmd: List[str], check: bool = True) -> int:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  ! {' '.join(cmd[:3])}... -> {r.stderr.strip()[:200]}")
    return r.returncode


def setup_repo(repo: str, base_commit: str, task_id: str) -> Path | None:
    """Clone `repo` once into the cache, add a detached worktree at base_commit."""
    slug = repo.replace("/", "__")
    cache = CACHE_DIR / slug
    dest = REPOS_DIR / task_id

    if dest.exists() and any(dest.iterdir()):
        return dest

    if not cache.exists():
        print(f"  cloning {repo} ...")
        if _run(["git", "clone", f"https://github.com/{repo}.git", str(cache)]) != 0:
            return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    rc = _run(["git", "-C", str(cache), "worktree", "add", "--detach",
               str(dest.resolve()), base_commit], check=False)
    if rc != 0:
        _run(["git", "-C", str(cache), "fetch", "--all", "--tags"], check=False)
        rc = _run(["git", "-C", str(cache), "worktree", "add", "--detach",
                   str(dest.resolve()), base_commit], check=False)
    if rc != 0:
        print(f"  ! could not check out {base_commit[:10]} for {task_id}")
        return None
    return dest


# ── main ───────────────────────────────────────────────────────────────────


def stratified_sample(instances: list, n: int, seed: int) -> list:
    """Sample n instances spread across repos (round-robin by repo)."""
    by_repo = defaultdict(list)
    for inst in instances:
        by_repo[inst["repo"]].append(inst)
    rng = random.Random(seed)
    for v in by_repo.values():
        rng.shuffle(v)
    picked, repos = [], sorted(by_repo)
    i = 0
    while len(picked) < n and any(by_repo.values()):
        bucket = by_repo[repos[i % len(repos)]]
        if bucket:
            picked.append(bucket.pop())
        i += 1
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0 dataset builder")
    ap.add_argument("--n", type=int, default=30, help="Number of tasks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="2-task sanity dataset")
    ap.add_argument("--split", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    n = 2 if args.smoke else args.n

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Loading {args.split} ...")
    ds = load_dataset(args.split, split="test")
    instances = [dict(x) for x in ds]
    print(f"  {len(instances)} instances available")

    picked = stratified_sample(instances, n, args.seed)
    print(f"Selected {len(picked)} tasks across "
          f"{len(set(p['repo'] for p in picked))} repos\n")

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, inst in enumerate(picked, 1):
        tid = inst["instance_id"]
        print(f"[{i}/{len(picked)}] {tid}")
        repo_path = setup_repo(inst["repo"], inst["base_commit"], tid)
        if repo_path is None:
            print("  skipped (checkout failed)")
            continue
        gold_files, gold_fqns = parse_patch(inst.get("patch", ""))
        if not gold_files:
            print("  skipped (no gold files in patch)")
            continue
        rows.append({
            "task_id": tid,
            "repo": inst["repo"],
            "base_commit": inst["base_commit"],
            "repo_path": str(repo_path),
            "query": inst.get("problem_statement", ""),
            "gold_files": gold_files,
            "gold_fqns": gold_fqns,
        })

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} tasks to {args.out}")
    if len(rows) < len(picked):
        print(f"  ({len(picked) - len(rows)} skipped — see log above)")


if __name__ == "__main__":
    main()
