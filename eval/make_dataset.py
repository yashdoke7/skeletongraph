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

# Multi-language enclosing-symbol extraction from the hunk-header context.
# The text after the second `@@` in a hunk header is the enclosing decl, but its
# keyword differs by language:
#   Python:  def NAME(... | class NAME | async def NAME
#   Go:      func NAME(... | func (r *T) NAME(... | type NAME ... | interface NAME
#   JS/TS:   function NAME(... | const/let/var NAME = (... | NAME(...) {  (method)
#   Java:    public/private ... NAME(... | class NAME | interface NAME
# This is best-effort — Pro's multi-language hunks were silently dropping
# gold_fqns for everything but Python (funcR@10 was `?` for go/js/ts).
_DEF_RES = (
    # Keyword-led declarations — keyword + identifier
    re.compile(r"\b(?:async\s+def|def|class|func(?:\s*\([^)]*\))?|function|interface|type)\s+([A-Za-z_$][\w$]*)"),
    # JS/TS arrow / function-expression assignment: const NAME = (...) => or = function(
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?[\(<]|\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*function\b"),
    # Java/C#-style: visibility + (generics?) + returnType + NAME ( … )
    re.compile(r"\b(?:public|private|protected|static|final|abstract|override)\s+(?:[\w<>,?\[\]\s]+\s+)?([A-Za-z_$][\w$]*)\s*\("),
    # Bare method-in-class headers (TS/Java/C#): NAME(args) { or : Type {
    re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[\w<>,?\[\]|&\s]+)?\s*\{"),
)


def _extract_decl_name(ctx: str) -> str | None:
    """Pick the first NAME captured by any language-pattern in the hunk context."""
    for rx in _DEF_RES:
        m = rx.search(ctx)
        if m:
            for g in m.groups():
                if g:
                    return g
    return None


# ── patch parsing ─────────────────────────────────────────────────────────


def parse_patch(patch: str) -> Tuple[List[str], List[str]]:
    """Return (gold_files, gold_fqns) from a unified-diff patch.

    Files are exact. FQNs are best-effort: the text after the second @@ in a
    hunk header usually names the enclosing def/class/func/function. Now
    multi-language (Python/Go/JS/TS/Java) — previously Python-only, which is
    why funcR@10 columns were `?` across non-Python Pro tasks.
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
            name = _extract_decl_name(ctx)
            if name:
                fqns.append(f"{current}::{name}")

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

    def _try_worktree() -> int:
        return _run(["git", "-C", str(cache), "worktree", "add", "--detach",
                     str(dest.resolve()), base_commit], check=False)

    # Progressive recovery for SWE-Bench-Pro: many base_commits are PR-head
    # commits that aren't in the default branch's history. Try the cheap path
    # first, fall back to wider/heavier fetches only when needed.
    rc = _try_worktree()
    if rc != 0:
        # (1) standard refresh — branches + tags
        _run(["git", "-C", str(cache), "fetch", "--all", "--tags"], check=False)
        rc = _try_worktree()
    if rc != 0:
        # (2) PR-head refs: many Pro tasks point at commits that live ONLY in
        # refs/pull/*/head — the default fetch ignores those. This rescues
        # most "could not check out <sha>" failures on protonmail/webclients,
        # element-hq/element-web, ansible/ansible PR-style instances, etc.
        _run(["git", "-C", str(cache), "fetch", "origin",
              "+refs/pull/*/head:refs/remotes/origin/pr/*"], check=False)
        rc = _try_worktree()
    if rc != 0:
        # (3) fetch the SHA directly — GitHub has uploadpack.allowAnySHA1InWant
        # since 2017, so unreachable commits (closed-without-merge PRs) often
        # still resolve. Last resort because it's the slowest.
        _run(["git", "-C", str(cache), "fetch", "origin", base_commit],
             check=False)
        rc = _try_worktree()
    if rc != 0:
        print(f"  ! could not check out {base_commit[:10]} for {task_id} "
              f"(tried: default, fetch-all, PR-refs, direct-SHA — commit "
              f"likely deleted upstream)")
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
            # SWE-bench Pro carries `repo_language` (e.g. JavaScript, Go); Verified
            # is Python-only. Captured for the per-language breakdown; "python"
            # default keeps Verified rows uniform.
            "language": (inst.get("repo_language") or "python"),
        })

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} tasks to {args.out}")
    if len(rows) < len(picked):
        print(f"  ({len(picked) - len(rows)} skipped — see log above)")


if __name__ == "__main__":
    main()
