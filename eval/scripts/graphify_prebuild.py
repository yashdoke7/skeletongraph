"""Prebuild graphify graphs for a dataset — ONE real LLM extraction per unique
REPO NAME (e.g. "django/django"), then COPY that graph.json into every other
task's checkout of the same repo.

FIX 2026-07: the original design (see docs/EVAL_PLAN_FINAL.md §8 — "built once
per repo, reused across task counts", ~150 LLM calls/repo avg, ~2 GPU-h total)
assumed one checkout was reused per repo. That stopped being true when
make_dataset.py moved to a git-worktree-per-task layout (needed for safe
PARALLEL agent runs — different tasks of the same repo now live in different
directories, e.g. 10 django tasks = 10 separate `django__django-*` dirs).
graphify's own `_ensure_extracted` (eval/backends/graphify.py) gates purely on
resolved path with no cross-directory awareness, so without this fix it would
silently do ONE EXTRACTION PER TASK (up to n, not ~12) — an ~8-10x cost blowup
this project almost spent real GPU-hours on before catching it.

This script restores the original per-repo-name economics WITHOUT touching
graphify's own extraction logic: group tasks by their dataset `repo` field
(not path), extract once per group via `_ensure_extracted` on one
representative checkout, then copy that graph.json (whole graphify-out/ dir)
into every sibling checkout so their own `_ensure_extracted` calls see it
already present and skip re-extraction. A graph built at one commit is treated
as valid for nearby commits of the same repo — SWE-bench issues are small,
targeted bug fixes, so a repo's overall structure doesn't meaningfully change
between two of its task commits; this is the same tradeoff the original
design already accepted (see EVAL_PLAN_FINAL.md's own "built once, reused").

Usage:
    python -m eval.scripts.graphify_prebuild <dataset.jsonl> --shard k/N

SHARDING for N parallel terminals (one NIM key per terminal) — shards by
UNIQUE REPO NAME now (not by path), same round-robin-balance intent as before:
    $env:OLLAMA_API_KEY="nvapi-KEYk"
    python -m eval.scripts.graphify_prebuild <dataset.jsonl> --shard k/N

Routing (whatever graphify env you set BEFORE calling this):
  - NIM (laptop):     OLLAMA_BASE_URL=https://integrate.api.nvidia.com/v1
                      OLLAMA_MODEL=meta/llama-3.3-70b-instruct
  - local vLLM (AMD): OLLAMA_BASE_URL=http://127.0.0.1:8000/v1
                      OLLAMA_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
  GRAPHIFY_OLLAMA_PARALLEL=1  GRAPHIFY_EXTRACT_TIMEOUT=3600  (or higher for astropy)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path


def _parse_shard(s: str) -> tuple[int, int]:
    try:
        k, n = s.split("/")
        k, n = int(k), int(n)
        if not (1 <= k <= n):
            raise ValueError
        return k, n
    except Exception:
        raise SystemExit(f"--shard must be k/N with 1<=k<=N, got {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Path to dataset .jsonl")
    ap.add_argument("--shard", default="1/1",
                    help="k/N — round-robin shard of unique REPO NAMES (default 1/1)")
    args = ap.parse_args()
    k, n = _parse_shard(args.shard)

    from eval.backends import graphify

    # 1) group all task checkouts by repo NAME (e.g. "django/django"), in
    # first-seen order (stable across shards).
    groups: dict = defaultdict(list)   # repo_name -> [repo_path, ...]
    order: list = []
    for line in open(args.dataset, encoding="utf-8"):
        rec = json.loads(line)
        rp, rn = rec.get("repo_path"), rec.get("repo")
        if not (rp and rn and Path(rp).is_dir()):
            continue
        if rn not in groups:
            order.append(rn)
        groups[rn].append(rp)

    # 2) round-robin assignment of REPO NAMES (not paths) across shards, so
    # workload is balanced by distinct repo, not diluted by task count.
    mine = [rn for i, rn in enumerate(order) if (i % n) == (k - 1)]
    total_checkouts = sum(len(groups[rn]) for rn in mine)
    print(f"shard {k}/{n}: this terminal will build {len(mine)} of "
          f"{len(order)} unique REPO NAMES ({total_checkouts} checkouts covered "
          f"via copy-after-build)\n", flush=True)
    for rn in mine:
        print(f"  • {rn}  ({len(groups[rn])} checkout(s))")
    print(flush=True)

    # 3) for each repo name: extract on ONE representative checkout, then copy
    # graphify-out/ into every sibling checkout of that same repo name.
    t_all = time.time()
    done = 0
    for rn in mine:
        paths = groups[rn]
        rep = Path(paths[0])
        t = time.time()
        graphify._ensure_extracted(rep)
        gj = rep / "graphify-out" / "graph.json"
        ok = gj.exists() and gj.stat().st_size > 0
        dt_extract = time.time() - t
        copied = 0
        if ok:
            src_dir = rep / "graphify-out"
            for sib in paths[1:]:
                dst_dir = Path(sib) / "graphify-out"
                if (dst_dir / "graph.json").exists():
                    continue   # already there (re-run after interruption)
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                copied += 1
        done += 1
        print(f"  [{done:2d}/{len(mine)}] {rn:32} extract={dt_extract:6.0f}s  "
              f"graph.json={'OK' if ok else 'MISSING'}  "
              f"copied to {copied}/{len(paths)-1} sibling checkout(s)", flush=True)
    print(f"\nshard {k}/{n} done — {done} repo names, "
          f"{total_checkouts} checkouts covered, {time.time()-t_all:.0f}s total "
          f"(re-running after an interruption is free — extraction AND copies "
          f"both skip what's already present).", flush=True)


if __name__ == "__main__":
    main()
