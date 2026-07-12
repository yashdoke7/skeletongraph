"""Prewarm fusion's dense-embedding cache for a dataset's repos, BEFORE a timed
agent run — the step skipped earlier that made a rerank-vs-fusion comparison
accidentally test rerank twice (fusion's dense leg had nothing to load, so it
silently degraded to rerank; see project_mcp_cost_root_cause).

First build per repo is index-class cost: minutes on CPU, seconds on GPU.
INCREMENTAL after that — only functions whose text changed re-embed (see
skeletongraph.retrieval.dense). Safe to re-run: a warm repo is a fast no-op.

Usage:
  python -m eval.agent.prewarm_fusion_dense --dataset <jsonl> --limit 30
  python -m eval.agent.prewarm_fusion_dense --dataset <jsonl> --limit 30 --start 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="tasks jsonl (repo_path must exist on disk)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    args = ap.parse_args()

    lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    tasks = [json.loads(l) for l in lines if l.strip()]
    end = len(tasks) if not args.limit else args.start + args.limit
    tasks = tasks[args.start:end]

    from skeletongraph.retrieval.fusion import warm as warm_retrieval
    from skeletongraph.retrieval.bm25_flat import _functions_with_text

    print(f"Prewarming fusion dense cache for {len(tasks)} repo(s) "
          f"({args.dataset}, tasks {args.start}-{end})")
    for i, task in enumerate(tasks, 1):
        tid = task.get("task_id", "?")
        repo = Path(task.get("repo_path", ""))
        if not repo.is_dir():
            print(f"[{i}/{len(tasks)}] SKIP {tid}: repo_path missing ({repo})")
            continue
        cache = repo / ".skeletongraph" / "dense_cache" / "embcache_code.npz"
        if cache.exists():
            print(f"[{i}/{len(tasks)}] {tid}: already warm, skipping")
            continue
        n_funcs = len(_functions_with_text(repo))
        print(f"[{i}/{len(tasks)}] {tid}: encoding {n_funcs} functions ...", flush=True)
        t0 = time.time()
        try:
            warm_retrieval(repo, mode="fusion", full_dense=True)
            print(f"[{i}/{len(tasks)}] {tid}: done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[{i}/{len(tasks)}] {tid}: FAILED ({time.time()-t0:.1f}s) — {e}")


if __name__ == "__main__":
    main()
