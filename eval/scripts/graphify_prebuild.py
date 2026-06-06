"""Prebuild graphify graphs ONCE per unique repo in a dataset (the one-time extract).

graphify builds a per-repo `graphify-out/graph.json` using an LLM. PER-REPO, not
per-task: SWE-bench Verified-100 = ~12 unique repos = 12 graph builds. The graph
lands inside each repo (`<repo>/graphify-out/graph.json`) and persists; subsequent
agent runs short-circuit on existing graphs (free).

SHARDING for N parallel terminals (one NIM key per terminal):
    # terminal k = 1..N, each with its own OLLAMA_API_KEY:
    $env:OLLAMA_API_KEY="nvapi-KEYk"
    python -m eval.scripts.graphify_prebuild <dataset.jsonl> --shard k/N

Repos are round-robin distributed across shards (--shard 2/5 takes repos at indices
1, 6, 11 …). Each shard is independent — terminals never touch the same repo, and
because storage is just on-disk graph.json files, "results" merge automatically.

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
import time
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
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Path to dataset .jsonl")
    ap.add_argument("--shard", default="1/1",
                    help="k/N — round-robin shard of unique repos (default 1/1)")
    args = ap.parse_args()
    k, n = _parse_shard(args.shard)

    from eval.backends import graphify

    # 1) collect unique repos in dataset order (stable across shards)
    all_repos: list[str] = []
    seen: set[str] = set()
    for line in open(args.dataset, encoding="utf-8"):
        rp = json.loads(line).get("repo_path")
        if rp and rp not in seen and Path(rp).is_dir():
            seen.add(rp); all_repos.append(rp)

    # 2) round-robin assignment so workload is balanced (alphabetical sort would
    # land big-repo Python projects in the same shard).
    mine = [r for i, r in enumerate(all_repos) if (i % n) == (k - 1)]
    print(f"shard {k}/{n}: this terminal will build {len(mine)} of "
          f"{len(all_repos)} unique repos\n", flush=True)
    for r in mine:
        print(f"  • {Path(r).name}")
    print(flush=True)

    # 3) build. Each repo's graphify-out/graph.json is the persistent artifact;
    # _ensure_extracted skips if it already exists, so re-running this terminal
    # after an interruption is FREE — it picks up where it left off.
    t_all = time.time()
    done = 0
    for rp in mine:
        t = time.time()
        graphify._ensure_extracted(Path(rp))
        gj = Path(rp) / "graphify-out" / "graph.json"
        size = gj.stat().st_size if gj.exists() else 0
        ok = size > 0
        done += 1
        print(f"  [{done:2d}/{len(mine)}] {Path(rp).name:32} "
              f"{time.time()-t:6.0f}s  graph.json={'OK' if ok else 'MISSING'} "
              f"({size//1024} KB)", flush=True)
    print(f"\nshard {k}/{n} done — {done} repos built in "
          f"{time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
