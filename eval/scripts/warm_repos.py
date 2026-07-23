"""Prewarm the product fusion engine (bm25 + dense + lean SG engine) for every
repo in a probe/eval dataset, BEFORE running a probe or agent stage.

Without this, retrieve_fusion's dense leg is racing a 20s timeout
(SG_DENSE_TIMEOUT_S, see src/skeletongraph/retrieval/fusion.py) against a cold
model load + full-corpus encode on the FIRST query per repo. Depending on
system load that first call may or may not make the deadline, silently
degrading to 2-signal (BM25+SG) fusion — non-deterministic behavior that
looks like retrieval noise but is actually just cache temperature. Calling
`warm()` here pays that cost up front, outside any timed measurement.

Usage:
    python -m eval.scripts.warm_repos --dataset eval/datasets/swe_rebench_100_prose.jsonl --limit 15
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skeletongraph.retrieval.fusion import warm


def main() -> None:
    ap = argparse.ArgumentParser(description="Prewarm bm25+dense+SG for a dataset's repos")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    repos = []
    seen = set()
    for t in tasks:
        rp = t.get("repo_path")
        if rp and rp not in seen:
            seen.add(rp)
            repos.append(rp)

    print(f"{len(repos)} unique repo(s) across {len(tasks)} task(s)")
    for i, rp in enumerate(repos, 1):
        p = Path(rp)
        if not p.exists():
            print(f"[{i}/{len(repos)}] SKIP (missing): {rp}")
            continue
        t0 = time.monotonic()
        try:
            warm(p, mode="fusion", full_dense=True)
            print(f"[{i}/{len(repos)}] {rp}  ({time.monotonic() - t0:.1f}s)")
        except Exception as e:
            print(f"[{i}/{len(repos)}] FAILED {rp}: {e}")


if __name__ == "__main__":
    main()
