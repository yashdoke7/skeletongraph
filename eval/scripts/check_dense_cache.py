"""Verify that the sg-fusion arm actually ran 3-way fusion -- i.e. that every
prepared repo copy has a warm dense-embedding cache built by the RIGHT model.

Why this matters
----------------
fusion's dense leg is guarded by a hard timeout (SG_DENSE_TIMEOUT_S, default
20s, see retrieval/fusion.py:119). If the corpus embeddings are not already
cached, the first sg_search call races a cold model load + full-corpus encode
against that bound -- and on a miss it DEGRADES SILENTLY to 2-way
(lexical + structural) rather than failing. A run that degraded looks exactly
like a normal run in the trajectory: same tool calls, same JSON, no error. The
only durable evidence is whether the cache was on disk before the agent
started.

prepare_repo() warms it (run_claude_code.py:481), so a correctly prepared
sg-fusion copy always has one. This script confirms that actually happened
rather than assuming it.

How the model is identified
---------------------------
The cache filename (`embcache_code.npz`) encodes the DOC TYPE, not the model,
so it cannot tell you which model wrote it. The embedding WIDTH can:

    768 = jinaai/jina-embeddings-v2-base-code   (SG_DENSE_MODEL default,
                                                 the semantic leg of fusion)
    384 = all-MiniLM-L6-v2                      (SG_EMBED_MODEL -- the
                                                 index-time confidence
                                                 tiebreaker, NOT the fusion leg)

So a 384-wide cache means the wrong model was wired in and the run is not the
3-way fusion the arm name claims.

Shapes are read from the .npy header inside the zip, so this stays fast and
constant-memory even across GB of caches.

Usage
-----
    python -m eval.scripts.check_dense_cache --dataset eval/datasets/swe_rebench_100.jsonl --limit 50
    python -m eval.scripts.check_dense_cache --dataset eval/datasets/swe_rebench_100.jsonl --limit 50 --verbose

Exit code is non-zero if any task is missing a cache or has the wrong width,
so it can gate a run.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy.lib.format as npf

EXPECTED_DIM = 768          # jina-embeddings-v2-base-code
TIEBREAKER_DIM = 384        # all-MiniLM-L6-v2 -- wrong model for this leg


def copy_dir(task: dict, arm: str) -> Path:
    """Mirror of run_claude_code._repo_dir / _copies_root, without importing it
    (that module pulls in the whole harness)."""
    base = Path(task["repo_path"]).resolve().parent.parent / "_claude_repos"
    return base / (task["task_id"] if arm == "sg-rerank"
                   else f"{task['task_id']}__{arm}")


def cache_info(repo: Path):
    """(status, dim, n_vecs, mtime) for a copy's dense cache."""
    if not repo.is_dir():
        return "NO_COPY", None, None, None
    cdir = repo / ".skeletongraph" / "dense_cache"
    files = sorted(cdir.glob("embcache_*.npz")) if cdir.is_dir() else []
    if not files:
        idx = (repo / ".skeletongraph").is_dir()
        return ("NO_CACHE" if idx else "NO_INDEX"), None, None, None

    p = files[0]
    dim = n = None
    try:
        with zipfile.ZipFile(p) as z:
            for nm in z.namelist():
                if not nm.startswith("vecs"):
                    continue
                with z.open(nm) as fh:
                    # Dispatch on the .npy version: the 1.0 and 2.0 headers
                    # differ in length field width, and the combined private
                    # helper is not stable API across numpy releases.
                    major, _ = npf.read_magic(fh)
                    reader = (npf.read_array_header_1_0 if major == 1
                              else npf.read_array_header_2_0)
                    shape, _, _ = reader(fh)
                if len(shape) == 2:
                    n, dim = shape
                break
    except Exception as e:  # noqa: BLE001 -- a corrupt cache is a finding
        return f"UNREADABLE ({type(e).__name__})", None, None, None

    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    if dim is None:
        return "NO_VECS", None, None, mtime
    if dim == EXPECTED_DIM:
        return "OK", dim, n, mtime
    if dim == TIEBREAKER_DIM:
        return "WRONG_MODEL(MiniLM)", dim, n, mtime
    return f"WRONG_DIM({dim})", dim, n, mtime


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--arm", default="sg-fusion")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true",
                    help="one line per task, not just the problems")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in
             args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        tasks = tasks[:args.limit]

    buckets: dict[str, list[str]] = {}
    dims: set[int] = set()
    times: list[datetime] = []
    total_vecs = 0
    for t in tasks:
        status, dim, n, mtime = cache_info(copy_dir(t, args.arm))
        buckets.setdefault(status, []).append(t["task_id"])
        if dim:
            dims.add(dim)
        if n:
            total_vecs += n
        if mtime:
            times.append(mtime)
        if args.verbose:
            extra = (f"  dim={dim} vecs={n:,} built={mtime:%Y-%m-%d %H:%M}"
                     if dim else "")
            print(f"  {status:22s} {t['task_id']}{extra}")

    print(f"\n{args.dataset.name} [{args.arm}] — {len(tasks)} tasks")
    for status in sorted(buckets):
        ids = buckets[status]
        print(f"  {status:22s} {len(ids):>4}")
        if status != "OK" and not args.verbose:
            print(f"      {', '.join(ids[:4])}"
                  + (f" +{len(ids) - 4} more" if len(ids) > 4 else ""))

    if times:
        print(f"\n  caches built {min(times):%Y-%m-%d %H:%M} .. "
              f"{max(times):%Y-%m-%d %H:%M}")
        print(f"  embedding widths seen: {sorted(dims) or 'none'}"
              + ("  (jina-v2-base-code)" if dims == {EXPECTED_DIM} else ""))
        print(f"  total cached vectors: {total_vecs:,}")

    bad = sum(len(v) for k, v in buckets.items() if k != "OK")
    if bad:
        print(f"\n{bad} task(s) would run DEGRADED (2-way, no dense leg).")
        print("Fix by re-preparing — warming is idempotent, so this is cheap "
              "and costs no agent tokens:")
        print(f"  python -m eval.agent.run_claude_code --dataset {args.dataset} "
              f"--arm {args.arm} --limit {len(tasks)} --prepare-only")
        sys.exit(1)
    print("\nAll copies carry a jina-width dense cache — fusion ran 3-way.")


if __name__ == "__main__":
    main()
