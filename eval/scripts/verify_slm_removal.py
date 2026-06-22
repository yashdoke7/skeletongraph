"""Behavior-preservation check for the SLM/dead-code removal.

Snapshots the EXACT ranked FQN lists that sg / sg-rerank return for a sample of
tasks. Run it BEFORE the removal (--out before.json) and AFTER (--out after.json),
then diff: if the lists are identical, the removal changed no retrieval behavior
(which it must not — slm_result is always None in the query path).

    python -m eval.scripts.verify_slm_removal --out before.json
    # ... do the removal ...
    python -m eval.scripts.verify_slm_removal --out after.json
    python -m eval.scripts.verify_slm_removal --diff before.json after.json

Retrieval is deterministic given (query, repo, index), and we load each repo's
existing .skeletongraph (no rebuild), so the only variable between runs is the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent.parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))


def _snapshot(dataset: str, n: int, out: str) -> None:
    from agent.tools import _retrieve_sg, _retrieve_rerank
    rows = [json.loads(l) for l in open(dataset, encoding="utf-8") if l.strip()]
    picked, snap = [], {}
    for r in rows:
        rp = Path(r.get("repo_path", ""))
        if (rp / ".skeletongraph").is_dir():     # only repos with a usable index
            picked.append(r)
        if len(picked) >= n:
            break
    print(f"snapshotting {len(picked)} tasks -> {out}")
    for r in picked:
        tid, q, rp = r["task_id"], r["query"], Path(r["repo_path"])
        try:
            sg = _retrieve_sg(q, rp, 10)
            rr = _retrieve_rerank(q, rp, 10)
        except Exception as e:
            sg, rr = [f"ERROR: {e}"], [f"ERROR: {e}"]
        snap[tid] = {"sg": list(sg), "rerank": list(rr)}
        print(f"  {tid}: sg={len(snap[tid]['sg'])} rerank={len(snap[tid]['rerank'])}")
    Path(out).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def _diff(a: str, b: str) -> None:
    da = json.loads(Path(a).read_text(encoding="utf-8"))
    db = json.loads(Path(b).read_text(encoding="utf-8"))
    keys = sorted(set(da) | set(db))
    diffs = 0
    for k in keys:
        for arm in ("sg", "rerank"):
            va = da.get(k, {}).get(arm)
            vb = db.get(k, {}).get(arm)
            if va != vb:
                diffs += 1
                print(f"DIFF {k} [{arm}]:\n  before={va}\n  after ={vb}")
    if diffs == 0:
        print(f"IDENTICAL across {len(keys)} tasks x2 arms — removal is behavior-preserving.")
    else:
        print(f"\n{diffs} difference(s) — investigate before trusting the removal.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/datasets/graphify_100.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", nargs=2, default=None, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.diff:
        _diff(*args.diff)
    elif args.out:
        _snapshot(args.dataset, args.n, args.out)
    else:
        ap.error("pass --out <file> to snapshot, or --diff before after")


if __name__ == "__main__":
    main()
