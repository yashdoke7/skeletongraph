"""Expand graphify.jsonl from its current subset to the EXACT main-100 task set.

The graphify arm was built on a 40-task subset of the 100 tasks every other arm
ran. To make the comparison matched, graphify must cover the SAME 100 task_ids —
not a fresh `make_dataset --n 100` sample (different stratification → different
tasks → unmatched to the other arms' results).

This script:
  1. keeps the existing graphify.jsonl rows VERBATIM (their graphs are already
     built; their repo_path/query/gold must not change), and
  2. generates rows for the missing task_ids by pulling problem_statement + patch
     from the SWE-bench Verified HF split and computing gold_fqns with the same
     parse_patch() make_dataset uses — so the new rows are schema-identical.

Ground-truth 100 ids come from the main run's result JSONs (or --ids <file>).

    python -m eval.scripts.expand_graphify_100 \
        --ids eval/datasets/main100_task_ids.txt \
        --in  eval/datasets/graphify.jsonl \
        --out eval/datasets/graphify.jsonl       # in-place expand (writes 100)

Repos for the new tasks must already be checked out (they are — the main run used
them); repo_path is set to the same base as the existing rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from eval.make_dataset import parse_patch


def _existing(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows[d["task_id"]] = d
    return rows


def _repo_base(rows: dict) -> str:
    """Infer the repo_path parent dir from an existing row (so new rows point to
    the SAME checkout location the built graphs live under)."""
    for d in rows.values():
        rp = d.get("repo_path", "")
        if rp:
            return str(Path(rp).parent)
    # fallback to the known swebench-data location
    return r"C:\Users\ASUS\Desktop\CS\Projects\swebench-data\repos"


def _load_ids(ids_arg: str | None) -> list[str]:
    if ids_arg and Path(ids_arg).is_file():
        return [l.strip() for l in Path(ids_arg).read_text(encoding="utf-8").splitlines() if l.strip()]
    # else derive from main-run sg result JSONs
    out = set()
    for p in glob.glob("eval/results/agent/nemotron_v2/*__sg__main__r0.json"):
        out.add(os.path.basename(p).split("__sg__")[0])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="eval/datasets/main100_task_ids.txt",
                    help="file with one task_id per line (the target set)")
    ap.add_argument("--in", dest="inp", default="eval/datasets/graphify.jsonl")
    ap.add_argument("--out", default="eval/datasets/graphify.jsonl")
    ap.add_argument("--split", default="princeton-nlp/SWE-bench_Verified")
    args = ap.parse_args()

    target = _load_ids(args.ids)
    existing = _existing(Path(args.inp))
    base = _repo_base(existing)
    missing = [t for t in target if t not in existing]
    print(f"target={len(target)}  existing={len(existing)}  to_generate={len(missing)}")
    if not missing:
        print("nothing to add — graphify.jsonl already covers the target set.")
        return

    from datasets import load_dataset
    print(f"loading {args.split} ...")
    ds = {x["instance_id"]: x for x in load_dataset(args.split, split="test")}

    new_rows = []
    skipped = []
    for tid in missing:
        inst = ds.get(tid)
        if inst is None:
            skipped.append((tid, "not in split")); continue
        gold_files, gold_fqns = parse_patch(inst.get("patch", ""))
        if not gold_files:
            skipped.append((tid, "no gold files")); continue
        rp = os.path.join(base, tid)
        if not os.path.isdir(rp):
            skipped.append((tid, "repo not checked out")); continue
        new_rows.append({
            "task_id": tid,
            "repo": inst["repo"],
            "base_commit": inst["base_commit"],
            "repo_path": rp,
            "query": inst.get("problem_statement", ""),
            "gold_files": gold_files,
            "gold_fqns": gold_fqns,
            "language": "python",
        })

    # write target-ordered: existing rows verbatim where present, new rows filled in
    merged = {**existing, **{r["task_id"]: r for r in new_rows}}
    ordered = [merged[t] for t in target if t in merged]
    Path(args.out).write_text(
        "\n".join(json.dumps(r) for r in ordered) + "\n", encoding="utf-8")
    print(f"wrote {len(ordered)} rows to {args.out}  (+{len(new_rows)} new)")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for tid, why in skipped:
            print(f"  {tid}: {why}")


if __name__ == "__main__":
    main()
