"""One command to build a DECONTAMINATED SWE-rebench dataset that drops into
both existing harnesses (Claude Code + NIM react-loop) unchanged.

Why this exists
---------------
SWE-bench Verified is saturated by memorization: in our own nemotron_v2 run the
`none` arm — no code access at all — scored 44% against the best retrieval
arm's 45%. Retrieval is worth ~1 point there, so pass@1 on Verified simply
cannot measure retrieval quality. The cause is that Verified is 12 FAMOUS repos
(django/astropy/sympy/...) the model half-knows.

SWE-rebench (Nebius) fixes both halves of that:
  - 3,400+ Python repos instead of 12 -> the model has no memorized prior
  - monthly splits, so you can take only months that POSTDATE a model cutoff
  - prebuilt Docker images (namespace `swerebench`) -> real pass@1 verification
  - SWE-bench-compatible schema -> our make_dataset.py already understands it

Usage
-----
    # default: post-Jan-2026 months (decontaminated for Sonnet), 100 tasks
    python -m eval.scripts.make_rebench

    # explicit control
    python -m eval.scripts.make_rebench --n 150 --splits 2026_02,2026_03 --seed 7
    python -m eval.scripts.make_rebench --list-splits     # what's available now

Then run EITHER harness against the produced jsonl (no other changes needed):

    $DS = "eval/datasets/swe_rebench_100.jsonl"

    # Claude Code
    python -m eval.agent.run_claude_code --arm native    --model sonnet --dataset $DS --limit 100
    python -m eval.agent.run_claude_code --arm sg-fusion --model sonnet --dataset $DS --limit 100

    # NIM react-loop
    python -m eval.agent.run_stage --stage v --only-arms fusion bm25 grep none hybrid --dataset $DS --limit 100

Verify (WSL + Docker — note the namespace, images are NOT under `swebench/`):

    python -m eval.agent.verify --all --run-tag <tag> \
        --dataset nebius/SWE-rebench-leaderboard \
        --namespace swerebench --hf-split 2026_03

Aggregate/compare/stats are unchanged — they read run JSONs, not the dataset.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "nebius/SWE-rebench-leaderboard"
# Anything strictly after a model's training cutoff is uncontaminated for it.
# Sonnet's cutoff is Jan 2026, so Feb/Mar 2026 are clean; adjust as new months
# land (--list-splits shows what exists) or as you evaluate other models.
DEFAULT_SPLITS = "2026_02,2026_03"
DEFAULT_SINCE = "2026-02-01"


def list_splits() -> None:
    try:
        from datasets import get_dataset_split_names
    except ImportError:
        sys.exit("ERROR: pip install datasets")
    names = get_dataset_split_names(DATASET)
    print(f"{DATASET} splits ({len(names)}):")
    from datasets import load_dataset
    for nm in names:
        try:
            n = len(load_dataset(DATASET, split=nm))
        except Exception:
            n = "?"
        print(f"  {nm:12} {n:>5} instances")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--splits", default=DEFAULT_SPLITS,
                    help=f"comma-separated monthly splits (default: {DEFAULT_SPLITS} "
                         "= post-Jan-2026, decontaminated for Sonnet)")
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help="drop instances created before this date (belt-and-braces "
                         "on top of --splits; set '' to disable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("eval/datasets/swe_rebench_100.jsonl"))
    ap.add_argument("--list-splits", action="store_true",
                    help="print available monthly splits + sizes, then exit")
    args = ap.parse_args()

    if args.list_splits:
        list_splits()
        return

    cmd = [
        sys.executable, "eval/make_dataset.py",
        "--split", DATASET,
        "--hf-split", args.splits,
        "--n", str(args.n),
        "--seed", str(args.seed),
        "--out", str(args.out),
    ]
    if args.since:
        cmd += ["--since", args.since]

    print("Building decontaminated SWE-rebench dataset")
    print(f"  dataset : {DATASET}")
    print(f"  splits  : {args.splits}   (since >= {args.since or 'n/a'})")
    print(f"  n       : {args.n}")
    print(f"  out     : {args.out}\n")
    print("  " + " ".join(cmd) + "\n")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(rc)

    print(f"\nNext — run either harness against {args.out}:")
    print(f'  python -m eval.agent.run_claude_code --arm native    --model sonnet --dataset {args.out} --limit {args.n}')
    print(f'  python -m eval.agent.run_claude_code --arm sg-fusion --model sonnet --dataset {args.out} --limit {args.n}')
    print(f'  python -m eval.agent.run_stage --stage v --only-arms fusion bm25 grep none hybrid --dataset {args.out} --limit {args.n}')
    print("\nVerify (WSL+Docker) — namespace matters, images are not under swebench/:")
    print(f'  python -m eval.agent.verify --all --run-tag <tag> \\\n'
          f'      --dataset {DATASET} --namespace swerebench --hf-split {args.splits.split(",")[-1]}')


if __name__ == "__main__":
    main()
