"""Point a frozen task file's repo_path fields at a directory on THIS machine.

The committed task files (eval/datasets/swebench_100.jsonl and friends) are the
exact task sets the study ran on, so they are kept verbatim — including the
repo_path each run used, which is an absolute path on the original machine. The
drivers (run_claude_code, run_codex, run_agent) and restore_repos all read
repo_path, so on any other machine write a relocated copy first:

    python -m eval.scripts.relocate_tasks --dataset eval/datasets/swebench_100.jsonl \
        --root D:/swebench-data
    # -> eval/datasets/swebench_100.local.jsonl, repo_path = D:/swebench-data/repos/<task>
    python -m eval.scripts.restore_repos --dataset eval/datasets/swebench_100.local.jsonl

Only repo_path changes; task ids, issue text, base commits and gold labels are
copied byte-for-byte. The .local.jsonl output is gitignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="frozen task file")
    ap.add_argument("--root", required=True, type=Path,
                    help="data root on this machine; clones go under <root>/repos/")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: <dataset>.local.jsonl next to the input)")
    args = ap.parse_args()

    out = args.out or args.dataset.with_name(args.dataset.stem + ".local.jsonl")
    root = args.root.resolve()
    rows = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    for r in rows:
        # The original paths are Windows paths; take the last component either way.
        name = PureWindowsPath(r["repo_path"]).name or Path(r["repo_path"]).name
        r["repo_path"] = str(root / "repos" / name)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} tasks -> {out}  (repo_path root: {root / 'repos'})")


if __name__ == "__main__":
    main()
