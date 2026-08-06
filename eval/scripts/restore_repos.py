"""Re-clone the source repos for an EXISTING dataset jsonl, without touching
the dataset itself.

Why this exists
---------------
`repo_path` in a built dataset points at `<SG_EVAL_DATA_ROOT>/repos/<task_id>`,
which is gitignored heavy IO (tens of GB). It routinely gets cleaned up between
runs. When it does, `run_claude_code --prepare-only` fails immediately:

    FileNotFoundError: source clone missing: .../eval/datasets/repos/<task_id>

The obvious recovery -- re-running make_rebench / make_dataset -- is WRONG for a
run already in flight. Those scripts RE-SAMPLE from the upstream HF dataset,
which for SWE-rebench gains monthly splits over time; the same --seed therefore
draws from a different pool and yields a different task list. That silently
strands every paired run already completed under the old list, and overwrites
any post-hoc dataset fixes (e.g. tighten_gold_fqns.py).

This script keeps the task list frozen and restores only the clones. It calls
the same `make_dataset.setup_repo()` used at build time, so the resulting
worktrees are byte-identical to what the dataset was built against.

Idempotent: repos already present are skipped, so re-run it freely after an
interruption. Safe to point at either the raw or the prose variant of a
dataset -- they share task_ids and therefore share clones, so restoring once
serves both.

`repo_path` pins the destination, not $SG_EVAL_DATA_ROOT
-----------------------------------------------------------
`setup_repo()` writes to `make_dataset.REPOS_DIR`, which is computed ONCE from
the `SG_EVAL_DATA_ROOT` env var at import time -- it has no idea what root the
dataset was actually BUILT with. If the current shell's env doesn't match
(e.g. an older dataset was built with SG_EVAL_DATA_ROOT set to a sibling
`swebench-data` dir, and this shell doesn't have it set), setup_repo clones
into the wrong place, reports success, and `--prepare-only` still fails with
"source clone missing" because it reads `repo_path` from the JSONL, not from
REPOS_DIR. This bit a real run (2026-08): 50 repos cloned successfully into
`eval/datasets/repos/`, useless, because `repo_path` said `swebench-data/repos/`.

Fixed by deriving the destination root from the dataset's OWN `repo_path`
(`repo_path = <root>/repos/<task_id>`, so `root = parent.parent`) and
redirecting `make_dataset.REPOS_DIR`/`CACHE_DIR` to it before cloning --
regardless of what the current shell's `SG_EVAL_DATA_ROOT` happens to be.

Usage:
    # restore exactly what a 50-task scale-up needs
    python -m eval.scripts.restore_repos --dataset eval/datasets/swe_rebench_100.jsonl --limit 50

    # check what's missing without cloning anything
    python -m eval.scripts.restore_repos --dataset eval/datasets/swe_rebench_100.jsonl --limit 50 --check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_dataset  # noqa: E402  (needs the path insert above)


def _pin_data_root(tasks: list) -> None:
    """Point make_dataset.REPOS_DIR/CACHE_DIR at the root the dataset's own
    repo_path fields actually expect, overriding whatever SG_EVAL_DATA_ROOT
    (or its absence) says in the CURRENT shell. All tasks in one dataset file
    must agree -- they were built in one make_dataset.py invocation."""
    roots = {Path(t["repo_path"]).resolve().parent.parent for t in tasks}
    if len(roots) > 1:
        sys.exit(f"tasks disagree on their data root ({sorted(map(str, roots))}) "
                  f"— this dataset file looks corrupted or hand-edited")
    root = roots.pop()
    if root != make_dataset.REPOS_DIR.parent:
        print(f"repo_path expects root {root} "
              f"(current SG_EVAL_DATA_ROOT resolves to "
              f"{make_dataset.REPOS_DIR.parent}) — redirecting clones there")
        make_dataset.REPOS_DIR = root / "repos"
        make_dataset.CACHE_DIR = root / "_repo_cache"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N tasks only (match the --limit you will run)")
    ap.add_argument("--check", action="store_true",
                    help="report present/missing and exit; clone nothing")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in
             args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        tasks = tasks[:args.limit]

    missing = [t for t in tasks if not Path(t["repo_path"]).is_dir()]
    print(f"{args.dataset.name}: {len(tasks)} tasks, "
          f"{len(tasks) - len(missing)} present, {len(missing)} missing")

    if args.check:
        for t in missing[:20]:
            print(f"  MISSING {t['task_id']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return
    if not missing:
        print("nothing to do")
        return

    _pin_data_root(tasks)

    t0 = time.time()
    ok, failed = 0, []
    for i, t in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {t['task_id']}")
        try:
            dest = make_dataset.setup_repo(t["repo"], t["base_commit"], t["task_id"])
        except Exception as e:  # noqa: BLE001 - report and continue
            dest, e_msg = None, str(e)
            print(f"  ! {e_msg}")
        if dest is None:
            failed.append(t["task_id"])
        else:
            ok += 1

    print(f"\nrestored {ok}/{len(missing)} in {time.time() - t0:.0f}s")
    if failed:
        # A task whose base_commit vanished upstream cannot be restored and must
        # be EXCLUDED from the run rather than silently re-sampled -- excluding
        # it keeps the arms paired, re-sampling does not.
        print(f"FAILED ({len(failed)}) — exclude these from the run, do not "
              f"re-sample the dataset:")
        for tid in failed:
            print(f"  {tid}")
        sys.exit(1)


if __name__ == "__main__":
    main()
