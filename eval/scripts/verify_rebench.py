"""One-command SWE-rebench verification for one or more run tags, with Docker
image retention so a LATER tag (typically the prose variant, same task_ids)
reuses every image the FIRST tag already pulled instead of re-downloading them.

Must run under WSL, not native Windows Python
-----------------------------------------------
`swebench.harness.__init__` unconditionally imports `prepare_images.py`, which
unconditionally does `import resource` (POSIX-only) -- confirmed directly:
`python -c "import swebench"` on native Windows raises
`ModuleNotFoundError: No module named 'resource'` before any of this script's
own code runs. This is why the earlier Verified verification also ran inside
WSL (see the docker_data.vhdx notes) and not in the same shell as the agent
runs. Launch this from a WSL shell, `cd` to the Windows-mounted repo path.

Two things "last time" needed that are easy to miss on a fresh call
--------------------------------------------------------------------
1. SWE-rebench needs `--namespace swerebench` (its own Docker Hub namespace)
   and a monthly `--hf-split`, unlike SWE-bench's single `test` split.
2. `eval.agent.verify` scans WHICH run directory to score from
   `SG_EVAL_RUN_TAG` (an env var baked in at import time by eval/agent/config.py)
   -- NOT from the `--run-tag` CLI flag, which only labels the harness's
   internal run_id/report filenames. Passing `--run-tag foo` alone silently
   verifies whatever tag the env var happens to already hold (often nothing),
   not `foo`. This script sets the env var per tag itself so that mismatch
   can't happen.

Split derivation: do NOT infer it from `created_at`
-----------------------------------------------------
An earlier version of this script derived --hf-split from each task's
`created_at` field, on the assumption that a task dated 2026-04 lives in the
'2026_04' monthly split. That assumption is FALSE and broke a real run:
`created_at` is the underlying GitHub issue's creation timestamp, and a task
can be archived into an EARLIER monthly split than its own created_at implies
(split lag) -- verified directly by checking a live 100-task sample's
instance_ids against the actual splits: 42 of them had created_at in
2026_04/2026_05, months that don't exist as splits on the Hub at all, yet
every one of the 100 was found sitting in 2026_02 or 2026_03.

Verified instead: `test` is the exact union of every monthly split (860
instances in `test` == the arithmetic sum of all 15 monthly splits' sizes),
so it is guaranteed to contain anything any monthly split does. This script
therefore just uses `test` -- no per-task month lookup, no risk of a lag
mismatch, and it needs no network probe beyond the harness's own dataset load.
If a future dataset revision ever splits `test` differently, `--hf-split`
below is the one place to override it; nothing else changes.

Docker image retention
-----------------------
The harness's own `clean_images()` (swebench/harness/docker_utils.py) removes
freshly-built `sweb.eval.*` (per-instance) images at the end of a run UNLESS
`--cache-level instance` is used -- at `env` (verify.py's own default) a
just-built instance image is deleted even though the CLI's `--clean` defaults
to False, because the removal condition is `clean OR not existed_before`, and
a brand-new image always has `existed_before == False`. So the raw run MUST
pass `--cache-level instance` for anything to survive for the prose run to
reuse. This script does that for every tag it verifies.

Usage
-----
    # WSL:
    python3 -m eval.scripts.verify_rebench \\
        --tag claude_rebench_v1 --tag claude_rebench_prose_v1

Run the RAW tag before the PROSE tag (list order matters only for pull
efficiency, not correctness) -- raw's cache-level=instance pull, once done,
is what prose then finds already on disk.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

DATASET = "nebius/SWE-rebench-leaderboard"
DEFAULT_SPLIT = "test"  # verified union of every monthly split -- see docstring


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-split", default=DEFAULT_SPLIT,
                    help=f"HF split to verify against (default '{DEFAULT_SPLIT}', "
                         "the verified union of every monthly split — see "
                         "docstring for why this replaced a per-task month guess)")
    ap.add_argument("--tag", action="append", required=True,
                    help="run tag to verify (repeatable; put the RAW tag "
                         "first so its cache-level=instance pull is what "
                         "later tags reuse)")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--incremental", action="store_true",
                    help="skip records already verified (cheap re-run as "
                         "more tasks finish)")
    args = ap.parse_args()

    if sys.platform.startswith("win"):
        sys.exit(
            "This must run under WSL: swebench.harness unconditionally "
            "imports the POSIX-only `resource` module. Open a WSL shell, "
            "cd to this repo's Windows-mounted path, and re-run there.")

    print(f"hf-split: {args.hf_split}\n")

    for tag in args.tag:
        print(f"=== verifying {tag} ===")
        cmd = [
            sys.executable, "-m", "eval.agent.verify", "--all",
            "--run-tag", tag,
            "--dataset", DATASET,
            "--namespace", "swerebench",
            "--hf-split", args.hf_split,
            "--cache-level", "instance",
            "--max-workers", str(args.max_workers),
        ]
        if args.incremental:
            cmd.append("--incremental")
        env = {**os.environ, "SG_EVAL_RUN_TAG": tag}
        subprocess.run(cmd, check=True, env=env)
        print()

    print("Done. Instance images were kept (cache-level=instance) so a "
          "later tag on the same task set skips the pull entirely.")
    print("Check what's retained with:  docker images | grep swerebench")
    print("These add up in docker_data.vhdx over time — the compaction "
          "ritual (quit Docker Desktop, `wsl --shutdown`, diskpart compact) "
          "must exclude BOTH `swerebench/*` and `swebench/sweb.eval.*` "
          "families if you still need SWE-Verified's images too.")


if __name__ == "__main__":
    main()
