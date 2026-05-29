"""Patch verification — the only honest source of pass@1.

The agent's `model_patch` is checked by the OFFICIAL SWE-bench evaluation
harness: it builds each task's environment, applies the patch, and runs the
task's FAIL_TO_PASS / PASS_TO_PASS tests. We do not re-implement that — we emit
a predictions file in SWE-bench format and shell out to the harness.

    python -m eval.agent.verify --stage B
    python -m eval.agent.verify --all

Requires:  pip install swebench   ·   Docker running (the harness needs it).

Output: writes pass/fail back into each run JSON as `resolved` (bool) and
`verify_detail`, so aggregate.py can compute pass@1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import config


def _run_records(stage: str | None,
                 only_arms: set | None = None,
                 skip_arms: set | None = None,
                 only_completed: bool = True,
                 only_unverified: bool = False) -> list:
    """Load run JSONs, optionally filtered by stage / arm allowlist / state.

    only_completed=True (default): drop records where stopped != submit / max_turns.
        Pass=False if you intentionally want to verify error-state records too
        (the harness will mark them unresolved, but it lets you see the full set).
    only_unverified=True: drop records that already have `resolved` set.
        Use this for incremental verify — skips arms/tasks already scored.
    """
    records = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == "summary.json":
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        r["_path"] = str(p)
        arm = r.get("arm")
        if stage and stage in config.STAGES:
            st = config.STAGES[stage]
            if arm not in st.arms or r.get("model") not in st.models:
                continue
        if only_arms and arm not in only_arms:
            continue
        if skip_arms and arm in skip_arms:
            continue
        if only_completed and r.get("stopped") not in ("submit", "max_turns"):
            continue
        if only_unverified and ("resolved" in r):
            continue
        records.append(r)
    return records


def write_predictions(records: list, out: Path) -> Path:
    """SWE-bench predictions file for ONE arm: one JSON object per line.

    The harness keys predictions by instance_id and expects ONE prediction per
    instance_id per run. So we write one file PER ARM (model_name_or_path is the
    arm), each containing instance_id=task_id. This is why verify loops arms —
    a single mixed file would collapse all arms to one verdict per task.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "instance_id": r["task_id"],
                "model_name_or_path": r.get("arm", "sg"),
                "model_patch": r.get("model_patch", ""),
            }) + "\n")
    return out


def run_harness(predictions: Path, run_tag: str, dataset: str) -> Path:
    """Invoke the official SWE-bench harness. Returns its results JSON path."""
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--predictions_path", str(predictions),
        "--run_id", run_tag,
        "--max_workers", "4",
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # the harness writes <model_name>.<run_id>.json in CWD.
    candidates = sorted(Path(".").glob(f"*{run_tag}*.json"))
    return candidates[-1] if candidates else Path("logs/run_evaluation")


def _resolved_task_ids(results_path: Path) -> set:
    """Parse the harness report → set of resolved instance_ids (task_ids)."""
    try:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  WARN: could not parse {results_path}: {e}")
        print("  Inspect logs/run_evaluation/ and set `resolved` manually, or "
              "adjust _resolved_task_ids() to the harness version's schema.")
        return set()
    if isinstance(data, dict):
        return set(data.get("resolved_ids", []) or data.get("resolved", []))
    return set()


def apply_results(records: list, resolved_ids: set) -> None:
    """Write `resolved` back into each run JSON for ONE arm's records.

    resolved_ids are task_ids resolved BY THIS ARM (the harness ran on a
    single-arm predictions file), so matching by task_id is now correct —
    different arms no longer share a verdict.
    """
    for r in records:
        r["resolved"] = bool(r["task_id"] in resolved_ids)
        r.pop("_path", None)
        Path(config.RUNS_DIR / f"{r['run_id']}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="verify one stage's runs")
    ap.add_argument("--all", action="store_true", help="verify every run")
    ap.add_argument("--only-arms", default="",
                    help="comma-separated arm names; verify only these arms "
                         "(e.g. cbmem,aider). Lets you re-verify one arm "
                         "while others are still running.")
    ap.add_argument("--skip-arms", default="",
                    help="comma-separated arm names; exclude these arms "
                         "(complement of --only-arms).")
    ap.add_argument("--incremental", action="store_true",
                    help="skip records that already have `resolved` set; "
                         "verify only newly-completed runs. Cheap re-runs "
                         "as more tasks finish.")
    ap.add_argument("--run-tag", default="sg_eval")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified",
                    help="HF dataset the harness scores against")
    args = ap.parse_args()

    stage = None if args.all else args.stage
    only = {a.strip() for a in args.only_arms.split(",") if a.strip()} or None
    skip = {a.strip() for a in args.skip_arms.split(",") if a.strip()} or None
    records = _run_records(stage, only, skip,
                           only_completed=True,
                           only_unverified=args.incremental)
    if not records:
        msg = "no run JSONs match the filter"
        if args.incremental:
            msg += " (incremental: every matching run is already verified)"
        raise SystemExit(msg)

    # Group by arm and verify each arm independently — one harness run per arm,
    # so each task's verdict is attributed to the correct arm.
    by_arm: dict = {}
    for r in records:
        by_arm.setdefault(r.get("arm", "sg"), []).append(r)

    total_ok = total = 0
    for arm, arm_recs in sorted(by_arm.items()):
        preds = write_predictions(arm_recs,
                                  config.RUNS_DIR / f"_predictions_{arm}.jsonl")
        tag = f"{args.run_tag}_{arm}"
        print(f"[{arm}] {len(arm_recs)} predictions -> {preds}")
        results = run_harness(preds, tag, args.dataset)
        resolved = _resolved_task_ids(results)
        apply_results(arm_recs, resolved)
        n_ok = sum(1 for r in arm_recs if r.get("resolved"))
        total_ok += n_ok
        total += len(arm_recs)
        print(f"[{arm}] resolved {n_ok}/{len(arm_recs)}")

    print(f"\nVerified: {total_ok}/{total} resolved across "
          f"{len(by_arm)} arms. `resolved` written back. Run aggregate for pass@1.")


if __name__ == "__main__":
    main()
