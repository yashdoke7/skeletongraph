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


def _run_records(stage: str | None) -> list:
    """Load run JSONs, optionally filtered to one stage's arms/models."""
    records = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        r["_path"] = str(p)
        if stage and stage in config.STAGES:
            st = config.STAGES[stage]
            if r.get("arm") not in st.arms or r.get("model") not in st.models:
                continue
        records.append(r)
    return records


def write_predictions(records: list, out: Path) -> Path:
    """SWE-bench predictions file: one JSON object per (run) line.

    instance_id is suffixed with the run_id so every arm/repeat is a distinct
    prediction the harness can score independently.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "instance_id": r["task_id"],
                "model_name_or_path": r["run_id"],   # unique per arm/repeat
                "model_patch": r.get("model_patch", ""),
            }) + "\n")
    return out


def run_harness(predictions: Path, run_tag: str) -> Path:
    """Invoke the official SWE-bench harness. Returns its results JSON path."""
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(predictions),
        "--run_id", run_tag,
        "--max_workers", "4",
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # the harness writes <model_name>.<run_id>.json in CWD; with mixed
    # model_name_or_path it writes per-prediction reports under logs/run_evaluation.
    candidates = sorted(Path(".").glob(f"*{run_tag}*.json"))
    return candidates[-1] if candidates else Path("logs/run_evaluation")


def apply_results(records: list, results_path: Path) -> None:
    """Read the harness verdict and write `resolved` back into each run JSON.

    The harness report maps instance keys -> resolved bool. Exact schema varies
    by swebench version; this reads the common {"resolved_ids": [...]} shape and
    falls back to a per-instance scan. VALIDATE the shape on-box once.
    """
    resolved_ids: set = set()
    try:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            resolved_ids = set(data.get("resolved_ids", [])
                               or data.get("resolved", []))
    except Exception as e:
        print(f"  WARN: could not parse {results_path}: {e}")
        print("  Inspect logs/run_evaluation/ and set `resolved` manually, or "
              "adjust apply_results() to the harness version's schema.")

    for r in records:
        key_run = r["run_id"]
        key_task = r["task_id"]
        resolved = key_run in resolved_ids or key_task in resolved_ids
        r["resolved"] = bool(resolved)
        r.pop("_path", None)
        Path(config.RUNS_DIR / f"{r['run_id']}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None, help="verify one stage's runs")
    ap.add_argument("--all", action="store_true", help="verify every run")
    ap.add_argument("--run-tag", default="sg_eval")
    args = ap.parse_args()

    stage = None if args.all else args.stage
    records = _run_records(stage)
    if not records:
        raise SystemExit("no run JSONs found — run run_stage.py first")

    preds = write_predictions(records, config.RUNS_DIR / "_predictions.jsonl")
    print(f"Wrote {len(records)} predictions -> {preds}")
    results = run_harness(preds, args.run_tag)
    apply_results(records, results)
    n_ok = sum(1 for r in records if r.get("resolved"))
    print(f"Verified: {n_ok}/{len(records)} resolved. `resolved` written back.")


if __name__ == "__main__":
    main()
