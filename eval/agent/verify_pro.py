"""SWE-bench Pro pass@1 — gather our agent patches into the scaleapi harness
format, audit that the harness actually executed, and write `resolved` back into
our run JSONs.

This is the Pro analog of verify.py. Pro pass@1 is NOT the official SWE-bench
harness — it uses scaleapi/SWE-bench_Pro-os with PREBUILT Docker images
(jefzda/sweap-images), so we don't build images. Flow:

  1. gather    — collect run JSONs (per arm) -> predictions JSON (list of
                 {instance_id, patch, prefix}) that swe_bench_pro_eval.py reads.
  2. (run the scaleapi harness on a Docker host — AMD/Linux/Modal; see
     eval/docs/PRO_PASS1_RUNBOOK.md). It writes per-instance output.json files
     plus an eval_results.json into --output_dir.
  3. check     — AUDIT the harness output dir: did the containers actually run?
                 An all-False eval_results.json with no per-instance output.json
                 means the harness NO-OP'd (e.g. Modal/Docker not available) —
                 that is "not evaluated", NOT a real 0% pass@1.
  4. writeback — set resolved=bool in each run JSON, but ONLY for instances the
                 harness actually executed. Refuses to write anything if there is
                 no execution evidence, so a null run can never be misread as 0%.

Our run `task_id` IS the Pro dataset `instance_id` (verified 99/99), so the
mapping is direct.

The execution-evidence rule (the whole point of `check`):
  An instance counts as EXECUTED iff the harness produced a per-instance
  `{prefix}_output.json` for it (that file is written only after the container
  ran the tests and the parser emitted JSON). `eval_results.json` alone is NOT
  trusted — main() in the harness writes `False` both for "tests failed" and for
  "container never ran / returned None", so False is ambiguous without the
  output.json evidence.

Usage:
  python -m eval.agent.verify_pro gather \
      --results eval/results/agent/nemotron_pro --arm fusion \
      --out eval/results/pro_preds_fusion.json
  # ... run the scaleapi harness (see runbook) ...
  python -m eval.agent.verify_pro check \
      --harness-dir /path/to/SWE-bench_Pro-os/results_fusion --arm fusion
  python -m eval.agent.verify_pro writeback \
      --results eval/results/agent/nemotron_pro --arm fusion \
      --harness-dir /path/to/SWE-bench_Pro-os/results_fusion
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_runs(results_dir: Path, arm: str) -> List[dict]:
    """All run JSONs for one arm in a results dir."""
    runs = []
    for fn in glob.glob(str(results_dir / f"*__{arm}__main__r0.json")):
        try:
            runs.append(json.load(open(fn, encoding="utf-8")))
        except Exception:
            pass
    return runs


def gather(results_dir: Path, arm: str, out: Path) -> Path:
    """Write predictions in the scaleapi format: a JSON LIST of
    {instance_id, patch, prefix}. Empty patches are dropped (the harness would
    mark them unresolved anyway; dropping keeps the file honest about coverage)."""
    runs = _load_runs(results_dir, arm)
    preds = []
    empty = 0
    for r in runs:
        patch = (r.get("model_patch") or "").strip()
        if not patch or "diff --git" not in patch:
            empty += 1
            continue
        preds.append({
            "instance_id": r["task_id"],   # == Pro dataset instance_id
            "patch": patch,
            "prefix": arm,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    print(f"  gathered {len(preds)} patches ({empty} empty/dropped) -> {out}")
    print(f"  NOTE: coverage = {len(preds)}/{len(runs)} runs had a usable patch.")
    return out


# ── execution-evidence audit ────────────────────────────────────────────────

def _instance_dirs(harness_dir: Path) -> List[Path]:
    """Per-instance subdirectories the harness creates (named by instance_id)."""
    return [p for p in harness_dir.iterdir()
            if p.is_dir() and p.name not in (".", "..")]


def _output_json_for(inst_dir: Path, arm: str) -> Optional[Path]:
    """The per-instance parser output, written ONLY after the container ran.
    Tries the prefixed name first ({arm}_output.json), then a couple of fallbacks
    so a differently-prefixed run still audits."""
    cands = [inst_dir / f"{arm}_output.json"]
    # any *_output.json (covers prefix="" or a different arm label)
    cands += sorted(inst_dir.glob("*_output.json"))
    cands += [inst_dir / "workspace" / "output.json"]
    for c in cands:
        if c.is_file() and c.stat().st_size > 0:
            return c
    return None


def _audit(harness_dir: Path, arm: str) -> Tuple[Dict[str, bool], List[str], List[str]]:
    """Return (resolved_by_id, executed_ids, not_executed_ids).

    resolved_by_id is computed from each EXECUTED instance's output.json + the
    harness eval_results.json. An instance is executed iff it has a non-empty
    *_output.json. Instances without one were never run (or the parser never
    produced JSON) and are reported separately — never silently counted False.
    """
    # Harness verdicts (ambiguous on their own — used only for executed ones).
    eval_results = {}
    er_path = harness_dir / "eval_results.json"
    if er_path.is_file():
        try:
            eval_results = json.loads(er_path.read_text(encoding="utf-8"))
        except Exception:
            eval_results = {}

    resolved: Dict[str, bool] = {}
    executed: List[str] = []
    not_executed: List[str] = []
    for inst in _instance_dirs(harness_dir):
        iid = inst.name
        oj = _output_json_for(inst, arm)
        if oj is None:
            not_executed.append(iid)
            continue
        executed.append(iid)
        # Prefer the harness's own verdict; fall back to recomputing from output.json
        # if eval_results is missing this id for some reason.
        if iid in eval_results:
            resolved[iid] = bool(eval_results[iid])
        else:
            resolved[iid] = _recompute_resolved(oj)
    return resolved, executed, not_executed


def _recompute_resolved(output_json: Path) -> bool:
    """Best-effort fallback: a run is resolved if every test reported PASSED.
    (The harness itself checks (f2p|p2p) <= passed; without the raw_sample here
    we approximate by "no non-PASSED test present", which is conservative.)"""
    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
        tests = data.get("tests", [])
        if not tests:
            return False
        return all(t.get("status") == "PASSED" for t in tests)
    except Exception:
        return False


def check(harness_dir: Path, arm: str) -> int:
    """Audit a harness output dir. Prints a clear REAL-RUN / NULL-RUN verdict.
    Exit code 0 = real run with at least one executed instance; 2 = null run."""
    if not harness_dir.is_dir():
        print(f"  ERROR: harness dir not found: {harness_dir}")
        return 2
    resolved, executed, not_executed = _audit(harness_dir, arm)
    total = len(executed) + len(not_executed)
    n_res = sum(1 for v in resolved.values() if v)
    print(f"  harness dir : {harness_dir}")
    print(f"  arm/prefix  : {arm}")
    print(f"  instances   : {total}")
    print(f"  EXECUTED    : {len(executed)}  (have a non-empty *_output.json)")
    print(f"  NOT executed: {len(not_executed)}  (no output.json — container never ran)")
    if executed:
        print(f"  resolved    : {n_res}/{len(executed)} "
              f"= {n_res / len(executed):.1%} pass@1 among executed")
    if not executed:
        print()
        print("  >>> NULL RUN — the harness produced NO output.json for any instance.")
        print("  >>> This is 'not evaluated', NOT a 0% pass@1. The Docker/Modal")
        print("  >>> containers did not run. On the AMD/Linux box run with")
        print("  >>> --use_local_docker and confirm a single instance first")
        print("  >>> (see eval/docs/PRO_PASS1_RUNBOOK.md, step 2a smoke test).")
        return 2
    if not_executed:
        print()
        print(f"  WARNING: {len(not_executed)} instance(s) were NOT executed; they")
        print("  will be left unevaluated (not counted as failures) on writeback.")
        for iid in not_executed[:5]:
            print(f"    - {iid}")
        if len(not_executed) > 5:
            print(f"    ... and {len(not_executed) - 5} more")
    print()
    print("  REAL RUN — execution evidence present. Safe to writeback.")
    return 0


# ── writeback (execution-gated) ──────────────────────────────────────────────

def writeback(results_dir: Path, arm: str,
              harness_dir: Optional[Path] = None,
              harness_report: Optional[Path] = None,
              force: bool = False) -> int:
    """Set resolved=bool in each run JSON — but only for instances the harness
    actually executed.

    Preferred: --harness-dir (the scaleapi --output_dir, e.g. results_fusion).
    We audit it for per-instance output.json evidence and REFUSE to write back if
    nothing executed (so a null run can't be misread as 0%). Only executed
    instances get a resolved bool; non-executed ones get resolved=null +
    pro_evaluated=false so aggregate can exclude them honestly.

    Legacy: --harness-report (a single {id:bool} json). Has no execution
    evidence, so it requires --force and is discouraged.
    """
    if harness_dir is not None:
        resolved, executed, not_executed = _audit(harness_dir, arm)
        if not executed and not force:
            print("  REFUSING to write back: NULL RUN (no instance produced an "
                  "output.json).")
            print("  The harness did not execute. See `check` above / the runbook.")
            print("  (Use --force only if you really intend to record all-unevaluated.)")
            return 2
        executed_set = set(executed)
        n = nres = nfail = nun = 0
        for fn in glob.glob(str(results_dir / f"*__{arm}__main__r0.json")):
            try:
                r = json.load(open(fn, encoding="utf-8"))
            except Exception:
                continue
            iid = r.get("task_id")
            if iid in executed_set:
                val = bool(resolved.get(iid, False))
                r["resolved"] = val
                r["pro_evaluated"] = True
                nres += int(val)
                nfail += int(not val)
            else:
                # Never executed — do not claim a result.
                r["resolved"] = None
                r["pro_evaluated"] = False
                nun += 1
            json.dump(r, open(fn, "w", encoding="utf-8"), indent=2)
            n += 1
        print(f"  wrote into {n} run JSONs for arm={arm}")
        print(f"    evaluated : {nres + nfail}  (resolved={nres}, unresolved={nfail})")
        print(f"    unevaluated (not executed): {nun}")
        if nres + nfail:
            print(f"    pass@1 among evaluated = {nres}/{nres + nfail} "
                  f"= {nres / (nres + nfail):.1%}")
        return 0

    # Legacy path — no execution evidence.
    if harness_report is None:
        print("  ERROR: provide --harness-dir (preferred) or --harness-report.")
        return 2
    if not force:
        print("  REFUSING: --harness-report has no execution evidence and can mask "
              "a null run.")
        print("  Use --harness-dir <output_dir> instead, or pass --force to override.")
        return 2
    resolved_set = _resolved_ids(harness_report)
    print(f"  [legacy/--force] harness reports {len(resolved_set)} resolved instances")
    n = 0
    for fn in glob.glob(str(results_dir / f"*__{arm}__main__r0.json")):
        try:
            r = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        r["resolved"] = r.get("task_id") in resolved_set
        json.dump(r, open(fn, "w", encoding="utf-8"), indent=2)
        n += 1
    print(f"  wrote resolved into {n} run JSONs for arm={arm}")
    return 0


def _resolved_ids(report: Path) -> set:
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("resolved_ids", "resolved", "resolved_instances"):
            if isinstance(data.get(key), list):
                return set(data[key])
        # mapping instance_id -> bool or {resolved: bool}
        out = set()
        for k, v in data.items():
            if isinstance(v, dict) and v.get("resolved"):
                out.add(k)
            elif v is True:
                out.add(k)
        return out
    if isinstance(data, list):
        return {d["instance_id"] for d in data
                if isinstance(d, dict) and d.get("resolved")}
    return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="SWE-bench Pro pass@1 bridge")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gather", help="collect run patches -> scaleapi preds json")
    g.add_argument("--results", type=Path, required=True)
    g.add_argument("--arm", required=True)
    g.add_argument("--out", type=Path, required=True)

    c = sub.add_parser("check", help="audit a harness output dir for real execution")
    c.add_argument("--harness-dir", type=Path, required=True,
                   help="the scaleapi --output_dir, e.g. results_fusion")
    c.add_argument("--arm", required=True, help="prefix used in gather (e.g. fusion)")

    w = sub.add_parser("writeback", help="set resolved in run JSONs (execution-gated)")
    w.add_argument("--results", type=Path, required=True)
    w.add_argument("--arm", required=True)
    w.add_argument("--harness-dir", type=Path, default=None,
                   help="preferred: the scaleapi --output_dir (audited for execution)")
    w.add_argument("--harness-report", type=Path, default=None,
                   help="legacy: a single {id:bool} json (requires --force)")
    w.add_argument("--force", action="store_true",
                   help="override the null-run / no-evidence refusal")

    a = ap.parse_args()
    if a.cmd == "gather":
        gather(a.results, a.arm, a.out)
    elif a.cmd == "check":
        sys.exit(check(a.harness_dir, a.arm))
    else:
        sys.exit(writeback(a.results, a.arm, a.harness_dir, a.harness_report, a.force))


if __name__ == "__main__":
    main()
