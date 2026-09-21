"""Docker-free tests for verify.py's glue logic.

The real pass@1 needs Docker + the official SWE-bench harness. The pieces that
silently break are pure data plumbing and ARE testable here: the predictions-file
format, parsing the harness report, writing verdicts back into run records, and
clearing the harness's cached per-task verdicts before a re-verify.

    python -m eval.agent.test_verify          # standalone
    python -m pytest eval/agent/test_verify.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import config, verify


def _rec(task_id: str, arm: str, patch: str) -> dict:
    rid = f"{task_id}__{arm}__main__r0"
    return {"run_id": rid, "task_id": task_id, "arm": arm,
            "model": "main", "repeat": 0, "model_patch": patch}


def test_write_predictions(tmp_path: Path) -> str:
    """One valid SWE-bench JSON object per run, named by arm (one file per arm)."""
    recs = [_rec("astropy__astropy-8707", "sg", "diff --git a/x b/x\n+fix"),
            _rec("django__django-14725", "sg", "")]   # empty patch is legal
    out = verify.write_predictions(recs, tmp_path / "_predictions.jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(recs), f"expected {len(recs)} lines, got {len(lines)}"
    for ln, rec in zip(lines, recs):
        obj = json.loads(ln)
        assert obj["instance_id"] == rec["task_id"], "instance_id wrong"
        assert obj["model_name_or_path"] == rec["arm"], "model_name should be the arm"
        assert obj["model_patch"] == rec["model_patch"], "patch not preserved"
        assert set(obj) == {"instance_id", "model_name_or_path", "model_patch"}
    return "write_predictions: valid SWE-bench JSONL, one arm per file"


def test_resolved_task_ids_both_shapes(tmp_path: Path) -> str:
    """The harness report is read in either {"resolved_ids": [...]} or
    {"resolved": [...]} shape; a corrupt report yields an empty set, not a crash."""
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"resolved_ids": ["t1", "t2"]}), encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"resolved": ["t3"]}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all {{{", encoding="utf-8")
    assert verify._resolved_task_ids(a) == {"t1", "t2"}
    assert verify._resolved_task_ids(b) == {"t3"}
    assert verify._resolved_task_ids(bad) == set()
    return "_resolved_task_ids: both schema shapes parsed, corrupt file -> empty"


def test_apply_results_writes_verdicts(tmp_path: Path) -> str:
    """apply_results writes True/False into each run record by task_id."""
    orig = config.RUNS_DIR
    config.RUNS_DIR = tmp_path
    try:
        recs = [_rec("astropy__astropy-8707", "sg", "patch"),
                _rec("django__django-14725", "sg", "patch")]
        for r in recs:
            (tmp_path / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")
        recs[0]["_path"] = "scratch"                 # must not be persisted
        verify.apply_results(recs, {"astropy__astropy-8707"})
        a = json.loads((tmp_path / "astropy__astropy-8707__sg__main__r0.json").read_text(encoding="utf-8"))
        d = json.loads((tmp_path / "django__django-14725__sg__main__r0.json").read_text(encoding="utf-8"))
        assert a["resolved"] is True, "resolved run not marked True"
        assert d["resolved"] is False, "unresolved run not marked False"
        assert "_path" not in a, "_path scratch key leaked into saved JSON"
    finally:
        config.RUNS_DIR = orig
    return "apply_results: verdicts written back by task_id"


def test_drop_stale_logs_only_touches_the_batch(tmp_path: Path) -> str:
    """A re-verify clears the cached per-task verdicts for exactly the runs being
    scored, so a re-run task is never scored against its old patch; other tasks'
    logs are left alone."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        base = Path("logs/run_evaluation/tag_sg/sg")
        for t in ("t1", "t2", "t3"):
            (base / t).mkdir(parents=True)
            (base / t / "report.json").write_text("{}", encoding="utf-8")
        n = verify._drop_stale_logs([{"task_id": "t1"}, {"task_id": "t2"}], "tag_sg", "sg")
        assert n == 2, f"expected 2 cleared, got {n}"
        assert not (base / "t1").exists() and not (base / "t2").exists()
        assert (base / "t3" / "report.json").exists(), "a task outside the batch was cleared"
        assert verify._drop_stale_logs([{"task_id": "t1"}], "no_such_tag", "sg") == 0
    finally:
        os.chdir(cwd)
    return "_drop_stale_logs: clears the batch, leaves everything else"


_TESTS = [
    test_write_predictions,
    test_resolved_task_ids_both_shapes,
    test_apply_results_writes_verdicts,
    test_drop_stale_logs_only_touches_the_batch,
]


def main() -> None:
    passed = failed = 0
    for fn in _TESTS:
        with tempfile.TemporaryDirectory() as td:
            try:
                msg = fn(Path(td))
                print(f"  PASS  {fn.__name__}: {msg}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {fn.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
