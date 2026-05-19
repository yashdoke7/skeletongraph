"""Docker-free smoke test for verify.py's glue logic.

The real pass@1 needs Docker + the official SWE-bench harness, which we cannot
run on a dev box. But the two pieces that silently break — the predictions-file
FORMAT and the verdict WRITE-BACK (apply_results' schema matching) — are pure
data plumbing and ARE testable here. If either is wrong you would only find out
mid-32B-run, after spending the compute. This catches it in one second.

    python -m eval.agent.test_verify
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import config, verify


def _rec(task_id: str, arm: str, patch: str) -> dict:
    rid = f"{task_id}__{arm}__main__r0"
    return {"run_id": rid, "task_id": task_id, "arm": arm,
            "model": "main", "repeat": 0, "model_patch": patch}


def test_write_predictions(tmp: Path) -> str:
    """write_predictions emits one valid SWE-bench JSON object per run."""
    recs = [_rec("astropy__astropy-8707", "sg", "diff --git a/x b/x\n+fix"),
            _rec("django__django-14725", "bm25", "")]   # empty patch is legal
    out = verify.write_predictions(recs, tmp / "_predictions.jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(recs), f"expected {len(recs)} lines, got {len(lines)}"
    for ln, rec in zip(lines, recs):
        obj = json.loads(ln)                       # must be valid JSON
        assert obj["instance_id"] == rec["task_id"], "instance_id wrong"
        assert obj["model_name_or_path"] == rec["run_id"], "model_name wrong"
        assert obj["model_patch"] == rec["model_patch"], "patch not preserved"
        assert set(obj) == {"instance_id", "model_name_or_path",
                            "model_patch"}, "unexpected keys in prediction"
    return "write_predictions: valid SWE-bench JSONL, patches preserved"


def test_apply_results_resolved_ids(tmp: Path) -> str:
    """apply_results writes the harness verdict back into each run JSON."""
    orig = config.RUNS_DIR
    config.RUNS_DIR = tmp
    try:
        recs = [_rec("astropy__astropy-8707", "sg", "patch"),
                _rec("django__django-14725", "sg", "patch")]
        for r in recs:                              # apply_results overwrites these
            (tmp / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")

        # harness output: only the astropy run resolved (keyed by run_id)
        results = tmp / "harness_results.json"
        results.write_text(json.dumps(
            {"resolved_ids": ["astropy__astropy-8707__sg__main__r0"]}),
            encoding="utf-8")

        verify.apply_results(recs, results)

        a = json.loads((tmp / "astropy__astropy-8707__sg__main__r0.json")
                       .read_text(encoding="utf-8"))
        d = json.loads((tmp / "django__django-14725__sg__main__r0.json")
                       .read_text(encoding="utf-8"))
        assert a["resolved"] is True, "resolved run not marked True"
        assert d["resolved"] is False, "unresolved run not marked False"
        assert "_path" not in a, "_path scratch key leaked into saved JSON"
    finally:
        config.RUNS_DIR = orig
    return "apply_results: resolved_ids verdict written back correctly"


def test_apply_results_task_id_fallback(tmp: Path) -> str:
    """apply_results also matches when the harness keys by task_id, not run_id,
    and accepts the alternate {"resolved": [...]} schema shape."""
    orig = config.RUNS_DIR
    config.RUNS_DIR = tmp
    try:
        rec = _rec("pallets__flask-5014", "sg", "patch")
        (tmp / f"{rec['run_id']}.json").write_text(json.dumps(rec), encoding="utf-8")

        results = tmp / "harness_results2.json"
        results.write_text(json.dumps(
            {"resolved": ["pallets__flask-5014"]}),   # task_id key + alt shape
            encoding="utf-8")

        verify.apply_results([rec], results)
        saved = json.loads((tmp / f"{rec['run_id']}.json")
                           .read_text(encoding="utf-8"))
        assert saved["resolved"] is True, "task_id fallback match failed"
    finally:
        config.RUNS_DIR = orig
    return "apply_results: task_id fallback + alternate schema shape handled"


def test_apply_results_unparseable(tmp: Path) -> str:
    """A corrupt/unexpected harness file must not crash — every run gets a
    definite False verdict so aggregate.py never reads a stale value."""
    orig = config.RUNS_DIR
    config.RUNS_DIR = tmp
    try:
        rec = _rec("mwaskom__seaborn-3069", "sg", "patch")
        (tmp / f"{rec['run_id']}.json").write_text(json.dumps(rec), encoding="utf-8")

        bad = tmp / "garbage.json"
        bad.write_text("not json at all {{{", encoding="utf-8")

        verify.apply_results([rec], bad)              # must not raise
        saved = json.loads((tmp / f"{rec['run_id']}.json")
                           .read_text(encoding="utf-8"))
        assert saved["resolved"] is False, "corrupt results should yield False"
    finally:
        config.RUNS_DIR = orig
    return "apply_results: corrupt harness file degrades safely to False"


_TESTS = [
    test_write_predictions,
    test_apply_results_resolved_ids,
    test_apply_results_task_id_fallback,
    test_apply_results_unparseable,
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
