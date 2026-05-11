#!/usr/bin/env python3
"""
compare_patches.py — Compare native and SG patches against golden patch.

Usage:
    python eval/scripts/compare_patches.py \
        --task-id requests-1142 \
        --run-dir eval/runs/claude_code/requests-1142 \
        --output eval/runs/claude_code/requests-1142/patch_comparison.json
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


def extract_changed_files(patch: str) -> set:
    """Extract set of files changed in a patch."""
    files = set()
    for line in patch.splitlines():
        # Match: +++ b/path/to/file or --- a/path/to/file
        m = re.match(r'^[+-]{3} [ab]/(.+)$', line)
        if m:
            f = m.group(1)
            if f != "/dev/null":
                files.add(f)
    return files


def extract_changed_functions(patch: str) -> set:
    """Extract function/method names mentioned in patch context lines."""
    functions = set()
    for line in patch.splitlines():
        # Match @@ context line which often has function name
        m = re.search(r'@@[^@]+@@\s+(.+)', line)
        if m:
            ctx = m.group(1).strip()
            fn_match = re.match(r'(?:def|class)\s+(\w+)', ctx)
            if fn_match:
                functions.add(fn_match.group(1))
    return functions


def score_patch(patch: str, golden: str) -> str:
    """Score a patch against the golden patch."""
    if not patch.strip():
        return "empty"

    patch_files = extract_changed_files(patch)
    golden_files = extract_changed_files(golden)

    if not patch_files:
        return "empty"

    if not golden_files:
        # No golden patch — cannot score against it
        return "no_golden"

    if patch_files == golden_files:
        # Same files — check if diff is semantically similar
        patch_fns = extract_changed_functions(patch)
        golden_fns = extract_changed_functions(golden)
        if patch_fns & golden_fns:
            return "exact"  # Same files, same functions
        return "partial"  # Same files, different area

    if patch_files & golden_files:
        return "partial"  # Overlapping files

    return "wrong"  # Different files entirely


def run_tests(test_cmd: str, patch_path: Path, repo_dir: Path) -> bool:
    """Apply patch, run tests, return pass/fail, reset."""
    if not patch_path.exists() or not patch_path.read_text().strip():
        return None

    try:
        # Apply patch
        result = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            # Try with --reject
            subprocess.run(
                ["git", "apply", "--reject", str(patch_path)],
                cwd=repo_dir, capture_output=True
            )

        # Run tests
        test_result = subprocess.run(
            test_cmd.split(),
            cwd=repo_dir,
            capture_output=True, text=True,
            timeout=120
        )
        passed = test_result.returncode == 0

    except subprocess.TimeoutExpired:
        passed = None
    except Exception:
        passed = None
    finally:
        # Always reset
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=repo_dir, capture_output=True
        )

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir

    # Load patches
    native_patch = (run_dir / "native.patch").read_text() if (run_dir / "native.patch").exists() else ""
    sg_patch = (run_dir / "sg.patch").read_text() if (run_dir / "sg.patch").exists() else ""
    golden_patch = (run_dir / "golden.patch").read_text() if (run_dir / "golden.patch").exists() else ""

    # Load task config for test command
    task_json_path = run_dir / "task.json"
    test_cmd = None
    if task_json_path.exists():
        task_cfg = json.loads(task_json_path.read_text())
        test_cmd = task_cfg.get("test_cmd")

    # Score patches
    native_score = score_patch(native_patch, golden_patch)
    sg_score = score_patch(sg_patch, golden_patch)

    # Test results (from pre-run test execution if available)
    native_test = None
    sg_test = None
    if (run_dir / "native_test_result.txt").exists():
        native_test = (run_dir / "native_test_result.txt").read_text().strip() == "PASS"
    if (run_dir / "sg_test_result.txt").exists():
        sg_test = (run_dir / "sg_test_result.txt").read_text().strip() == "PASS"

    result = {
        "task_id": args.task_id,
        "native_patch_score": native_score,
        "sg_patch_score": sg_score,
        "native_test_passed": native_test,
        "sg_test_passed": sg_test,
        "sg_regression": sg_score in ("wrong", "empty") and native_score not in ("wrong", "empty"),
        "sg_improvement": sg_score in ("exact", "partial") and native_score in ("wrong", "empty"),
        "native_files_changed": sorted(extract_changed_files(native_patch)),
        "sg_files_changed": sorted(extract_changed_files(sg_patch)),
        "golden_files_expected": sorted(extract_changed_files(golden_patch)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    n = f"{native_score}" + (f" ({'✓' if native_test else '✗' if native_test is False else '?'})" if native_test is not None else "")
    s = f"{sg_score}" + (f" ({'✓' if sg_test else '✗' if sg_test is False else '?'})" if sg_test is not None else "")
    print(f"  Native: {n}")
    print(f"  SG:     {s}")
    if result["sg_regression"]:
        print(f"  ⚠ REGRESSION")


if __name__ == "__main__":
    main()

