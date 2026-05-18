"""run_all.py — ONE command runs the whole staged evaluation end to end.

Per stage:  run agents -> verify (optional) -> aggregate -> checkpoint to git.

  # AMD 32B: start vLLM (bash eval/agent/serve_model.sh), then:
  python -m eval.agent.run_all --stages 1-core,2-strong,2-ablation,2-variance \
      --workers 8 --verify --push

  # Local 7B smoke (RunPod unavailable): start Ollama, then (PowerShell):
  $env:SG_EVAL_MODEL="qwen2.5-coder:7b"
  $env:SG_EVAL_API_BASE="http://localhost:11434/v1"
  python -m eval.agent.run_all --stages 1-core --probe

Only the model endpoint differs between local and AMD — set SG_EVAL_MODEL and
SG_EVAL_API_BASE, nothing else changes. Resumable: completed (task,arm) runs are
skipped, so a re-run continues where it stopped.

Stage readiness (May 2026): `1-core` (sg/bm25/grep/none) and `3-scale` (sg/bm25)
run fully today. `2-strong`/`2-variance` need the hybrid + aider backends and
`2-ablation` needs the SG ablation toggles — all flagged NotImplementedError
until built (see eval/agent/tools.py). The orchestrator runs whatever is ready
and aggregates the rest gracefully.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

from . import config

REPO_ROOT = config.REPO_ROOT
DEFAULT_ORDER = ["1-core", "2-strong", "2-ablation", "2-variance", "3-scale"]


def _preflight(verify: bool) -> None:
    """Fail fast and clearly before spending GPU time."""
    url = config.API_BASE.rstrip("/") + "/models"
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f"  model endpoint OK: {config.API_BASE}")
    except Exception as e:
        raise SystemExit(
            f"Model endpoint unreachable at {config.API_BASE} ({e}).\n"
            f"  AMD:   bash eval/agent/serve_model.sh\n"
            f"  Local: install Ollama, then `ollama pull qwen2.5-coder:7b` "
            f"(Ollama serves on :11434 automatically)")

    if not config.DATASET.exists():
        raise SystemExit(
            f"Dataset missing: {config.DATASET}\n"
            f"  Build it:  python eval/make_dataset.py --n 150\n"
            f"  (a 30-task stage0.jsonl from Stage 0 also works for --probe)")

    if verify:
        try:
            ok = subprocess.run(["docker", "--version"],
                                capture_output=True).returncode == 0
        except FileNotFoundError:
            ok = False
        if not ok:
            raise SystemExit("--verify needs Docker (the SWE-bench harness). "
                             "Start Docker, or drop --verify for a smoke run.")
    print("  preflight OK")


def _run(cmd: list) -> int:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def _checkpoint(stage: str) -> None:
    """Force-add the (gitignored) eval results and push — a per-stage checkpoint
    so a crash mid-run never loses an earlier stage's results."""
    try:
        subprocess.run(["git", "add", "-f", "eval/results/agent"],
                       cwd=str(REPO_ROOT), check=True, capture_output=True)
        commit = subprocess.run(
            ["git", "commit", "-m", f"eval: results checkpoint after stage {stage}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True)
        if commit.returncode != 0:
            print(f"  (nothing new to commit for stage {stage})")
            return
        push = subprocess.run(["git", "push"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True)
        if push.returncode == 0:
            print(f"  pushed results checkpoint for stage {stage}")
        else:
            print(f"  WARN: push failed ({push.stderr.strip()[:160]}) — "
                  f"results are committed locally; push manually later")
    except Exception as e:
        print(f"  WARN: git checkpoint failed ({e}) — results are on disk regardless")


def run_all(stages, workers: int, probe: bool, verify: bool, push: bool) -> None:
    py = sys.executable
    t0 = time.time()
    for stage in stages:
        print(f"\n{'=' * 64}\nSTAGE {stage}  ·  {config.STAGES[stage].note}\n{'=' * 64}")

        # 1. run the agents for this stage (resumable; per-run isolation)
        cmd = [py, "-m", "eval.agent.run_stage", "--stage", stage,
               "--workers", str(workers)]
        if probe:
            cmd.append("--probe")
        if _run(cmd) != 0:
            print(f"  stage {stage}: run_stage reported errors — "
                  f"continuing to aggregate whatever completed")

        # 2. verify against the official SWE-bench harness (optional — Docker)
        if verify:
            _run([py, "-m", "eval.agent.verify", "--stage", stage])

        # 3. aggregate -> SUMMARY.md + metric tables
        _run([py, "-m", "eval.agent.aggregate", "--stage", stage])

        # 4. checkpoint results to git
        if push:
            _checkpoint(stage)

    mins = (time.time() - t0) / 60
    print(f"\n{'=' * 64}\nALL STAGES DONE — {mins:.1f} min")
    print(f"Results:  {config.RUNS_DIR}")
    print(f"Summary:  {config.RUNS_DIR / 'SUMMARY.md'}\n{'=' * 64}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole staged eval end to end")
    ap.add_argument("--stages", default="1-core",
                    help="comma-separated stage keys, in order (default: 1-core)")
    ap.add_argument("--all", action="store_true",
                    help=f"run every stage: {DEFAULT_ORDER}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe", action="store_true",
                    help="5-task probe per stage (use for local smoke / timing)")
    ap.add_argument("--verify", action="store_true",
                    help="run SWE-bench verification after each stage (needs Docker)")
    ap.add_argument("--push", action="store_true",
                    help="commit + push eval/results to git after each stage")
    args = ap.parse_args()

    stages = (DEFAULT_ORDER if args.all
              else [s.strip() for s in args.stages.split(",") if s.strip()])
    unknown = [s for s in stages if s not in config.STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; "
                         f"choose from {list(config.STAGES)}")

    print(f"Stages:   {stages}")
    print(f"Model:    {config.MODEL_NAME}")
    print(f"Endpoint: {config.API_BASE}")
    _preflight(args.verify)
    run_all(stages, args.workers, args.probe, args.verify, args.push)


if __name__ == "__main__":
    main()
