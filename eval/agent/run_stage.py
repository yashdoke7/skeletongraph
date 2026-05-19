"""Stage orchestrator — run a whole stage, parallel workers, resumable.

    python -m eval.agent.run_stage --stage B
    python -m eval.agent.run_stage --stage B --workers 10
    python -m eval.agent.run_stage --stage B --probe        # 5 tasks only

Resumable: a (task, arm, model, repeat) whose result JSON already exists is
skipped, so a crashed stage just re-runs the command.

Each (task, arm) run is fully isolated (own workspace) so workers are safe to
run concurrently — see isolation.py. vLLM batches their requests server-side.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .isolation import run_id
from .run_agent import load_tasks, run_one


def _stage_jobs(stage: config.Stage, probe: bool) -> list:
    """Expand a stage into a flat list of (task, arm, model, repeat) jobs."""
    tasks = load_tasks()
    n = 5 if probe else stage.n_tasks
    tasks = tasks[:n]
    jobs = []
    for model in stage.models:
        for repeat in range(stage.repeats):
            for task in tasks:
                for arm in stage.arms:
                    jobs.append((task, arm, model, repeat))
    return jobs


def _already_done(task: dict, arm: str, model: str, repeat: int) -> bool:
    """A run counts as done only if it FINISHED cleanly — the agent called
    submit or exhausted MAX_TURNS. `error` (endpoint/exception) and `no_tool`
    (the model never produced a usable tool call) are incomplete: a re-run
    should retry them rather than skip. This makes the harness self-healing
    after a fix — no manual results purge needed."""
    rid = run_id(task["task_id"], arm, repeat, model)
    p = config.RUNS_DIR / f"{rid}.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False                      # corrupt/partial — re-run it
    return rec.get("stopped") in ("submit", "max_turns")


def run_stage(stage_name: str, workers: int = 8, probe: bool = False,
              force: bool = False) -> None:
    if stage_name not in config.STAGES:
        raise SystemExit(f"unknown stage {stage_name}; "
                         f"choose from {list(config.STAGES)}")
    stage = config.STAGES[stage_name]
    if stage.benchmark != "swebench":
        print(f"NOTE: stage {stage_name} uses benchmark '{stage.benchmark}'. "
              f"This harness currently drives SWE-bench. Wire the {stage.benchmark} "
              f"loader before running it (see STAGES.md).")

    jobs = _stage_jobs(stage, probe)
    if not force:
        jobs = [j for j in jobs if not _already_done(j[0], j[1], j[2], j[3])]

    print(f"Stage {stage_name}: {stage.note}")
    print(f"  arms={stage.arms} models={stage.models} repeats={stage.repeats}")
    print(f"  {len(jobs)} runs pending  ·  {workers} workers")
    if not jobs:
        print("  nothing to do (all results exist) — run aggregate.py + verify.py")
        return

    t0 = time.time()
    done = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, t, a, r, m): (t["task_id"], a, m, r)
                for (t, a, m, r) in jobs}
        for fut in as_completed(futs):
            tid, arm, model, rep = futs[fut]
            try:
                rec = fut.result()
                done += 1
                print(f"  [{done+fail}/{len(jobs)}] {rec['run_id']}: "
                      f"{rec['stopped']} turns={rec['n_turns']} "
                      f"hit={rec['retrieval_hit']} gold={rec['edited_gold_file']}")
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(jobs)}] {tid}/{arm}/{model}/r{rep} "
                      f"FAILED: {type(e).__name__}: {e}")

    mins = (time.time() - t0) / 60
    print(f"\nStage {stage_name} done: {done} ok, {fail} failed, {mins:.1f} min")
    print(f"Results in {config.RUNS_DIR}")
    print("Next:  python -m eval.agent.verify --stage " + stage_name)
    print("Then:  python -m eval.agent.aggregate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe", action="store_true", help="5-task timing probe")
    ap.add_argument("--force", action="store_true", help="re-run completed jobs")
    args = ap.parse_args()
    run_stage(args.stage, args.workers, args.probe, args.force)


if __name__ == "__main__":
    main()
