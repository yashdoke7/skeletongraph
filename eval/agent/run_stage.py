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


def _run_one_with_key(task: dict, arm: str, repeat: int, model: str,
                      key_idx: int) -> dict:
    """Wrapper that assigns a per-job NIM API key before calling run_one.

    key_idx is the job's position in the pending list; config._NIM_KEYS is
    indexed modulo the number of available keys. This ensures round-robin
    distribution across all NIM accounts. If _NIM_KEYS is empty (single-account
    or vLLM), the call is a no-op and falls back to the global API_KEY.
    """
    if config._NIM_KEYS:
        config.set_thread_api_key(config._NIM_KEYS[key_idx % len(config._NIM_KEYS)])
    try:
        return run_one(task, arm, repeat, model)
    finally:
        config.set_thread_api_key(None)   # release so no stale key lingers


def _stage_jobs(stage: config.Stage, probe: bool, limit: int = 0,
                dataset: str = "",
                only_arms: set | None = None,
                skip_arms: set | None = None) -> list:
    """Expand a stage into a flat list of (task, arm, model, repeat) jobs.

    probe → first 5 tasks; limit>0 → first N tasks (overrides probe); else the
    stage's n_tasks. Tasks are deterministic (dataset order) so a 10-task 14B
    run hits the SAME first 10 tasks as the 7B run for a clean comparison.
    dataset → load tasks from a non-default jsonl (e.g. contextbench.jsonl).
    only_arms / skip_arms → narrow the stage's arm list at the CLI without
    redefining the stage. Useful when an arm has different concurrency needs
    (cbmem/graphify are CPU-bound; the rest GPU-bound) and you want to fire
    them with different --workers.
    """
    from pathlib import Path
    tasks = load_tasks(Path(dataset)) if dataset else load_tasks()
    if limit and limit > 0:
        n = limit
    elif probe:
        n = 5
    else:
        n = stage.n_tasks
    tasks = tasks[:n]
    arms = stage.arms
    if only_arms:
        arms = [a for a in arms if a in only_arms]
    if skip_arms:
        arms = [a for a in arms if a not in skip_arms]
    if not arms:
        return []
    jobs = []
    for model in stage.models:
        for repeat in range(stage.repeats):
            for task in tasks:
                for arm in arms:
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
              force: bool = False, limit: int = 0, dataset: str = "",
              only_arms: set | None = None,
              skip_arms: set | None = None) -> None:
    if stage_name not in config.STAGES:
        raise SystemExit(f"unknown stage {stage_name}; "
                         f"choose from {list(config.STAGES)}")
    stage = config.STAGES[stage_name]
    if stage.benchmark != "swebench" and not dataset:
        print(f"NOTE: stage {stage_name} uses benchmark '{stage.benchmark}'. "
              f"Pass --dataset eval/datasets/{stage.benchmark}.jsonl if not "
              f"already overridden.")

    jobs = _stage_jobs(stage, probe, limit, dataset, only_arms, skip_arms)
    if not force:
        jobs = [j for j in jobs if not _already_done(j[0], j[1], j[2], j[3])]

    print(f"Stage {stage_name}: {stage.note}")
    print(f"  arms={stage.arms} models={stage.models} repeats={stage.repeats}")
    print(f"  {len(jobs)} runs pending  ·  {workers} workers")
    if not jobs:
        print("  nothing to do (all results exist) — run aggregate.py + verify.py")
        return

    n_keys = len(config._NIM_KEYS)
    if n_keys:
        print(f"  Multi-account NIM: {n_keys} API keys → "
              f"~{n_keys}× rate-limit headroom  "
              f"(SG_EVAL_API_KEYS)")
    else:
        print(f"  Single-account mode (SG_EVAL_API_KEYS not set)")

    t0 = time.time()
    done = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_one_with_key, t, a, r, m, i): (t["task_id"], a, m, r)
                for i, (t, a, m, r) in enumerate(jobs)}
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
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N tasks (e.g. 10 for a quick 14B/NIM read)")
    ap.add_argument("--dataset", default="",
                    help="tasks jsonl to use instead of the default (e.g. "
                         "eval/datasets/contextbench.jsonl)")
    ap.add_argument("--only-arms", default="",
                    help="comma-separated arm names; restrict to these arms "
                         "(e.g. cbmem,graphify to run only CPU-bound arms)")
    ap.add_argument("--skip-arms", default="",
                    help="comma-separated arm names; exclude these arms "
                         "(complement of --only-arms)")
    args = ap.parse_args()
    only = {a.strip() for a in args.only_arms.split(",") if a.strip()} or None
    skip = {a.strip() for a in args.skip_arms.split(",") if a.strip()} or None
    run_stage(args.stage, args.workers, args.probe, args.force, args.limit,
              args.dataset, only, skip)


if __name__ == "__main__":
    main()
