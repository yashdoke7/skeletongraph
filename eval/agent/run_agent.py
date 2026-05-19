"""Single (task, arm) run: isolate -> ReAct loop -> capture patch -> save.

Usually driven by run_stage.py, but runnable standalone for debugging:

    python -m eval.agent.run_agent --task-id sympy__sympy-24066 --arm sg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .isolation import (cleanup_workspace, diff_patch, prepare_workspace, run_id)
from .react import run_react
from .tools import ToolExecutor


def load_tasks(path: Path = config.DATASET) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def run_one(task: dict, arm: str, repeat: int = 0, model: str = "qwen-32b",
            keep_workspace: bool = False) -> dict:
    """Execute one run. Returns a record dict (also written to RUNS_DIR)."""
    rid = run_id(task["task_id"], arm, repeat, model)
    out_path = config.RUNS_DIR / f"{rid}.json"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    repo = prepare_workspace(task, arm, repeat, model)
    try:
        executor = ToolExecutor(repo, config.ARMS[arm].backend)
        traj = run_react(task, arm, executor, model=model)
        patch = diff_patch(repo)
    finally:
        if not keep_workspace:
            cleanup_workspace(repo)

    gold = task.get("gold_files", [])
    hits = traj.first_search_hits
    rmetrics = _retrieval_metrics(gold, hits)

    record = traj.to_dict()
    record.update({
        "run_id": rid,
        "repeat": repeat,
        "repo": task.get("repo", ""),
        "base_commit": task.get("base_commit", ""),
        "gold_files": gold,
        "model_patch": patch,
        # Axis 2 — retrieval quality of the FIRST search.
        # retrieval_hit (recall): was any gold file anywhere in the results.
        #   Gameable — a backend that dumps 50 noisy files "hits" by luck.
        # retrieval_precision: gold files / total files returned. Punishes the
        #   noise dump; this is the discriminating metric for the paper.
        # retrieval_rank: 1-indexed rank of the first gold file (0 = absent).
        "retrieval_hit": rmetrics["hit"],
        "retrieval_precision": rmetrics["precision"],
        "retrieval_rank": rmetrics["rank"],
        "edited_gold_file": _edited_gold(patch, gold),
        # SG arms: True iff the semantic embedding index actually built.
        # False = the run silently degraded to BM25-only — NOT real SG.
        "embeddings_used": executor.embeddings_used,
    })
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _retrieval_metrics(gold_files: list, hits: list) -> dict:
    """Recall (hit), precision, and rank-of-first-gold for one search ranking.

    `hits` is the ordered list of distinct files from the first search.
    A binary hit alone rewards backends that return a huge unranked list;
    precision and rank expose that. rank is 1-indexed; 0 means "not found".
    """
    gold = set(gold_files)
    found = [f for f in hits if f in gold]
    rank = 0
    for i, f in enumerate(hits):
        if f in gold:
            rank = i + 1
            break
    return {
        "hit": bool(found),
        "precision": round(len(found) / len(hits), 4) if hits else 0.0,
        "rank": rank,
    }


def _edited_gold(patch: str, gold_files: list) -> bool:
    """True if the agent's diff touches any gold file (cheap pre-verify signal)."""
    touched = {ln[6:].strip() for ln in patch.splitlines()
               if ln.startswith("+++ b/")}
    return bool(touched & set(gold_files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--model", default="qwen-32b", choices=list(config.MODELS))
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--keep-workspace", action="store_true")
    args = ap.parse_args()

    tasks = {t["task_id"]: t for t in load_tasks()}
    if args.task_id not in tasks:
        raise SystemExit(f"task {args.task_id} not in {config.DATASET}")

    rec = run_one(tasks[args.task_id], args.arm, args.repeat, args.model,
                  args.keep_workspace)
    print(f"{rec['run_id']}: stopped={rec['stopped']} turns={rec['n_turns']} "
          f"hit={rec['retrieval_hit']} prec={rec['retrieval_precision']} "
          f"rank={rec['retrieval_rank']} edited_gold={rec['edited_gold_file']} "
          f"emb={rec['embeddings_used']} "
          f"in={rec['billed_input']} out={rec['billed_output']} "
          f"cost=${rec['imputed_cost']}")


if __name__ == "__main__":
    main()
