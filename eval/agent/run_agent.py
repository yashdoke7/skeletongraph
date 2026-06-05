"""Single (task, arm) run: isolate -> ReAct loop -> capture patch -> save.

Usually driven by run_stage.py, but runnable standalone for debugging:

    python -m eval.agent.run_agent --task-id sympy__sympy-24066 --arm sg

The record JSON written here is the single source of truth for every paper
figure. We deliberately over-capture: trajectory shape, per-call retrieval,
per-call file access, edit attempts, patch shape, consolidation-gap signals.
Adding a new figure/table later must NEVER require re-running the agent.
See docs/RESEARCH_PLAN.md §6 for the metric catalog.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

from . import config
from .isolation import (cleanup_workspace, diff_patch, prepare_workspace, run_id)
from .react import run_react
from .tools import ToolExecutor, preflight_arm


def load_tasks(path: Path = config.DATASET) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def run_one(task: dict, arm: str, repeat: int = 0, model: str = "main",
            keep_workspace: bool = False) -> dict:
    """Execute one run. Returns a record dict (also written to RUNS_DIR)."""
    rid = run_id(task["task_id"], arm, repeat, model)
    out_path = config.RUNS_DIR / f"{rid}.json"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    repo = prepare_workspace(task, arm, repeat, model)

    # ── STRICT preflight (opt-in via SG_EVAL_STRICT=1) ───────────────────────
    # Verify the arm can run its EXACT intended config (e.g. embeddings actually
    # built) BEFORE spending agent tokens. On failure, abort with a clean
    # stopped="error" record — excluded from metrics, auto-retried by run_stage.
    # This is what makes a re-run trustworthy: no silent BM25-only degradation.
    if os.environ.get("SG_EVAL_STRICT") == "1":
        err = preflight_arm(config.ARMS[arm].backend, repo)
        if err:
            if not keep_workspace:
                cleanup_workspace(repo)
            rec = {
                "run_id": rid, "task_id": task["task_id"], "arm": arm,
                "model": model, "repeat": repeat, "stopped": "error",
                "error": err, "n_turns": 0,
                "retrieval_hit": False, "edited_gold_file": False,
            }
            out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            print(f"  ABORT {rid}: {err}")
            return rec

    try:
        executor = ToolExecutor(repo, config.ARMS[arm].backend)
        traj = run_react(task, arm, executor, model=model)
        patch = diff_patch(repo)
    finally:
        if not keep_workspace:
            cleanup_workspace(repo)

    gold = task.get("gold_files", [])
    # Normalize gold-file separators once — Windows paths in search hits use
    # backslashes; we compare in forward-slash form everywhere.
    gold_norm = [g.replace("\\", "/") for g in gold]

    # First-search retrieval (Axis 2 headline) — kept for backwards compat.
    rmetrics = _retrieval_metrics(gold_norm, [h.replace("\\", "/")
                                              for h in traj.first_search_hits])

    # Per-call trajectory analytics (the heavy ones — every figure depends on
    # these). All read post-hoc from traj.turns so react.py stays minimal.
    search_calls = _search_call_metrics(traj.turns, gold_norm)
    files_read = _file_access_metrics(traj.turns, gold_norm)
    edit_attempts = _edit_metrics(traj.turns, gold_norm)
    patch_metrics = _patch_metrics(patch)
    shape = _trajectory_shape_metrics(traj.turns, gold_norm,
                                       search_calls, files_read, edit_attempts)
    consolidation = _consolidation_metrics(files_read, patch_metrics["files"])

    record = traj.to_dict()
    record.update({
        "run_id": rid,
        "repeat": repeat,
        "repo": task.get("repo", ""),
        "base_commit": task.get("base_commit", ""),
        "gold_files": gold,
        "model_patch": patch,

        # ── Headline retrieval (first search) — kept for backwards compat ──
        # retrieval_hit (recall): was any gold file anywhere in the results.
        #   Gameable — a backend that dumps 50 noisy files "hits" by luck.
        # retrieval_precision: gold files / total files returned. Discriminating.
        # retrieval_rank: 1-indexed rank of the first gold file (0 = absent).
        "retrieval_hit": rmetrics["hit"],
        "retrieval_precision": rmetrics["precision"],
        "retrieval_rank": rmetrics["rank"],
        "edited_gold_file": _edited_gold(patch, gold_norm),

        # SG arms: True iff the semantic embedding index actually built.
        # False = the run silently degraded to BM25-only — NOT real SG.
        "embeddings_used": executor.embeddings_used,

        # ── Per-call retrieval (for Pareto + cumulative-recall figures) ──
        # Every search_code call with query, hits, gold intersection,
        # precision, and cumulative recall up to that point.
        "search_calls": search_calls,
        "n_search_calls": len(search_calls),
        "unique_files_retrieved_total": len({
            f for sc in search_calls for f in sc["hits"]
        }),

        # ── Per-call file access (for consolidation-gap figure) ──
        "files_read": files_read,

        # ── Per-call edit attempts (for failure-mode taxonomy) ──
        "edit_attempts": edit_attempts,

        # ── Trajectory shape (for time-to-edit, edit-success, guard firing) ──
        "time_to_first_edit_turn": shape["time_to_first_edit_turn"],
        "time_to_first_gold_read_turn": shape["time_to_first_gold_read_turn"],
        "edits_attempted": shape["edits_attempted"],
        "edits_successful": shape["edits_successful"],
        "empty_submit_blocked": shape["empty_submit_blocked"],

        # ── Patch shape (for code-impact figures) ──
        "patch_lines_added": patch_metrics["lines_added"],
        "patch_lines_removed": patch_metrics["lines_removed"],
        "patch_files_touched": patch_metrics["files_touched"],
        "patch_hunks": patch_metrics["hunks"],

        # ── Consolidation gap (the ContextBench-style headline figure) ──
        # files_read_count: how many distinct files the agent opened.
        # files_in_patch_count: how many files the final patch touches.
        # files_read_and_used_count: intersection.
        # consolidation_gap_files: 1 - used/read. 0.0 = perfect (everything
        #   read ended up in the patch). 1.0 = nothing read was used.
        # Block/line-level usage_drop requires ContextBench gold annotations
        # — backfilled by a later dataset step (see IMPLEMENTATION_PLAN.md).
        "consolidation": consolidation,
    })
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# ── Retrieval analytics ─────────────────────────────────────────────────────


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


_RANK_LINE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")


def _parse_search_result_files(result: str) -> List[str]:
    """Extract ordered file paths from a search_code tool result.

    Backends differ: BM25/grep return bare paths; SG returns FQNs (file::symbol).
    We strip the symbol part and dedup-preserve order. ERROR results yield [].
    Windows separators normalized to forward slash to match gold_files.
    """
    if not result or result.startswith("ERROR"):
        return []
    out: List[str] = []
    seen: set = set()
    for line in result.splitlines():
        m = _RANK_LINE.match(line)
        if not m:
            continue
        path = m.group(1).split("::", 1)[0].strip().replace("\\", "/")
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _search_call_metrics(turns, gold_files) -> list:
    """One dict per search_code call, in turn order."""
    gold = set(gold_files)
    seen_gold_so_far: set = set()
    out = []
    for t in turns:
        for call in t.tool_calls:
            if call.get("name") not in ("search_code", "cbmem_search"):
                continue
            result = call.get("result", "") or ""
            hits = _parse_search_result_files(result)
            gold_in_hits = sorted(gold & set(hits))
            seen_gold_so_far |= set(gold_in_hits)
            out.append({
                "turn": t.index,
                "query": (call.get("args") or {}).get("query", ""),
                "hits": hits,
                "n_hits": len(hits),
                "gold_in_hits": gold_in_hits,
                "precision": (round(len(gold_in_hits) / len(hits), 4)
                              if hits else 0.0),
                "cumulative_recall": (round(len(seen_gold_so_far) / len(gold), 4)
                                       if gold else 0.0),
                "error": result.startswith("ERROR"),
            })
    return out


def _file_access_metrics(turns, gold_files) -> list:
    """One dict per read_file call, in turn order."""
    gold = set(gold_files)
    out = []
    for t in turns:
        for call in t.tool_calls:
            name = call.get("name")
            if name not in ("read_file", "read_symbol"):
                continue
            args = call.get("args") or {}
            if name == "read_symbol":          # SG native fetch — path is the fqn's file
                path = (args.get("fqn") or "").split("::", 1)[0].replace("\\", "/")
            else:
                path = (args.get("path") or "").replace("\\", "/")
            out.append({
                "turn": t.index,
                "path": path,
                "was_gold": path in gold,
            })
    return out


def _edit_metrics(turns, gold_files) -> list:
    """One dict per edit_file call, in turn order. Captures successes AND
    failures — the model attempting to edit but failing (`old_str not found`)
    is a distinct failure mode from never attempting an edit."""
    gold = set(gold_files)
    out = []
    for t in turns:
        for call in t.tool_calls:
            if call.get("name") != "edit_file":
                continue
            args = call.get("args") or {}
            path = (args.get("path") or "").replace("\\", "/")
            result = call.get("result", "") or ""
            success = result.startswith("Edited ")
            out.append({
                "turn": t.index,
                "path": path,
                "was_gold": path in gold,
                "success": success,
                "error_kind": (None if success else
                                 ("not_found" if "not found" in result
                                  else "multiple" if "matches" in result
                                  else "other")),
            })
    return out


def _trajectory_shape_metrics(turns, gold_files,
                               search_calls, files_read, edit_attempts) -> dict:
    """Aggregate trajectory shape signals."""
    gold = set(gold_files)
    time_to_first_edit = None
    for ev in edit_attempts:
        if ev["success"]:
            time_to_first_edit = ev["turn"]
            break
    time_to_first_gold_read = None
    for ev in files_read:
        if ev["was_gold"]:
            time_to_first_gold_read = ev["turn"]
            break
    # Empty-submit guard fires (and the model retries) when a submit returns
    # the "you have not edited any file" sentinel from tools.py.
    empty_submit_blocked = False
    for t in turns:
        for call in t.tool_calls:
            if call.get("name") == "submit":
                r = (call.get("result") or "").lower()
                if "have not edited" in r:
                    empty_submit_blocked = True
    return {
        "time_to_first_edit_turn": time_to_first_edit,
        "time_to_first_gold_read_turn": time_to_first_gold_read,
        "edits_attempted": len(edit_attempts),
        "edits_successful": sum(1 for e in edit_attempts if e["success"]),
        "empty_submit_blocked": empty_submit_blocked,
    }


def _patch_metrics(patch: str) -> dict:
    """Parse a unified diff into shape stats."""
    if not patch:
        return {"lines_added": 0, "lines_removed": 0,
                "files_touched": 0, "hunks": 0, "files": set()}
    files: set = set()
    added = removed = hunks = 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip().replace("\\", "/"))
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"lines_added": added, "lines_removed": removed,
            "files_touched": len(files), "hunks": hunks, "files": files}


def _consolidation_metrics(files_read, patch_files: set) -> dict:
    """How much of what was retrieved actually shows up in the final patch.

    File-granularity proxy for ContextBench's usage-drop metric. Block/line
    granularity needs gold-block annotations from the ContextBench dataset
    (see IMPLEMENTATION_PLAN.md §4).
    """
    read_set = {ev["path"] for ev in files_read if ev.get("path")}
    used = read_set & patch_files
    n_read = len(read_set)
    return {
        "files_read_count": n_read,
        "files_in_patch_count": len(patch_files),
        "files_read_and_used_count": len(used),
        "files_retrieved_but_unused_count": len(read_set - patch_files),
        "consolidation_gap_files": (round(1.0 - len(used) / n_read, 4)
                                     if n_read else 0.0),
    }


def _edited_gold(patch: str, gold_files: list) -> bool:
    """True if the agent's diff touches any gold file (cheap pre-verify signal)."""
    touched = {ln[6:].strip().replace("\\", "/") for ln in patch.splitlines()
               if ln.startswith("+++ b/")}
    return bool(touched & set(gold_files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--model", default="main", choices=list(config.MODELS))
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
          f"edits={rec['edits_successful']}/{rec['edits_attempted']} "
          f"gap={rec['consolidation']['consolidation_gap_files']} "
          f"emb={rec['embeddings_used']} "
          f"in={rec['billed_input']} out={rec['billed_output']} "
          f"cost=${rec['imputed_cost']}")


if __name__ == "__main__":
    main()
