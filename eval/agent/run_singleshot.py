"""Single-shot SG (no agent loop) — the "why agent" ablation.

Pipeline: SG retrieves ONCE on the raw issue → top-k retrieved files are stuffed
into a single prompt → ONE model generation returns edits as JSON → edits are
applied with the SAME edit semantics as the agent harness → capture the patch.
No iteration, no tool feedback. This isolates the contribution of the *agent
loop* itself (agentic SG vs single-shot SG-RAG).

Recorded as arm `sg-noagent` so it lands beside `sg` in aggregate.py / plots.py.

    python -m eval.agent.run_singleshot --task-id astropy__astropy-8707
    python -m eval.agent.run_singleshot --limit 10
    python -m eval.agent.run_singleshot --all

Uses the same endpoint/model as the agent harness (SG_EVAL_API_BASE / MODEL).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

from . import config
from .isolation import cleanup_workspace, diff_patch, prepare_workspace, run_id
from .react import _client, _usage
from .run_agent import (_consolidation_metrics, _edit_metrics, _edited_gold,
                        _file_access_metrics, _patch_metrics, _retrieval_metrics,
                        _search_call_metrics, load_tasks)
from .tools import ToolExecutor

_ARM = "sg-noagent"
_MODEL_LABEL = "sg"          # run_id label (matches config.ARMS["sg-noagent"].backend)
_TOP_FILES = 4               # how many retrieved files to stuff into the prompt
_FILE_CHAR_CAP = 6000        # per-file content cap (keep the prompt bounded)

SYSTEM = """You are an autonomous software engineer. You are given a bug report \
and the most relevant source files (already retrieved for you). Produce the \
minimal correct fix.

Return ONLY a JSON array of edits, nothing else. Each edit is:
  {"path": "<file path>", "old_str": "<exact text to replace>", "new_str": "<replacement>"}
old_str must appear EXACTLY ONCE in the named file. Make the smallest change \
that fixes the bug. If no edit is needed, return []."""

USER_TMPL = """--- ISSUE ---
{issue}

--- RETRIEVED FILES ---
{context}

Return the JSON array of edits now."""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_edits(text: str) -> List[dict]:
    """Extract the edits JSON array from the model output (tolerant of fences)."""
    if not text:
        return []
    body = text
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        body = m.group(1)
    else:
        m2 = _JSON_ARRAY_RE.search(text)
        if m2:
            body = m2.group(0)
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(obj, list):
        return []
    out = []
    for e in obj:
        if isinstance(e, dict) and "path" in e and "old_str" in e and "new_str" in e:
            out.append({"path": str(e["path"]), "old_str": str(e["old_str"]),
                        "new_str": str(e["new_str"])})
    return out


def run_one(task: dict, repeat: int = 0, model: str = _MODEL_LABEL,
            keep_workspace: bool = False) -> dict:
    rid = run_id(task["task_id"], _ARM, repeat, model)
    out_path = config.RUNS_DIR / f"{rid}.json"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    repo = prepare_workspace(task, _ARM, repeat, model)
    t0 = time.time()
    stopped, error = "submit", ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    # synthetic turn log so the run_agent metric helpers can be reused
    search_turn = type("T", (), {"index": 0, "tool_calls": []})()
    edit_turn = type("T", (), {"index": 0, "tool_calls": []})()

    try:
        executor = ToolExecutor(repo, "sg")
        # ── 1. one SG retrieval on the raw issue ────────────────────────────
        search_result = executor.run("search_code", {"query": task["query"], "k": 10})
        search_turn.tool_calls.append({"name": "search_code",
                                       "args": {"query": task["query"]},
                                       "result": search_result})
        hits = executor.first_search_hits[:_TOP_FILES]

        # ── 2. stuff the top retrieved files into the prompt ────────────────
        ctx_parts = []
        for rel in hits:
            body = executor.run("read_file", {"path": rel})
            edit_turn.tool_calls.append({"name": "read_file", "args": {"path": rel},
                                         "result": ""})
            ctx_parts.append(f"### {rel}\n{body[:_FILE_CHAR_CAP]}")
        context = "\n\n".join(ctx_parts) if ctx_parts else "(no files retrieved)"

        # ── 3. ONE generation → edits ──────────────────────────────────────
        client = _client()
        resp = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER_TMPL.format(
                          issue=task["query"], context=context)}],
            temperature=config.TEMPERATURE, seed=config.SEED,
        )
        usage = _usage(resp)
        edits = _parse_edits(resp.choices[0].message.content or "")

        # ── 4. apply edits (same semantics as the agent's edit_file) ────────
        for e in edits:
            res = executor.run("edit_file", e)
            edit_turn.tool_calls.append({"name": "edit_file", "args": e, "result": res})

        patch = diff_patch(repo)
    except Exception as exc:
        stopped, error = "error", f"{type(exc).__name__}: {exc}"
        patch = ""
    finally:
        if not keep_workspace:
            cleanup_workspace(repo)

    gold = task.get("gold_files", [])
    gold_norm = [g.replace("\\", "/") for g in gold]
    turns = [search_turn, edit_turn]

    rmetrics = _retrieval_metrics(gold_norm, [h.replace("\\", "/")
                                              for h in executor.first_search_hits])
    search_calls = _search_call_metrics(turns, gold_norm)
    files_read = _file_access_metrics(turns, gold_norm)
    edit_attempts = _edit_metrics(turns, gold_norm)
    patch_metrics = _patch_metrics(patch)
    consolidation = _consolidation_metrics(files_read, patch_metrics["files"])
    in_tok, out_tok = usage["prompt_tokens"], usage["completion_tokens"]

    record = {
        "task_id": task["task_id"], "arm": _ARM, "model": model,
        "stopped": stopped, "error": error, "n_turns": 1,
        "billed_input": in_tok, "billed_output": out_tok,
        "cached_input": usage.get("cached_tokens", 0),
        "peak_context": in_tok, "tool_counts": {"search_code": 1},
        "first_search_hits": executor.first_search_hits,
        "wall_s": round(time.time() - t0, 1),
        "imputed_cost": config.impute_cost(in_tok, out_tok, usage.get("cached_tokens", 0)),
        "run_id": rid, "repeat": repeat,
        "repo": task.get("repo", ""), "base_commit": task.get("base_commit", ""),
        "gold_files": gold, "model_patch": patch,
        "retrieval_hit": rmetrics["hit"], "retrieval_precision": rmetrics["precision"],
        "retrieval_rank": rmetrics["rank"],
        "edited_gold_file": _edited_gold(patch, gold_norm),
        "embeddings_used": executor.embeddings_used,
        "search_calls": search_calls, "n_search_calls": len(search_calls),
        "unique_files_retrieved_total": len({f for sc in search_calls for f in sc["hits"]}),
        "files_read": files_read, "edit_attempts": edit_attempts,
        "time_to_first_edit_turn": next((e["turn"] for e in edit_attempts
                                         if e["success"]), None),
        "time_to_first_gold_read_turn": next((e["turn"] for e in files_read
                                              if e["was_gold"]), None),
        "edits_attempted": len(edit_attempts),
        "edits_successful": sum(1 for e in edit_attempts if e["success"]),
        "empty_submit_blocked": False,
        "patch_lines_added": patch_metrics["lines_added"],
        "patch_lines_removed": patch_metrics["lines_removed"],
        "patch_files_touched": patch_metrics["files_touched"],
        "patch_hunks": patch_metrics["hunks"],
        "consolidation": consolidation,
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--keep-workspace", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.task_id:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            raise SystemExit(f"task {args.task_id} not found")
    elif args.limit and not args.all:
        tasks = tasks[:args.limit]

    print(f"Single-shot SG (no agent) — {len(tasks)} task(s)")
    for i, t in enumerate(tasks, 1):
        # One bad task (e.g. a workspace/git failure) must not abort the whole
        # run — log it and continue, like run_stage does.
        try:
            rec = run_one(t, keep_workspace=args.keep_workspace)
            print(f"  [{i}/{len(tasks)}] {rec['run_id']}: {rec['stopped']} "
                  f"hit={rec['retrieval_hit']} edited_gold={rec['edited_gold_file']} "
                  f"edits={rec['edits_successful']}/{rec['edits_attempted']} "
                  f"cost=${rec['imputed_cost']}")
        except Exception as e:
            print(f"  [{i}/{len(tasks)}] {t['task_id']} FAILED: "
                  f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
