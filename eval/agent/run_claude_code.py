"""Drive REAL Claude Code (headless) on SWE-bench, with SkeletonGraph wired in
as an MCP server providing the engine-side `sg-rerank` retrieval.

This is the "real frontier agent uses SG" arm — distinct from the controlled
ReAct harness (run_stage.py). Instead of our 5-tool ReAct loop talking to a
vLLM/NIM endpoint, we launch the actual `claude` CLI in print/headless mode on
a per-task editable repo copy. SG is the project's MCP server (.mcp.json), so
Claude reaches code through sg_search/sg_get/sg_expand (sg-rerank) exactly as a
real user would. Native Read/Grep/Edit stay enabled — the honest "SG available,
prefer it" setup, not a forced ablation.

Pipeline per task:
  1. prepare_repo  — persistent editable copy at base commit, clean git baseline,
                     `sg build` index + `sg install --ide claude-code` (.mcp.json
                     + hooks + CLAUDE.md). SG artifacts are gitignored so they
                     never pollute the patch. Idempotent (skip if prepared).
  2. run_claude    — `claude -p` (stream-json) in that dir, SG as strict MCP.
  3. extract_patch — `git add -A && git diff --cached` (captures new files too;
                     gitignored SG state excluded).
  4. write a run JSON in the SAME schema verify.py / aggregate.py consume, so the
     sg-rerank/Claude arm folds into the existing pass@1 + tables unchanged.

Isolate results from the vLLM/NIM runs with a distinct tag:
    $env:SG_EVAL_RUN_TAG = "claude_sgrerank"

Run 4-5 of these in parallel terminals, each pinned to a task shard:
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --shard 1/5
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --shard 2/5
    ... (shards 3/5, 4/5, 5/5 in their own terminals)

Pre-stage every editable copy first (one-time, satisfies "make sure they are
copied") without running any agent:
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --prepare-only

Then score with the existing harness (same tag):
    python -m eval.agent.verify --all --only-arms sg-rerank --run-tag claude_sgrr
    python -m eval.agent.aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .isolation import _GIT, _rmtree_safe, run_id
from .run_agent import load_tasks

SG = shutil.which("sg") or "sg"
CLAUDE = shutil.which("claude") or "claude"

ARM_SG = "sg-rerank"    # MCP server's heuristic_query IS engine-side sg-rerank
ARM_NATIVE = "native"   # Claude Code on its own — no SG, native tools only
ARMS = (ARM_SG, ARM_NATIVE)

# SG artifacts + standard caches kept OUT of the agent's patch. Written to the
# copy's .gitignore BEFORE the baseline commit, so `git add -A` never stages
# them and `git diff` never shows them — same belt-and-braces idea as
# isolation._WORKSPACE_GITIGNORE, plus the Claude-Code-specific files.
_GITIGNORE = """\
# SkeletonGraph + Claude Code eval — keep SG/agent config out of the patch
.skeletongraph/
.mcp.json
.claude/
CLAUDE.md
.sg_prepared
.hybrid_index/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
"""

# Reinforces CLAUDE.md in case -p mode does not surface project memory the same
# way an interactive session does. Kept short — the MCP tool descriptions and
# CLAUDE.md carry the detail.
_SG_APPEND_SYSTEM = (
    "SkeletonGraph (SG) is wired in as an MCP server for this repo. Prefer its "
    "tools to locate code: call sg_overview once at the start, then sg_search "
    "(a whole-task context assembler, not grep) to find edit targets, and "
    "sg_get/sg_expand for exact follow-ups. Use native Grep/Read only when SG "
    "does not return what you need."
)

_SG_PROMPT = """Fix the following GitHub issue in this repository by editing the \
source files directly.

--- ISSUE ---
{issue}

Guidelines:
- Make the smallest change that correctly fixes the issue.
- Prefer the SkeletonGraph MCP tools (sg_overview, sg_search, sg_get, sg_expand) \
to locate the relevant code before reading or grepping.
- Do NOT run or write tests — the test environment is not available.
- When the fix is complete, stop.
"""

# Native baseline — Claude Code on its own. No SG mention, so the agent uses its
# own tools (Grep/Read/Edit/...) exactly as it would for any user. This is the
# control the SG-wrapped arm is measured against.
_NATIVE_PROMPT = """Fix the following GitHub issue in this repository by editing \
the source files directly.

--- ISSUE ---
{issue}

Guidelines:
- Make the smallest change that correctly fixes the issue.
- Do NOT run or write tests — the test environment is not available.
- When the fix is complete, stop.
"""


# ── git helpers (env mirrors isolation._init_clean_git) ──────────────────────

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "sg-eval", "GIT_AUTHOR_EMAIL": "eval@local",
    "GIT_COMMITTER_NAME": "sg-eval", "GIT_COMMITTER_EMAIL": "eval@local",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([_GIT, *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, **_GIT_ENV}, check=check)


def _copies_root(task: dict) -> Path:
    """Persistent editable copies live next to the source clones, in a sibling
    _claude_repos dir (NOT the ephemeral _agent_work tree)."""
    return Path(task["repo_path"]).resolve().parent.parent / "_claude_repos"


def _repo_dir(task: dict, arm: str = ARM_SG) -> Path:
    # sg-rerank keeps the bare task dir (back-compat with already-prepared
    # copies); every other arm namespaces so SG and native copies never clash.
    base = _copies_root(task)
    return base / (task["task_id"] if arm == ARM_SG
                   else f"{task['task_id']}__{arm}")


# ── prepare: editable copy + clean baseline + (SG arm only) index + MCP ──────

def prepare_repo(task: dict, arm: str = ARM_SG, rebuild: bool = False,
                 verbose: bool = True) -> Path:
    """Create (or reuse) a persistent editable copy.

    SG arm: + `sg build` index + `sg install --ide claude-code` (.mcp.json,
    hooks, CLAUDE.md), all gitignored. Native arm: clean repo only — NO SG, so
    it is a fair "Claude on its own" control. Idempotent via a `.sg_prepared`
    marker; pass rebuild=True to wipe and redo.
    """
    repo = _repo_dir(task, arm)
    marker = repo / ".sg_prepared"

    if marker.exists() and not rebuild:
        reset_repo(repo)
        return repo

    src = Path(task["repo_path"]).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source clone missing: {src}")

    if verbose:
        print(f"  prepare {task['task_id']} -> {repo}")
    _rmtree_safe(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)

    # Copy the worktree, excluding .git (a worktree's .git is a FILE pointing
    # into a shared cache — copying it leaves a broken pointer) and any SG state.
    excludes = (".git", ".skeletongraph", ".mcp.json", ".claude", "CLAUDE.md",
                ".hybrid_index")
    for attempt in range(3):
        try:
            shutil.copytree(src, repo,
                            ignore=shutil.ignore_patterns(*excludes),
                            symlinks=False, ignore_dangling_symlinks=True)
            break
        except Exception:
            _rmtree_safe(repo)
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))

    # .gitignore SG/agent state BEFORE the baseline commit.
    gi = repo / ".gitignore"
    existing = gi.read_text(encoding="utf-8", errors="replace") if gi.exists() else ""
    if "# SkeletonGraph + Claude Code eval" not in existing:
        gi.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "")
                      + _GITIGNORE, encoding="utf-8")

    # Clean git baseline — the agent's diff is taken against this commit.
    _init_baseline(repo)

    if arm == ARM_SG:
        # Build the SG index, then install Claude Code integration (.mcp.json +
        # hooks + CLAUDE.md). Both write only gitignored paths. The native arm
        # skips this entirely so it stays a genuine SG-free baseline.
        _sg(repo, "build", "--path", str(repo))
        _sg(repo, "install", "--ide", "claude-code", "--path", str(repo))

    # Safety net: prepare must leave a CLEAN tracked tree (SG state all ignored).
    dirty = _git(repo, "status", "--porcelain").stdout.strip()
    if dirty:
        print(f"  WARN {task['task_id']}: SG install touched tracked files — "
              f"patch may be polluted:\n{dirty[:400]}")

    marker.write_text("ok\n", encoding="utf-8")
    return repo


def _init_baseline(repo: Path) -> None:
    seq = (
        [_GIT, "init", "-q"],
        [_GIT, "config", "core.longpaths", "true"],
        [_GIT, "config", "core.autocrlf", "false"],
        [_GIT, "add", "-A"],
        [_GIT, "commit", "-q", "-m", "baseline", "--no-verify"],
    )
    last = ""
    for attempt in range(4):
        err = None
        for cmd in seq:
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               env={**os.environ, **_GIT_ENV})
            if r.returncode != 0:
                err = (f"git {' '.join(cmd[1:])} failed ({r.returncode}): "
                       f"{(r.stderr or r.stdout).strip()[:200]}")
                break
        if err is None:
            return
        last = err
        gp = repo / ".git"
        if gp.is_dir():
            _rmtree_safe(gp)
        if attempt < 3:
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(last + "  (after 4 attempts — likely AV/file-lock)")


def _sg(repo: Path, *args: str) -> None:
    r = subprocess.run([SG, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(repo))
    if r.returncode != 0:
        raise RuntimeError(f"`sg {' '.join(args)}` failed ({r.returncode}): "
                           f"{(r.stderr or r.stdout).strip()[:300]}")


def reset_repo(repo: Path) -> None:
    """Return the copy to its baseline (discard the previous run's edits and any
    agent-created files). Gitignored SG state (.skeletongraph/.mcp.json) survives
    `git clean -fd` — only -x would remove ignored files — so the index is reused."""
    _git(repo, "reset", "--hard", "HEAD")
    _git(repo, "clean", "-fd")


def extract_patch(repo: Path) -> str:
    """Agent's changes as a unified diff, including new files. SG state is
    gitignored, so `git add -A` never stages it."""
    _git(repo, "add", "-A")
    patch = _git(repo, "diff", "--cached", "HEAD").stdout
    _git(repo, "reset", "-q")   # unstage; leave working tree as-is for inspection
    return patch


# ── run Claude Code headless ─────────────────────────────────────────────────

def run_claude(repo: Path, issue: str, model: str, timeout: int,
               arm: str = ARM_SG, disallow_grep: bool = False) -> dict:
    """Launch `claude -p` in the repo.

    SG arm: SG as the strict MCP server + an SG-first system nudge + SG prompt.
    Native arm: `--strict-mcp-config` with NO config (disables ALL project/global
    MCP, so it is SG-free) + a neutral prompt.
    disallow_grep: block native Grep/Glob (forces search through SG / read) —
    use to test SG as the SOLE retrieval surface.

    Returns {ok, exit, transcript (stream-json objects), result, raw}.
    """
    cmd = [
        CLAUDE, "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--dangerously-skip-permissions",
    ]
    if arm == ARM_SG:
        cmd += ["--mcp-config", str(repo / ".mcp.json"), "--strict-mcp-config",
                "--append-system-prompt", _SG_APPEND_SYSTEM]
        prompt = _SG_PROMPT.format(issue=issue)
    else:
        # An explicit EMPTY config + --strict-mcp-config ⇒ exactly zero MCP
        # servers (no project or global leakage). Truly Claude-on-its-own.
        cmd += ["--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config"]
        prompt = _NATIVE_PROMPT.format(issue=issue)
    if disallow_grep:
        cmd += ["--disallowedTools", "Grep", "Glob"]
    try:
        r = subprocess.run(cmd, cwd=str(repo), input=prompt,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        raw, exit_code = r.stdout, r.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or "") if isinstance(e.stdout, str) else ""
        exit_code, timed_out = -1, True

    objs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            pass
    result = next((o for o in reversed(objs) if o.get("type") == "result"), None)
    ok = (not timed_out) and exit_code == 0 and result is not None \
        and not result.get("is_error", False)
    return {"ok": ok, "exit": exit_code, "timed_out": timed_out,
            "transcript": objs, "result": result, "raw": raw}


def parse_transcript(objs: list, result: dict | None) -> dict:
    """Pull every comparison signal we can from a stream-json transcript:
    turn count, cumulative token usage, cost, PEAK context window, and per-tool
    call counts (split into SG-MCP vs native).

    Token model: Claude Code prompt-caches the re-sent history, so per-message
    `input_tokens` is only the fresh (uncached) slice; the real context the model
    saw each turn is input + cache_read + cache_creation. The PEAK of that across
    turns is the context-window high-water mark — the number to compare against a
    baseline that pastes everything.
    """
    tool_counts: dict = {}
    peak_context = 0
    for o in objs:
        if o.get("type") != "assistant":
            continue
        msg = o.get("message", {}) or {}
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
        u = msg.get("usage", {}) or {}
        ctx = ((u.get("input_tokens", 0) or 0)
               + (u.get("cache_read_input_tokens", 0) or 0)
               + (u.get("cache_creation_input_tokens", 0) or 0))
        peak_context = max(peak_context, ctx)

    usage = (result or {}).get("usage", {}) or {}
    billed_in = usage.get("input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_create = usage.get("cache_creation_input_tokens", 0) or 0
    sg_calls = sum(v for k, v in tool_counts.items()
                   if k.startswith("mcp__skeletongraph"))
    n_calls = sum(tool_counts.values())
    return {
        "n_turns": (result or {}).get("num_turns", 0) or 0,
        "billed_input": billed_in,
        "billed_output": usage.get("output_tokens", 0) or 0,
        "cached_input": cache_read,
        "cache_creation_input": cache_create,
        # Total input the model actually processed (fresh + both cache classes).
        "total_input_tokens": billed_in + cache_read + cache_create,
        "peak_context_tokens": peak_context,
        "cost_usd": (result or {}).get("total_cost_usd", 0.0) or 0.0,
        "duration_ms": (result or {}).get("duration_ms", 0) or 0,
        "tool_counts": tool_counts,
        "n_tool_calls": n_calls,
        "sg_tool_calls": sg_calls,
        "native_tool_calls": n_calls - sg_calls,
    }


# ── one task end-to-end ──────────────────────────────────────────────────────

def _model_tag(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "claude"


def _edited_gold(patch: str, gold: list) -> bool:
    touched = {ln[6:].strip().replace("\\", "/") for ln in patch.splitlines()
               if ln.startswith("+++ b/")}
    return bool(touched & {g.replace("\\", "/") for g in gold})


def _patch_metrics(patch: str) -> dict:
    """Unified-diff shape (mirrors run_agent._patch_metrics)."""
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
            "files_touched": len(files), "hunks": hunks, "files": sorted(files)}


# ── retrieval metrics from the SG MCP calls in the transcript ────────────────
# aggregate.py computes rec@1/rec@cum/prec from `search_calls`, and funcR@10 from
# turns[].tool_calls[].result. The Claude arm retrieves via MCP sg_search (not the
# harness search_code), so we reconstruct those fields here from the stream-json.

def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict))
    return ""


def _parse_sg_result_text(text: str):
    """Ordered, de-duped (fqns, files) from one sg_search result blob.

    SG ranks results as `## N. <file::symbol>` headers and `- \\`<file::symbol>\\``
    bullets (Other matches). We pull the FQN from both and derive the file."""
    fqns, files, seen_q, seen_p = [], [], set(), set()
    for line in text.splitlines():
        m = re.match(r"^##\s+\d+\.\s+(\S.+?)\s*$", line) \
            or re.match(r"^-\s+`([^`]+)`", line)
        if not m:
            continue
        fqn = m.group(1).strip()
        path = fqn.split("::", 1)[0].replace("\\", "/").strip()
        if not path:
            continue
        if fqn not in seen_q:
            seen_q.add(fqn); fqns.append(fqn)
        if path not in seen_p:
            seen_p.add(path); files.append(path)
    # Also count files surfaced via the "Module constants" (`NAME = … # path`)
    # and "Lexical matches" (`path:line:`) sections, so file-recall reflects the
    # constants/symbols SG returns — not just the function-graph hits.
    for line in text.splitlines():
        m = re.search(r"#\s+([\w./-]+\.\w+)\s*$", line) \
            or re.match(r"\s*([\w./-]+\.\w+):\d+:", line)
        if not m:
            continue
        p = m.group(1).replace("\\", "/").strip()
        if p and p not in seen_p:
            seen_p.add(p); files.append(p)
    return fqns, files


def _retrieval_from_transcript(objs: list, gold_files: list) -> dict:
    """Reconstruct search_calls + first/all FQN lists + first-search file recall
    metrics from the sg_search calls in a stream-json transcript."""
    gold = {g.replace("\\", "/") for g in gold_files}
    pending: dict = {}        # tool_use_id -> (query, order)
    order = 0
    calls = []                # (order, query, fqns, files)
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            for b in content:
                if (isinstance(b, dict) and b.get("type") == "tool_use"
                        and str(b.get("name", "")).endswith("sg_search")):
                    pending[b.get("id")] = (
                        (b.get("input") or {}).get("query", ""), order)
                    order += 1
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    query, od = pending.pop(tid)
                    fqns, files = _parse_sg_result_text(
                        _tool_result_text(b.get("content")))
                    calls.append((od, query, fqns, files))
    calls.sort(key=lambda c: c[0])

    search_calls, seen_gold = [], set()
    for od, query, fqns, files in calls:
        gih = sorted(gold & set(files))
        seen_gold |= set(gih)
        search_calls.append({
            "turn": od, "query": query, "hits": files, "n_hits": len(files),
            "gold_in_hits": gih,
            "precision": round(len(gih) / len(files), 4) if files else 0.0,
            "cumulative_recall": (round(len(seen_gold) / len(gold), 4)
                                  if gold else 0.0),
            "error": False,
        })
    first_files = calls[0][3] if calls else []
    first_fqns = calls[0][2] if calls else []
    all_fqns, seen = [], set()
    for _, _, fqns, _ in calls:
        for fq in fqns:
            if fq not in seen:
                seen.add(fq); all_fqns.append(fq)
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        "first_search_fqns": first_fqns,
        "all_search_fqns": all_fqns,
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
    }


def run_one_task(task: dict, arm: str, model: str, timeout: int,
                 rebuild: bool = False, disallow_grep: bool = False,
                 keep_transcript: bool = True) -> dict:
    model_tag = _model_tag(model)
    rid = run_id(task["task_id"], arm, 0, model_tag)
    out_path = config.RUNS_DIR / f"{rid}.json"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    repo = prepare_repo(task, arm, rebuild=rebuild, verbose=True)
    reset_repo(repo)   # guarantee a clean tree even if a prior run left edits

    run = run_claude(repo, task["query"], model, timeout, arm, disallow_grep)
    patch = extract_patch(repo)
    meta = parse_transcript(run["transcript"], run["result"])
    pm = _patch_metrics(patch)
    # Retrieval metrics only exist for the SG arm (native has no sg_search).
    ret = (_retrieval_from_transcript(run["transcript"], task.get("gold_files", []))
           if arm == ARM_SG else {
               "search_calls": [], "first_search_fqns": [], "all_search_fqns": [],
               "retrieval_hit": False, "retrieval_precision": 0.0,
               "retrieval_rank": 0})
    wall = round(time.time() - t0, 1)

    if run["timed_out"]:
        stopped = "timeout"
    elif run["ok"]:
        stopped = "submit"          # verify.py counts submit / max_turns
    else:
        stopped = "error"

    gold = task.get("gold_files", [])
    record = {
        "run_id": rid,
        "task_id": task["task_id"],
        "arm": arm,
        "model": model_tag,
        "model_full": model,
        "repeat": 0,
        "stopped": stopped,
        "harness": "claude-code",     # real-agent arm (SG-MCP or native)
        "disallow_grep": disallow_grep,
        "repo": task.get("repo", ""),
        "base_commit": task.get("base_commit", ""),
        "gold_files": gold,
        "model_patch": patch,
        "edited_gold_file": _edited_gold(patch, gold),
        "n_turns": meta["n_turns"],
        "billed_input": meta["billed_input"],
        "billed_output": meta["billed_output"],
        "cached_input": meta["cached_input"],
        "cache_creation_input": meta["cache_creation_input"],
        "total_input_tokens": meta["total_input_tokens"],
        "peak_context_tokens": meta["peak_context_tokens"],
        "imputed_cost": round(meta["cost_usd"], 6),
        "wall_s": wall,
        "claude_exit": run["exit"],
        "tool_counts": meta["tool_counts"],
        "n_tool_calls": meta["n_tool_calls"],
        "sg_tool_calls": meta["sg_tool_calls"],
        "native_tool_calls": meta["native_tool_calls"],
        # ── retrieval (reconstructed from sg_search; empty for native) ──
        "retrieval_hit": ret["retrieval_hit"],
        "retrieval_precision": ret["retrieval_precision"],
        "retrieval_rank": ret["retrieval_rank"],
        "search_calls": ret["search_calls"],
        "n_search_calls": len(ret["search_calls"]),
        "first_search_fqns": ret["first_search_fqns"],
        "all_search_fqns": ret["all_search_fqns"],
        "files_read": [], "edit_attempts": [],
        # ── patch shape + consolidation (so patch% / patch figures fill) ──
        "patch_lines_added": pm["lines_added"],
        "patch_lines_removed": pm["lines_removed"],
        "patch_files_touched": pm["files_touched"],
        "patch_hunks": pm["hunks"],
        "consolidation": {"files_in_patch_count": pm["files_touched"],
                          "files_read_count": 0, "files_read_and_used_count": 0,
                          "consolidation_gap_files": 0.0},
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if keep_transcript and run["raw"]:
        tdir = config.RUNS_DIR / "_claude_transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{rid}.jsonl").write_text(run["raw"], encoding="utf-8")

    try:
        write_index(config.RUNS_DIR)
    except Exception:
        pass   # index is a convenience; never fail a run over it

    return record


def write_index(runs_dir: Path) -> None:
    """Regenerate `_INDEX.md` — a traversable table of every Claude-Code run in
    this results dir, with metrics + clickable paths to each transcript and
    editable repo copy. Rewritten from all JSONs each call, so it is always a
    complete snapshot (race-tolerant across parallel shards)."""
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("harness") != "claude-code":
            continue
        rows.append(r)
    if not rows:
        return

    copies_root = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/_claude_repos"
    lines = [
        "# Claude Code runs (SG-MCP vs native) — index",
        "",
        f"_Auto-generated. {len(rows)} run(s). Results dir: `{runs_dir}`._",
        "",
        "- Run JSON (full record): `<run_id>.json` in this dir.",
        "- Stream-json transcript (every message + tool call): "
        "`_claude_transcripts/<run_id>.jsonl`.",
        "- Editable repo copy (inspect the agent's edits): "
        f"`{copies_root}/<task_id>/` — `git -C <copy> diff HEAD` after a run.",
        "",
        "**Columns** — peak_ctx = context-window high-water mark (fresh+cached "
        "input the model saw in one turn); total_in = all input processed; "
        "sg/nat = SkeletonGraph-MCP vs native tool calls.",
        "",
        "| task | arm | stopped | turns | sg | nat | peak_ctx | total_in | out | "
        "cost$ | wall_s | edited_gold | transcript |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot_cost = tot_in = tot_out = 0
    for r in rows:
        rid = r.get("run_id", "?")
        tot_cost += r.get("imputed_cost", 0) or 0
        tot_in += r.get("total_input_tokens", 0) or 0
        tot_out += r.get("billed_output", 0) or 0
        lines.append(
            f"| {r.get('task_id','?')} | {r.get('arm','?')} "
            f"| {r.get('stopped','?')} "
            f"| {r.get('n_turns',0)} | {r.get('sg_tool_calls',0)} "
            f"| {r.get('native_tool_calls',0)} "
            f"| {r.get('peak_context_tokens',0):,} "
            f"| {r.get('total_input_tokens',0):,} | {r.get('billed_output',0):,} "
            f"| {r.get('imputed_cost',0):.4f} | {r.get('wall_s',0)} "
            f"| {r.get('edited_gold_file',False)} "
            f"| `_claude_transcripts/{rid}.jsonl` |"
        )
    n = len(rows)
    lines += [
        "",
        f"**Totals** — runs: {n} · cost: ${tot_cost:.2f} · "
        f"total input: {tot_in:,} · output: {tot_out:,} · "
        f"mean cost/run: ${tot_cost / n:.4f} · "
        f"mean peak_ctx: {sum(x.get('peak_context_tokens',0) for x in rows)//n:,}",
        "",
    ]
    (runs_dir / "_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def _already_done(task: dict, arm: str, model_tag: str) -> bool:
    rid = run_id(task["task_id"], arm, 0, model_tag)
    p = config.RUNS_DIR / f"{rid}.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return rec.get("stopped") in ("submit", "max_turns")


def _parse_shard(shard: str):
    if not shard:
        return None
    try:
        k, n = (int(x) for x in shard.split("/"))
        if not (1 <= k <= n):
            raise ValueError
        return (k, n)
    except Exception:
        raise SystemExit(f"--shard must be 'k/N' with 1<=k<=N (got {shard!r})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="tasks jsonl (e.g. swebench_100.jsonl)")
    ap.add_argument("--arm", default=ARM_SG, choices=list(ARMS),
                    help="sg-rerank = Claude + SkeletonGraph MCP (default); "
                         "native = Claude on its own, no SG (the baseline).")
    ap.add_argument("--disallow-grep", action="store_true",
                    help="block native Grep/Glob so search goes through SG "
                         "(SG-as-sole-retrieval). Only meaningful for sg-rerank.")
    ap.add_argument("--model", default="sonnet",
                    help="Claude model: alias ('sonnet'/'opus') or full id "
                         "(default: sonnet)")
    ap.add_argument("--shard", default="",
                    help="'k/N' — run only the k-th of N strided task shards "
                         "(1-based). Run the SAME command in N terminals.")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent claude processes IN THIS terminal (default 1; "
                         "raise only if you want one terminal to drive several)")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="seconds per task before killing claude (default 1200)")
    ap.add_argument("--force", action="store_true", help="re-run completed tasks")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe + rebuild each editable copy (re-copy, re-index)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="only stage the editable copies (copy+index+install); "
                         "run no agent. Use once up front to pre-copy all repos.")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.dataset))
    if args.limit > 0:
        tasks = tasks[:args.limit]
    shard = _parse_shard(args.shard)
    if shard:
        k, n = shard
        tasks = tasks[k - 1::n]

    arm = args.arm
    model_tag = _model_tag(args.model)
    if not args.force and not args.prepare_only:
        tasks = [t for t in tasks if not _already_done(t, arm, model_tag)]

    grep_note = " | grep DISALLOWED" if args.disallow_grep else ""
    print(f"Claude Code [{arm}] | model={args.model}{grep_note} "
          f"| tag={config._RUN_TAG or '(none)'} | {len(tasks)} tasks "
          f"| workers={args.workers}")
    print(f"  results -> {config.RUNS_DIR}")
    if shard:
        print(f"  shard {shard[0]}/{shard[1]} (strided)")
    if not tasks:
        print("  nothing to do (all done) — run verify + aggregate")
        return

    if args.prepare_only:
        for i, t in enumerate(tasks, 1):
            try:
                prepare_repo(t, arm, rebuild=args.rebuild, verbose=True)
                print(f"  [{i}/{len(tasks)}] prepared {t['task_id']}")
            except Exception as e:
                print(f"  [{i}/{len(tasks)}] PREPARE FAILED {t['task_id']}: "
                      f"{type(e).__name__}: {e}")
        print("Prepared. Drop --prepare-only to run the agent.")
        return

    t0 = time.time()
    done = fail = 0
    if args.workers <= 1:
        for t in tasks:
            try:
                rec = run_one_task(t, arm, args.model, args.timeout,
                                   args.rebuild, args.disallow_grep)
                done += 1
                _report(done + fail, len(tasks), rec)
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(tasks)}] {t['task_id']} FAILED: "
                      f"{type(e).__name__}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one_task, t, arm, args.model, args.timeout,
                                args.rebuild, args.disallow_grep): t
                    for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    rec = fut.result()
                    done += 1
                    _report(done + fail, len(tasks), rec)
                except Exception as e:
                    fail += 1
                    print(f"  [{done+fail}/{len(tasks)}] {t['task_id']} FAILED: "
                          f"{type(e).__name__}: {e}")

    mins = (time.time() - t0) / 60
    print(f"\nDone: {done} ok, {fail} failed, {mins:.1f} min. Results in {config.RUNS_DIR}")
    print(f"Next:  python -m eval.agent.verify --all --only-arms {arm} "
          f"--run-tag <harness_tag>")
    print( "Then:  python -m eval.agent.aggregate")


def _report(n: int, total: int, rec: dict) -> None:
    tools = rec.get("tool_counts", {})
    sg_calls = sum(v for k, v in tools.items() if k.startswith("mcp__skeletongraph"))
    print(f"  [{n}/{total}] {rec['run_id']}: {rec['stopped']} "
          f"turns={rec['n_turns']} sg_calls={sg_calls} "
          f"edited_gold={rec['edited_gold_file']} "
          f"in={rec['billed_input']} out={rec['billed_output']} "
          f"${rec['imputed_cost']} {rec['wall_s']}s")


if __name__ == "__main__":
    main()
