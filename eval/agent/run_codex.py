"""Drive OpenAI Codex CLI (headless `codex exec`) on SWE-bench, with and without
SkeletonGraph over MCP — the second production agent for the deployment study.

Mirrors run_claude_code.py so records are directly comparable and flow through
verify.py / stats.py / aggregate.py unchanged:
  * same task file, same persistent repo copies (reused from the Claude arms:
    `<task>__native` and `<task>__sg-fusion`, already indexed), same baseline commit,
    same `git diff` patch extraction, same record schema;
  * same user prompts and scope block (_NATIVE_PROMPT / _SG_PROMPT);
  * SG arm = SG as shipped for an agent: the MCP server pinned to fusion retrieval,
    the same project rules text Claude reads from CLAUDE.md (here written to
    AGENTS.md, which Codex reads), and the same system-level SG guidance Claude gets
    from --append-system-prompt and its session hook (here developer_instructions).
    Claude's PreToolUse hook (defer Grep until sg_search) has no Codex equivalent
    and is NOT reproduced — disclose this when reporting.

Isolation and parity choices:
  * Clean Codex home (<data root>/_codex_home, holding only a copy of auth.json) and
    --ignore-user-config: the user's own ~/.codex (config, installed plugins, skills,
    memories, session history) must reach neither arm. Features unrelated to editing
    a repository (sub-agents, plugins, apps, browser/computer use, image generation)
    are disabled for both arms; sub-agents also because their usage may not be
    counted in the parent's token totals.
  * No web access. The Claude arms could fetch the upstream fix (they did, on 10 of
    100 tasks; the paper reports that sensitivity), so here the web search tool is
    disabled and shell commands get an unreachable proxy (curl/git/pip/Invoke-
    WebRequest fail; Codex's own API connection is unaffected). Any remaining
    attempt is recorded per run in `leak_flags`.
  * A run cut off by the account's usage limit is discarded (no record) and the
    driver stops; rerun the same command after the limit resets. The rule is
    outcome-independent, like the rate-limit reruns in the controlled loop.
  * --dangerously-bypass-approvals-and-sandbox: parity with Claude's
    --dangerously-skip-permissions (both arms run unsandboxed, headless).
  * The SG MCP server gets the full (non-secret) user environment. Codex otherwise
    launches MCP servers with a minimal environment, in which `~` does not expand:
    SG's embedding cache then resolves to a literal ./~ folder, re-downloads model
    code, and either errors ("[Errno 22]") or silently falls back to a mismatched
    embedder — a broken SG, not SG.

Codex exposes no per-turn usage and no cost; the record stores Codex's own token
totals and an imputed cost from OpenAI's published price sheet (PRICES below).
`n_turns` counts completed agent items (tool calls, edits, messages), the closest
analogue of Claude's assistant turns; compare Codex arms with each other only.

Usage (PowerShell, from the repo root):
    $env:SG_EVAL_DATA_ROOT = "C:\\Users\\ASUS\\Desktop\\CS\\Projects\\swebench-data"
    $env:SG_EVAL_RUN_TAG   = "codex_v1"
    python -m eval.agent.run_codex --dataset eval\\datasets\\swebench_100.jsonl --arm codex-native --limit 5
    python -m eval.agent.run_codex --dataset eval\\datasets\\swebench_100.jsonl --arm codex-sg-fusion --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import config
from .isolation import run_id
from .run_agent import load_tasks
from .run_claude_code import (ARM_FUSION, ARM_NATIVE, _NATIVE_PROMPT, _SCOPE_BLOCK,
                              _SG_APPEND_SYSTEM, _SG_PROMPT, _edited_gold, _model_tag,
                              _parse_sg_result_text, _parse_shard, _patch_metrics,
                              _repo_dir, extract_patch, prepare_repo, reset_repo)

ARM_CX_NATIVE = "codex-native"
ARM_CX_SG = "codex-sg-fusion"
ARMS = (ARM_CX_NATIVE, ARM_CX_SG)
# Codex arms reuse the Claude arms' prepared copies (same code, same index).
_COPY_ARM = {ARM_CX_NATIVE: ARM_NATIVE, ARM_CX_SG: ARM_FUSION}

DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_EFFORT = "medium"
# USD per million tokens, from https://developers.openai.com/api/docs/models/gpt-5.6-terra
# (input $2, cached input $0.20, cache writes 1.25x input, output $12).
PRICES = {"gpt-5.6-terra": {"input": 2.00, "cached": 0.20, "cache_write": 2.50, "output": 12.00}}

CODEX = shutil.which("codex") or os.path.join(os.environ.get("APPDATA", ""), "npm", "codex.cmd")
SG = shutil.which("sg") or "sg"
_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
_QUOTA = re.compile(r"usage limit|rate limit|quota|too many requests|\b429\b", re.I)
_DISABLED_FEATURES = ("multi_agent", "plugins", "remote_plugin", "apps", "browser_use",
                      "browser_use_external", "computer_use", "in_app_browser",
                      "image_generation", "tool_suggest", "goals", "memories")


_DEAD_PROXY = "http://127.0.0.1:9"
_NO_NET = "{" + ",".join(f'{k}="{_DEAD_PROXY}"' for k in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) + ',NO_PROXY="",no_proxy=""}'
_NET_CMD = re.compile(r"\b(curl|wget|Invoke-WebRequest|iwr|Invoke-RestMethod|irm)\b|git\s+(fetch|pull|clone|ls-remote)|pip\s+(download|install)", re.I)
_GIT_DIG = re.compile(r"git\s+(fsck|cat-file|reflog|stash\s+(show|list|apply|pop))|lost-found|unreachable|dangling", re.I)


class QuotaExhausted(RuntimeError):
    pass


def _codex_home() -> Path:
    """A clean CODEX_HOME holding only the login, refreshed from the user's own home."""
    home = config._DATA_ROOT / "_codex_home"
    home.mkdir(parents=True, exist_ok=True)
    src = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"
    if not src.exists():
        raise SystemExit(f"no Codex login at {src}; run `codex login` first")
    dst = home / "auth.json"
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
    return home

# Shell-command categories, so Codex's single shell tool maps onto the paper's
# Search / Read / Execute buckets (Claude has separate Grep/Read/Bash tools).
_SEARCH = re.compile(r"(^|[\s'\"|;&(])(rg|grep|egrep|findstr|Select-String|sls|ack|ag)(\.exe)?\b|git\s+grep", re.I)
_READ = re.compile(r"(^|[\s'\"|;&(])(Get-Content|gc|cat|type|sed|head|tail|more|less|nl|bat)(\.exe)?\b", re.I)
_EXEC = re.compile(r"\bpython3?\b|\bpytest\b|\bpip\b|manage\.py|\btox\b", re.I)
_LIST = re.compile(r"(^|[\s'\"|;&(])(ls|dir|Get-ChildItem|gci|find|tree|fd)(\.exe)?\b", re.I)
_PATHLINE = re.compile(r"^\.?[\\/]?([\w.\-][\w.\-\\/]*\.\w+)(?::\d+)?(?::|$)")


def _toml_str(s: str) -> str:
    return json.dumps(s)          # a JSON string literal is a valid TOML basic string


def _mcp_env() -> str:
    env = {k: v for k, v in os.environ.items()
           if not _SECRET.search(k) and "\n" not in v and len(v) < 4000}
    env["SG_MCP_RETRIEVAL"] = "fusion"
    env["SG_MCP_BODY_TOP"] = "0"
    return "{" + ",".join(f"{json.dumps(k)}={_toml_str(v)}" for k, v in sorted(env.items())) + "}"


def _classify(cmd: str) -> str:
    inner = cmd
    m = re.search(r"-Command\s+['\"](.*)['\"]\s*$", cmd, re.S)
    if m:
        inner = m.group(1)
    if _EXEC.search(inner):
        return "Exec"
    if _SEARCH.search(inner):
        return "Search"
    if _READ.search(inner):
        return "Read"
    if _LIST.search(inner):
        return "List"
    if re.search(r"\bgit\b", inner):
        return "Git"
    return "Shell"


def _write_agents_md(repo: Path) -> None:
    """Same rules text the Claude SG arm read from CLAUDE.md, as AGENTS.md (ignored by git)."""
    src = repo / "CLAUDE.md"
    text = src.read_text(encoding="utf-8") if src.exists() else ""
    (repo / "AGENTS.md").write_text(text, encoding="utf-8")
    excl = repo / ".git" / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    cur = excl.read_text(encoding="utf-8") if excl.exists() else ""
    if "AGENTS.md" not in cur:
        excl.write_text(cur + ("\n" if cur and not cur.endswith("\n") else "") + "AGENTS.md\n",
                        encoding="utf-8")


def _remove_agents_md(repo: Path) -> None:
    try:
        (repo / "AGENTS.md").unlink()
    except FileNotFoundError:
        pass


def run_codex(repo: Path, issue: str, model: str, effort: str, timeout: int, arm: str) -> dict:
    cmd = [CODEX, "exec", "--json", "--ignore-user-config", "--skip-git-repo-check",
           "--dangerously-bypass-approvals-and-sandbox",
           "-m", model, "-c", f"model_reasoning_effort={_toml_str(effort)}", "-C", str(repo)]
    for f in _DISABLED_FEATURES:
        cmd += ["--disable", f]
    cmd += ["-c", 'web_search="disabled"', "-c", f"shell_environment_policy.set={_NO_NET}"]
    if arm == ARM_CX_SG:
        from skeletongraph.hooks.claude_code import _USE_SG_SYSTEM_MSG
        dev = _SG_APPEND_SYSTEM + "\n\n" + _USE_SG_SYSTEM_MSG
        cmd += ["-c", f"developer_instructions={_toml_str(dev)}",
                "-c", f"mcp_servers.skeletongraph.command={_toml_str(SG.replace(chr(92), '/'))}",
                "-c", "mcp_servers.skeletongraph.args=[\"serve\",\"--path\","
                      f"{_toml_str(str(repo).replace(chr(92), '/'))}]",
                "-c", "mcp_servers.skeletongraph.startup_timeout_sec=120",
                "-c", "mcp_servers.skeletongraph.tool_timeout_sec=300",
                "-c", f"mcp_servers.skeletongraph.env={_mcp_env()}"]
        prompt = _SG_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    else:
        prompt = _NATIVE_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    cmd.append("-")
    try:
        env = dict(os.environ, CODEX_HOME=str(_codex_home()))
        r = subprocess.run(cmd, cwd=str(repo), input=prompt, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        raw, err, exit_code, timed_out = r.stdout, r.stderr, r.returncode, False
    except subprocess.TimeoutExpired as e:
        raw = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        exit_code, timed_out = -1, True
    objs = []
    for line in raw.splitlines():
        try:
            objs.append(json.loads(line))
        except Exception:
            pass
    done = any(o.get("type") == "turn.completed" for o in objs)
    failed = next((o for o in objs if o.get("type") in ("turn.failed", "error")), None)
    return {"ok": done and not timed_out and exit_code == 0 and failed is None,
            "exit": exit_code, "timed_out": timed_out, "objs": objs, "raw": raw,
            "stderr": err[-4000:], "error": (failed or {}).get("error") or (failed or {}).get("message")}


def leak_flags(objs: list) -> list:
    """Attempts to reach the upstream fix: web, network commands, or digging in git storage."""
    out = []
    for o in objs:
        it = o.get("item") or {}
        if o.get("type") != "item.completed":
            continue
        if it.get("type") == "web_search":
            out.append("web_search: " + str(it.get("query") or "")[:200])
        elif it.get("type") == "command_execution":
            c = it.get("command") or ""
            if _NET_CMD.search(c):
                out.append("network: " + c[:200])
            elif _GIT_DIG.search(c):
                out.append("git: " + c[:200])
    return out


def parse_codex(objs: list, gold_files: list, repo: Path, model: str) -> dict:
    gold = {g.replace("\\", "/") for g in gold_files}
    items = [o["item"] for o in objs if o.get("type") == "item.completed" and o.get("item")]
    usage = {}
    for o in objs:
        if o.get("type") == "turn.completed":
            for k, v in (o.get("usage") or {}).items():
                usage[k] = usage.get(k, 0) + (v or 0)
    tool_counts, sg_calls, native_calls = {}, 0, 0
    searches, reads = [], []            # (order, query, files) for search-like calls
    order = 0
    for it in items:
        t = it.get("type")
        if t == "command_execution":
            cat = _classify(it.get("command", ""))
            tool_counts[cat] = tool_counts.get(cat, 0) + 1
            native_calls += 1
            if cat == "Search":
                files = []
                for ln in (it.get("aggregated_output") or "").splitlines():
                    m = _PATHLINE.match(ln.strip())
                    if m:
                        p = m.group(1).replace("\\", "/").lstrip("./")
                        if p not in files:
                            files.append(p)
                searches.append((order, it.get("command", ""), files, "shell"))
                order += 1
        elif t == "file_change":
            n = len(it.get("changes") or [])
            tool_counts["Edit"] = tool_counts.get("Edit", 0) + max(n, 1)
            native_calls += 1
        elif t == "mcp_tool_call":
            name = it.get("tool", "")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            sg_calls += 1
            if name == "sg_search":
                res = it.get("result")
                text = ""
                if isinstance(res, dict):
                    text = "\n".join(c.get("text", "") for c in res.get("content", []) if isinstance(c, dict))
                elif isinstance(res, str):
                    text = res.replace("\\n", "\n")
                _, files = _parse_sg_result_text(text)
                searches.append((order, (it.get("arguments") or {}).get("query", ""), files, "sg"))
                order += 1
        elif t == "web_search":
            tool_counts["WebSearch"] = tool_counts.get("WebSearch", 0) + 1
            native_calls += 1
    # Retrieval surface, as in the Claude records: an arm with SG scores its sg_search
    # calls; the native arm scores its own text searches.
    src = "sg" if any(s[3] == "sg" for s in searches) else "shell"
    use = [s for s in searches if s[3] == src]
    search_calls, seen = [], set()
    for od, q, files, _ in use:
        gih = sorted(gold & set(files))
        seen |= set(gih)
        search_calls.append({"turn": od, "query": q[:300], "hits": files[:50], "n_hits": len(files),
                             "gold_in_hits": gih,
                             "precision": round(len(gih) / len(files), 4) if files else 0.0,
                             "cumulative_recall": round(len(seen) / len(gold), 4) if gold else 0.0,
                             "error": False})
    first = use[0][2] if use else []
    fin = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    cwrite = usage.get("cache_write_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    pr = PRICES.get(model, PRICES[DEFAULT_MODEL])
    cost = ((max(fin - cached - cwrite, 0)) * pr["input"] + cached * pr["cached"]
            + cwrite * pr["cache_write"] + out * pr["output"]) / 1e6
    return {
        "n_turns": len(items),
        "total_input_tokens": fin, "cached_input": cached, "cache_creation_input": cwrite,
        "billed_input": max(fin - cached - cwrite, 0), "billed_output": out,
        "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
        "cost_usd": cost,
        "tool_counts": tool_counts, "n_tool_calls": sg_calls + native_calls,
        "sg_tool_calls": sg_calls, "native_tool_calls": native_calls,
        "search_calls": search_calls,
        "first_search_fqns": first, "all_search_fqns": sorted({f for s in use for f in s[2]}),
        "retrieval_hit": bool(gold & set(first)),
        "retrieval_precision": round(len(gold & set(first)) / len(first), 4) if first else 0.0,
        "retrieval_rank": next((i for i, f in enumerate(first, 1) if f in gold), 0),
    }


def run_one_task(task: dict, arm: str, model: str, effort: str, timeout: int) -> dict:
    tag = _model_tag(model)
    rid = run_id(task["task_id"], arm, 0, tag)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    repo = prepare_repo(task, _COPY_ARM[arm], verbose=True)
    assert repo == _repo_dir(task, _COPY_ARM[arm])
    reset_repo(repo)
    if arm == ARM_CX_SG:
        _write_agents_md(repo)
    try:
        run = run_codex(repo, task["query"], model, effort, timeout, arm)
        patch = extract_patch(repo)
    finally:
        if arm == ARM_CX_SG:
            _remove_agents_md(repo)
    if not run["ok"] and _QUOTA.search(str(run["error"] or "") + run["stderr"][-2000:]):
        reset_repo(repo)
        raise QuotaExhausted(str(run["error"] or run["stderr"][-300:]))
    meta = parse_codex(run["objs"], task.get("gold_files", []), repo, model)
    if arm == ARM_CX_SG and meta["sg_tool_calls"] and not meta["search_calls"]:
        print(f"  WARN {task['task_id']}: sg tools called but no sg_search result parsed")
    pm = _patch_metrics(patch)
    gold = task.get("gold_files", [])
    stopped = "timeout" if run["timed_out"] else ("submit" if run["ok"] else "error")
    version = ""
    try:
        version = subprocess.run([CODEX, "--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    record = {
        "run_id": rid, "task_id": task["task_id"], "arm": arm, "model": tag, "model_full": model,
        "reasoning_effort": effort, "repeat": 0, "stopped": stopped, "harness": "codex-cli",
        "harness_version": version, "repo": task.get("repo", ""),
        "base_commit": task.get("base_commit", ""), "gold_files": gold, "model_patch": patch,
        "edited_gold_file": _edited_gold(patch, gold),
        "n_turns": meta["n_turns"], "billed_input": meta["billed_input"],
        "billed_output": meta["billed_output"], "cached_input": meta["cached_input"],
        "cache_creation_input": meta["cache_creation_input"],
        "total_input_tokens": meta["total_input_tokens"],
        "reasoning_output_tokens": meta["reasoning_output_tokens"],
        "peak_context_tokens": None,              # not exposed by Codex
        "imputed_cost": round(meta["cost_usd"], 6), "cost_source": "price sheet",
        "wall_s": round(time.time() - t0, 1), "agent_exit": run["exit"],
        "error": None if stopped == "submit" else (run["error"] or run["stderr"][-500:] or "timeout"),
        "leak_flags": leak_flags(run["objs"]),
        "tool_counts": meta["tool_counts"], "n_tool_calls": meta["n_tool_calls"],
        "sg_tool_calls": meta["sg_tool_calls"], "native_tool_calls": meta["native_tool_calls"],
        "retrieval_hit": meta["retrieval_hit"], "retrieval_precision": meta["retrieval_precision"],
        "retrieval_rank": meta["retrieval_rank"], "search_calls": meta["search_calls"],
        "n_search_calls": len(meta["search_calls"]),
        "first_search_fqns": meta["first_search_fqns"], "all_search_fqns": meta["all_search_fqns"],
        "files_read": [], "edit_attempts": [],
        "patch_lines_added": pm["lines_added"], "patch_lines_removed": pm["lines_removed"],
        "patch_files_touched": pm["files_touched"], "patch_hunks": pm["hunks"],
    }
    (config.RUNS_DIR / f"{rid}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    tdir = config.RUNS_DIR / "_codex_transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{rid}.jsonl").write_text(run["raw"], encoding="utf-8")
    if run["stderr"]:
        (tdir / f"{rid}.stderr.txt").write_text(run["stderr"], encoding="utf-8")
    return record


def _already_done(task: dict, arm: str, tag: str) -> bool:
    p = config.RUNS_DIR / f"{run_id(task['task_id'], arm, 0, tag)}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("stopped") == "submit"
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks in dataset order")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (overrides --limit)")
    ap.add_argument("--shard", default="", help="'k/N' strided shard")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true", help="re-run tasks that already have a record")
    args = ap.parse_args()
    if not config._RUN_TAG:
        raise SystemExit("set SG_EVAL_RUN_TAG (e.g. codex_v1) so Codex records get their own directory")
    tasks = load_tasks(Path(args.dataset))
    if args.tasks:
        want = [t.strip() for t in args.tasks.split(",") if t.strip()]
        tasks = [t for t in tasks if t["task_id"] in set(want)]
    elif args.limit:
        tasks = tasks[:args.limit]
    sh = _parse_shard(args.shard)
    if sh:
        tasks = tasks[sh[0] - 1::sh[1]]
    tag = _model_tag(args.model)
    if not args.force:
        tasks = [t for t in tasks if not _already_done(t, args.arm, tag)]
    print(f"Codex [{args.arm}] model={args.model} effort={args.effort} tag={config._RUN_TAG} "
          f"| {len(tasks)} tasks | results -> {config.RUNS_DIR}")
    for i, t in enumerate(tasks, 1):
        try:
            r = run_one_task(t, args.arm, args.model, args.effort, args.timeout)
            print(f"  [{i}/{len(tasks)}] {t['task_id']}: {r['stopped']} turns={r['n_turns']} "
                  f"tokens={r['total_input_tokens']:,} cost=${r['imputed_cost']:.3f} "
                  f"sg_calls={r['sg_tool_calls']} gold_edit={r['edited_gold_file']} {r['wall_s']}s"
                  + (f"  ERROR: {str(r['error'])[:150]}" if r["stopped"] != "submit" else "")
                  + (f"  LEAK-ATTEMPT x{len(r['leak_flags'])}" if r["leak_flags"] else ""))
        except QuotaExhausted as e:
            print(f"  [{i}/{len(tasks)}] {t['task_id']}: usage limit reached, run discarded.\n"
                  f"  {e}\n  Stopped. Rerun the same command after the limit resets; "
                  f"finished tasks are skipped.")
            break
        except Exception as e:
            print(f"  [{i}/{len(tasks)}] {t['task_id']} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
