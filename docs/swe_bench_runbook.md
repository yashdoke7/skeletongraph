# SWE-bench evaluation runbook for SkeletonGraph

End-to-end procedure for benchmarking SG vs native (no-SG) on SWE-bench, across
Claude Code, GitHub Copilot, Cursor, Antigravity, and Codex. The goal is a
**clean apples-to-apples comparison** of tokens, success rate, and tool calls
between native agents and SG-augmented agents on the same SWE-bench tasks.

## TL;DR

1. **Docker once** to get a clean Python 3.11 + git + dependencies env so SWE-bench
   tasks don't fail on version mismatch.
2. **Smoke test:** 1 task on Claude Code, native vs SG, manually verify everything
   works end-to-end.
3. **Full run:** 20–30 tasks on Claude Code (token-rich logging), 20–30 on Cursor,
   ~10 each on Copilot / Antigravity / Codex.
4. **Compare** SG-on vs SG-off per agent, then cross-agent.

---

## 0. Prerequisites

- Docker Desktop (Windows) — for the build env
- Python 3.11 on host (matches `requires-python` in `pyproject.toml`)
- API keys for whichever providers you'll use:
  - `ANTHROPIC_API_KEY` (Claude Code)
  - `OPENAI_API_KEY` (Codex)
  - GitHub Copilot subscription
  - Cursor account
  - Antigravity (Gemini) key
- ~30 GB disk for cloned repos + SWE-bench dataset

---

## 1. Docker build for clean evaluation env

Different SWE-bench repos use different Python versions (3.6, 3.7, 3.8, 3.9, 3.10).
Trying to run them all on one host Python wastes hours on dependency hell. Solve
once by using SWE-bench's own Docker images.

```bash
# On Windows, run in PowerShell or WSL bash
git clone https://github.com/princeton-nlp/SWE-bench /opt/swe-bench
cd /opt/swe-bench
pip install -e .

# Pull pre-built per-instance Docker images (SWE-bench official)
python -m swebench.harness.run_evaluation --help

# Pull dataset (verified split, 2294 problems)
python -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-bench_Verified', split='test')"
```

Then in your `Dockerfile` for the SG side, install SG into the image:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git build-essential
WORKDIR /app
COPY . /app/skeletongraph
RUN pip install -e /app/skeletongraph[llm]
# optional: install Ollama for Tier-0.5
# RUN curl -fsSL https://ollama.com/install.sh | sh
ENTRYPOINT ["sg"]
```

Build once: `docker build -t sg:0.1 .`

You only need Docker if you want isolated per-task Python envs. For evaluating
the **agent + SG pipeline** (not SWE-bench's patch verification), the host
Python is fine — SG just reads the cloned repo and indexes it.

---

## 2. Pick the SWE-bench split

| Split | Size | Use |
| --- | --- | --- |
| `SWE-bench_Lite` | 300 | smoke + first 30 tasks |
| `SWE-bench_Verified` | 500 | main benchmark — human-verified, fewer false negatives |
| `SWE-bench` | 2294 | full split, only if you have compute |

**Use Verified.** Lite skews toward easy tasks; Verified is the standard for
publishable comparisons.

Pick 20–30 task IDs spanning small / medium / large repos (django, flask, sympy,
sklearn, requests, pylint, etc.). Save them to `eval/swe_bench_subset.txt`:

```text
django__django-11099
flask__flask-4045
pytest-dev__pytest-5103
sympy__sympy-13895
scikit-learn__scikit-learn-13496
... (25 more)
```

---

## 3. Per-task setup

For each task ID:

```bash
# Pseudocode — adapt to your harness
TASK_ID=django__django-11099
REPO_URL=https://github.com/django/django.git
COMMIT=<base_commit from SWE-bench instance>

# 1. Clone at base commit
git clone $REPO_URL eval/runs/$AGENT/$TASK_ID/repo
cd eval/runs/$AGENT/$TASK_ID/repo
git checkout $COMMIT

# 2. Initialize SG (only for SG-on runs)
sg init --non-interactive --agent claude_code
sg build
sg install --ide claude-code
# Optional: pre-summarize for cleaner runs
# sg summarize --tier local        # Ollama Tier-0.5
# sg summarize --tier cloud --force  # Tier-1 if you want best quality

# 3. Hand the task to the agent — the agent reads the problem_statement
#    from SWE-bench and produces a patch
```

The native (no-SG) baseline runs steps 1 and 3 only. **Same repo state, same
prompt, same model.** Only difference: SG hooks + MCP server are active or not.

---

## 4. Per-agent procedure

### 4.1 Claude Code (token-rich)

This is the agent that gives you the cleanest token telemetry.

**Native (no-SG):**
1. `cd eval/runs/claude_code_native/$TASK_ID/repo`
2. `claude --model sonnet-4-5 --print "$(cat problem_statement.md)" 2>&1 | tee transcript.txt`
   - or interactive: `claude` then paste the problem statement
3. After it produces a patch, export the transcript:
   - `/export transcript_$TASK_ID.json` inside the Claude Code session
   - Or: read the JSON log under `~/.claude/projects/<encoded-path>/<uuid>.jsonl`
4. Extract tokens from the JSONL — each message has `usage: {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}`. Sum across all assistant turns.

**SG-on:**
1. `cd eval/runs/claude_code_sg/$TASK_ID/repo`
2. `sg init --non-interactive --agent claude_code && sg build && sg install --ide claude-code`
3. Run Claude Code the same way as native. SG hooks fire automatically.
4. Export transcript the same way.
5. Also collect `.skeletongraph/last_hook.log`, `.skeletongraph/session/*.jsonl`, and `.skeletongraph/metrics.jsonl` if it exists.

**Token extraction script** (Claude Code JSONL → totals):
```python
import json
from pathlib import Path
def tokens_from_session(jsonl_path):
    totals = {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0}
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
            u = ev.get("message", {}).get("usage") or ev.get("usage")
            if not u: continue
            totals["input"] += u.get("input_tokens", 0)
            totals["output"] += u.get("output_tokens", 0)
            totals["cache_create"] += u.get("cache_creation_input_tokens", 0)
            totals["cache_read"] += u.get("cache_read_input_tokens", 0)
        except Exception:
            pass
    totals["total"] = sum(totals.values())
    return totals
```

### 4.2 Cursor (token-rich, in-IDE)

Cursor 1.7+ shows token counts per request in the sidebar and persists them in
the local SQLite DB at `%APPDATA%\Cursor\User\workspaceStorage\<hash>\state.vscdb`.

**Native:**
1. Open repo in Cursor with no `.cursor/rules/skeletongraph.mdc` present.
2. Set model to `Sonnet 4.6` (or whatever you're benchmarking).
3. Paste the SWE-bench `problem_statement` into Composer.
4. Let it run to completion. Save chat: **Cmd/Ctrl+Shift+P → "Cursor: Export Chat as Markdown"**.
5. Token totals appear in the export markdown.

**SG-on:**
1. `sg install --ide cursor` in the repo — writes `.cursor/mcp.json` + rules.
2. Restart Cursor to pick up MCP server.
3. Run the same prompt. Export the chat.
4. Compare token totals.

### 4.3 GitHub Copilot

Less token transparency, but you can get **request count** and approximate
context size from the Copilot panel (when "Show Details" is on).

**Native:**
1. Disable SG MCP (remove `.vscode/mcp.json` skeletongraph entry).
2. Use Copilot Chat in VS Code: paste problem_statement.
3. Export: **Cmd/Ctrl+Shift+P → "Chat: Export Session…"** → JSON.

**SG-on:**
1. `sg install --ide copilot` — writes `.vscode/mcp.json` + `.github/copilot-instructions.md`.
2. Restart VS Code.
3. Run prompt, export.

Tokens: parse the exported JSON's `requests` field — each request has token
counts since Copilot 1.245+.

### 4.4 Antigravity

Tokens visible in the chat sidebar per request. Less programmatic export:

**Native / SG-on:** same pattern. After completion, screenshot the token panel
or use **Settings → Export Logs** to dump JSONL.

### 4.5 Codex (OpenAI)

Codex CLI logs are at `~/.codex/sessions/`. Each session has a `usage.json`
with token totals.

```bash
codex --model gpt-5.2 < problem_statement.md > out.diff
cat ~/.codex/sessions/$(ls -t ~/.codex/sessions | head -1)/usage.json
```

---

## 5. Smoke test (do this first)

Pick **one easy task** (e.g. `flask__flask-4045` or `pytest-dev__pytest-5103`)
and run it twice with Claude Code: once native, once SG-on. Confirm:

- [ ] SG built the index successfully (`.skeletongraph/index.json` exists)
- [ ] SG MCP server registered (`/mcp` in Claude Code lists `skeletongraph`)
- [ ] `sg_overview` returned constraints + top functions
- [ ] Agent actually called `sg_search` / `sg_get` (check `.skeletongraph/last_hook.log` for `post_tool_use` entries naming the SG tools)
- [ ] Token totals from the JSONL look sane (not zero, not 10×native)
- [ ] Both runs produced a patch (success unimportant for smoke; just non-empty diff)

If any of the above fails, fix before the full run.

**Likely failure modes:**
- MCP server doesn't start → `sg doctor` shows missing fields
- Hooks don't fire → check `.claude/settings.json` was written correctly
- Cold-start error → `auto_build_on_query` is now True by default (just fixed); upgrade SG and re-run

---

## 6. Full run

Once smoke passes:

| Agent | Tasks | Logging quality |
| --- | --- | --- |
| Claude Code | 25 | Best — exact API token counts per turn |
| Cursor | 25 | Good — per-request tokens visible |
| Copilot | 10 | OK — request counts + approximate |
| Antigravity | 10 | Manual screenshot or export logs |
| Codex | 10 | OK — usage.json per session |

Run each task **twice**: once native, once SG-on. Save outputs under:

```text
eval/runs/<agent>_<mode>/<task_id>/
  repo/                  # cloned repo at base commit
  transcript.jsonl       # or .md depending on agent
  usage.json             # token totals
  patch.diff             # final patch
  .skeletongraph/        # SG-on only
```

---

## 7. Scoring

Patch correctness — use SWE-bench's own harness:

```bash
python -m swebench.harness.run_evaluation \
  --predictions_path eval/runs/<agent>_<mode>/predictions.jsonl \
  --max_workers 4 \
  --split test \
  --run_id <agent>_<mode>_<date>
```

Outputs `pass@1`. Combine with token totals for `cost per pass`.

---

## 8. What to publish

For Cursor and Claude Code specifically (since their telemetry is richest),
publish:

| Metric | Native | SG-on | Delta |
| --- | --- | --- | --- |
| pass@1 | x% | y% | +Δ |
| avg input tokens / task | a | b | -Δ% |
| avg output tokens / task | c | d | -Δ% |
| avg tool calls / task | e | f | -Δ |
| avg file-read calls / task | g | h | -Δ |
| avg time / task | t1 | t2 | -Δ |

Plus qualitative: agent transcripts on 2–3 representative tasks showing
SG-on calling `sg_search` first vs native doing grep+read-loop.

---

## 9. Reuse-friendly files

Save the SWE-bench subset under `eval/swe_bench_subset.txt` and the run logs
under `eval/runs/<agent>_<mode>/<task_id>/`. Re-running with newer SG is then
just `git pull && pip install -e . && python scripts/rerun.py eval/swe_bench_subset.txt`.

Currently the old `eval/` scaffolding has been cleaned out of SG itself — the
SWE-bench harness is the source of truth.
