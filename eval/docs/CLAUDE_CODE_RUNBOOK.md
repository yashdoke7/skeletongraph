# Runbook — Real Claude Code + SkeletonGraph (MCP) on SWE-bench

The **`sg-rerank` arm driven by the actual `claude` CLI**, not the controlled
ReAct harness. Each task runs in a persistent editable repo copy where SG is the
project's MCP server (`.mcp.json`), so Claude reaches code through
`sg_search`/`sg_get`/`sg_expand` (engine-side **sg-rerank**) exactly as a real
user would. Native Read/Grep/Edit stay enabled — the honest "SG available,
prefer it" setup. Results land in the **same run-JSON schema** the existing
`verify.py` / `aggregate.py` consume, so this arm folds into pass@1 + the tables
unchanged.

Driver: [`eval/agent/run_claude_code.py`](../agent/run_claude_code.py).

---

## 0. Prerequisites (check once)

```powershell
(Get-Command sg).Source        # sg.exe on PATH
(Get-Command claude).Source    # claude.exe on PATH
claude --version               # >= 2.x
```

- **Claude must be logged in** (run `claude` once interactively, sign in, exit).
- **Dataset**: `C:\Users\ASUS\Desktop\CS\Projects\swebench-data\swebench_100.jsonl`
  (100 tasks; source clones already in `…\swebench-data\repos\`).
- **Disk**: editable copies live in `…\swebench-data\_claude_repos\<task_id>\`.
  Budget ~10–30 GB for 100 copies (repo + `.skeletongraph` index each). They are
  persistent (not auto-deleted) so you can inspect/re-run.
- **Defender**: the copy + `git init` retries handle transient AV file locks, but
  excluding `…\swebench-data\_claude_repos` from real-time scanning makes prepare
  much faster.

---

## 1. Isolate this run from the vLLM/NIM results

Set a **distinct run tag** in every terminal you use for this arm. This gives the
Claude arm its own results dir (`eval\results\agent\claude_sgrr\`) and never
touches `nemotron_v3`.

```powershell
$env:SG_EVAL_RUN_TAG = "claude_sgrr"
```

> The arm name is fixed to `sg-rerank`; the run **model** tag is derived from
> `--model` (e.g. `sonnet` → `sonnet`), so run ids are
> `…__sg-rerank__sonnet__r0` and never collide with the NIM `sg-rerank` runs
> (which use `…__sg-rerank__main__r0` under the `nemotron_v3` tag).

---

## 2. Pre-stage all editable copies (one-time, no API cost)

Copies the repo at base commit, builds a clean git baseline, runs `sg build`
(index) + `sg install --ide claude-code` (`.mcp.json` + hooks + `CLAUDE.md`).
All SG/agent files are **gitignored**, so they never enter the patch.

```powershell
$ds = "C:\Users\ASUS\Desktop\CS\Projects\swebench-data\swebench_100.jsonl"
python -m eval.agent.run_claude_code --dataset $ds --prepare-only
```

Idempotent — re-running skips already-prepared copies. Use `--rebuild` to wipe
and redo a copy from scratch. Pre-staging here means the index build happens
**once per repo** (not per run) and the 5 worker terminals below never contend on
`sg build`.

> Doing this while the NIM run is still going? It is CPU-heavy (tree-sitter
> parse per repo). Either wait for `nemotron_v3` to finish, or run it but expect
> slower indexing while both compete for cores.

---

## 3. Run 4–5 Claude Code CLIs, one task-shard each

Open **5 terminals**. In each, set the tag, then run **one shard**. Strided
sharding (`k/5`) gives every terminal a balanced repo mix. Each terminal drives
one `claude` process at a time over its ~20 tasks; 5 terminals = 5 concurrent
agents.

```powershell
# ---- every terminal first ----
$env:SG_EVAL_RUN_TAG = "claude_sgrr"
$ds = "C:\Users\ASUS\Desktop\CS\Projects\swebench-data\swebench_100.jsonl"

# ---- terminal 1 ----
python -m eval.agent.run_claude_code --dataset $ds --model sonnet --shard 1/5
# ---- terminal 2 ----
python -m eval.agent.run_claude_code --dataset $ds --model sonnet --shard 2/5
# ---- terminal 3 ----
python -m eval.agent.run_claude_code --dataset $ds --model sonnet --shard 3/5
# ---- terminal 4 ----
python -m eval.agent.run_claude_code --dataset $ds --model sonnet --shard 4/5
# ---- terminal 5 ----
python -m eval.agent.run_claude_code --dataset $ds --model sonnet --shard 5/5
```

- **Model**: `--model sonnet` (default, cost-sane) or `--model opus` / a full id.
  Keep it the same across all five for a clean comparison.
- **Resumable**: a task whose run JSON exists with `stopped=submit` is skipped, so
  a crashed terminal just re-runs the same command. Add `--force` to redo.
- **Timeout**: `--timeout 1200` (default) kills a stuck task and records
  `stopped=timeout` (excluded from pass@1 like an error).
- **One terminal, several agents**: `--workers 3` drives 3 claude processes from a
  single terminal (each task has its own copy dir, so they don't collide). Prefer
  separate terminals for clarity; use `--workers` only if you want fewer windows.

Per-task line shows: `stopped`, turns, **`sg_calls`** (how many SG MCP tools the
agent actually used), `edited_gold`, tokens, $cost, wall-seconds. Full
stream-json transcripts are saved under
`eval\results\agent\claude_sgrr\_claude_transcripts\` for later analysis.

---

## 4. Verify (pass@1) and aggregate

Run with the **same tag** so verify/aggregate read the Claude results dir. Use a
**fresh harness run-tag** to avoid the WSL/SWE-bench verification cache
(re-using an old tag skips re-running Docker tests).

```powershell
$env:SG_EVAL_RUN_TAG = "claude_sgrr"
# Windows:
python -m eval.agent.verify --all --only-arms sg-rerank --run-tag claude_sgrr_v1
python -m eval.agent.aggregate
```

If you verify on WSL/Linux (Docker harness there), mirror the run JSONs to that
checkout, export `SG_EVAL_RUN_TAG=claude_sgrr`, and run the same `verify` command
with a fresh `--run-tag` (e.g. `claude_sgrr_v1`). Delete any stale
`sg-rerank.claude_sgrr_v1.json` in the harness CWD before re-verifying.

---

## 4b. Where the results live (how to traverse a run)

Everything for this arm is under `eval\results\agent\claude_sgrr\`:

- **`_INDEX.md`** — auto-generated table of every run: stopped, turns, SG vs
  native tool calls, **peak context window**, total input tokens, output, cost,
  wall-seconds, edited-gold, and the transcript path. Start here. Refreshed after
  each task.
- **`<run_id>.json`** — the full record (e.g.
  `astropy__astropy-8707__sg-rerank__sonnet__r0.json`): `model_patch`, all token/
  cost/context metrics, `tool_counts`.
- **`_claude_transcripts\<run_id>.jsonl`** — the complete Claude Code stream-json:
  every assistant message, tool call (SG + native), and tool result. This is the
  session trace to replay what the agent did.
- **Editable copy** `…\swebench-data\_claude_repos\<task_id>\` — the actual repo
  the agent edited. `git -C <copy> diff HEAD` shows its changes; the `.skeletongraph`
  index and `.mcp.json` live here too.

Comparison signals captured per run (maximised from the Claude CLI): `imputed_cost`
($), `billed_input`/`cached_input`/`cache_creation_input`/`total_input_tokens`,
`billed_output`, **`peak_context_tokens`** (context-window high-water mark),
`n_turns`, `wall_s`, and `tool_counts` / `sg_tool_calls` / `native_tool_calls`.

---

## 5. What the patch is (so results are trustworthy)

- Baseline = the repo copy at the task's base commit, committed clean (SG state
  gitignored). The agent edits files directly.
- Patch = `git add -A && git diff --cached HEAD` — captures modified **and**
  new files, with `.skeletongraph/`, `.mcp.json`, `.claude/`, `CLAUDE.md`,
  `.sg_prepared` all excluded by `.gitignore`.
- The official SWE-bench harness applies this patch to the real base-commit image
  in Docker and runs FAIL_TO_PASS / PASS_TO_PASS — identical scoring to every
  other arm.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `/mcp` shows skeletongraph not connected (interactive check) | confirm `sg serve --path <copy>` runs by hand; ensure `sg` on PATH |
| Agent never calls `sg_*`, only greps | expected sometimes; the `sg_calls` column tracks it. CLAUDE.md + `--append-system-prompt` already nudge SG-first |
| Every task `stopped=error`, exit≠0 | `claude` not logged in, or model id invalid — run `claude` once interactively |
| `PREPARE FAILED … git … file-lock` | Defender locked a file mid-copy; re-run prepare (idempotent), or exclude `_claude_repos` from AV |
| Patch shows `.skeletongraph`/`.mcp.json` | a copy was prepared before the gitignore fix — re-prepare it with `--rebuild` |
| Disk filling up | copies are persistent; delete `…\_claude_repos\<task_id>` for finished tasks, or the whole dir after verify |

---

## 7. Where this sits in the plan

1. **Now** — Claude Code + SG (`sg-rerank`) on `swebench_100`, while/after
   `nemotron_v3` finishes. ← this runbook
2. **nemotron pro** — same arms on the Pro (multi-language) set via NIM.
3. **swe pro** — Claude Code + SG on the Pro set (re-point `--dataset`).
4. **AMD** — move per findings; decide conference/workshop vs. post-results-then-paper.

The memory layer (summary-index / hierarchical paging) is a **separate** future
contribution — it needs a long-context/multi-turn benchmark (LongMemEval /
LoCoMo), not SWE-bench. Do not fold it into these results.
