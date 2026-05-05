# SkeletonGraph v3 — Evaluation Implementation Brief

## How to Use This Document

You are receiving this brief along with 4 Python/shell scripts and the v3 pipeline plan. Read this document fully before touching any script or running any command. This document explains:

1. What the evaluation framework is trying to prove
2. What scripts exist, what they do, and how they interact
3. What you need to verify against the local setup before running anything
4. How to execute the full evaluation end-to-end
5. What the results mean and what to do with them

The scripts are complete but may need minor adjustments for local paths, installed tool versions, and which repos are available. This document tells you exactly what to check.

---

## What We Are Proving

The core hypothesis: **SkeletonGraph v3's mode-based routing + graph assembly reduces LLM token consumption by >5x vs native agent file exploration, without reducing correctness.**

Secondary hypotheses:
- PLANNING mode (no code bodies) saves tokens vs STANDARD for non-coding queries
- REVIEW mode (session memory only) is cheap and accurate for summary queries
- REFACTOR mode + BLAST_FIRST modifier prevents breaking callers
- Mode classifier routes correctly to expected mode >80% of the time

This is not just a cost benchmark. It is a correctness + cost + routing accuracy benchmark simultaneously.

---

## Files You Have Received

### `eval/scripts/setup_workspaces.py`
**What it does:** Clones all 40 task repos, checks out the correct commits, saves task metadata, and generates one runbook markdown file per task × agent combination (200 runbook files total).

**What it contains:** The complete TASKS registry with all 40 tasks across 5 modes (CODE_FIX/FAST × 5, CODE_FIX/STANDARD × 10, PLANNING × 10, REVIEW × 5, REFACTOR × 5, DEBUG × 5). Each task has: repo URL, commit hash, problem statement, expected mode, expected SG token count, test command, eval checklist.

**What you must verify before running:**
- `sg` CLI is installed and on PATH: `sg --version`
- `git` is installed: `git --version`
- You are running from the project root (where `eval/` will be created)
- Commit hashes in the TASKS registry match actual SWE-bench verified commits. The hashes in the script are placeholders — you must replace them with real SWE-bench verified commit hashes before cloning. See "Commit Hash Verification" section below.

---

### `eval/scripts/run_claude_code.sh`
**What it does:** Fully automated baseline + SG run for a single task on Claude Code. Disables SG, runs native, captures diff + tokens. Re-enables SG, runs with SG, captures diff + hit log + tokens. Calls parse and compare scripts automatically.

**What you must verify before running:**
- `claude` CLI is installed and on PATH: `claude --version`
- `claude -p "..." --output-format json` works in your version. If not, check the current Claude Code CLI flags — `--output-format` may be `--json` or similar depending on version.
- `.claude/settings.json` exists in the task repo after `sg install`. The script backs it up and restores it. If your SG hooks use a different config location, update lines 60-70 of the script.
- `sg build` and `sg status` commands exist in your CLI. If the commands differ (e.g., `skeletongraph build`), update the script accordingly.

---

### `eval/scripts/parse_claude_logs.py`
**What it does:** Parses Claude Code's `--output-format json` output to extract per-turn token counts, tool call breakdown, and exploration cost. Merges with SG hit log. Computes reduction ratio and cost estimate.

**What you must verify:**
- The JSON structure from `claude --output-format json` matches what the parser expects. Claude Code's JSON output format may have changed. The parser expects messages with `role`, `content`, and `usage.input_tokens`/`usage.output_tokens`. Run `claude -p "hello" --output-format json` and inspect the output structure. If different, update `parse_claude_output()` accordingly.
- Token cost constants in `compute_reduction()` use Claude Sonnet 4 pricing ($3/M input, $15/M output). Update these if using a different model.

---

### `eval/scripts/compare_patches.py`
**What it does:** Compares native.patch and sg.patch against golden.patch. Scores each as `exact` (same files + same functions), `partial` (overlapping files), `wrong` (different files), or `empty`. Also contains `aggregate_all()` for report generation.

**What you must verify:**
- Golden patches for SWE-bench tasks must be placed in `eval/runs/<agent>/<task_id>/golden.patch` before comparison. The setup script writes them from the TASKS registry, but the TASKS registry currently has placeholder golden patches. Real golden patches come from the SWE-bench verified dataset. See "SWE-bench Dataset" section below.

---

### `eval/scripts/aggregate_metrics.py`
**What it does:** Loads all per-task `metrics.json` files from `eval/runs/*/`, computes averages by IDE and by mode, checks modifier firing accuracy, flags regressions, and writes `eval/results/benchmark_report.md`.

**What you must verify:**
- Run this only after at least one full task has completed (both native and SG runs)
- It imports from `setup_workspaces.py` to get expected modes per task. Both files must be in the same directory.

---

## Critical: Commit Hash and Golden Patch Verification

The TASKS registry in `setup_workspaces.py` contains placeholder commit hashes and empty golden patches. You must populate these before running `setup_workspaces.py`. Here is exactly how:

### Step 1: Get the SWE-bench verified dataset

```bash
pip install datasets
python3 -c "
from datasets import load_dataset
ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
# Save as JSON for reference
import json
data = [dict(row) for row in ds]
with open('eval/dataset/swe_bench_verified_full.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Downloaded {len(data)} tasks')
"
```

### Step 2: Extract commits and patches for our 20 SWE-bench tasks

```python
import json

ds = json.load(open('eval/dataset/swe_bench_verified_full.json'))
ds_by_id = {row['instance_id']: row for row in ds}

# Our SWE-bench task IDs (from setup_workspaces.py TASKS registry):
our_tasks = [
    "requests-1142", "requests-0963", "requests-2078", "requests-1733", "requests-1776",
    "requests-2153", "requests-2311", "requests-2949", "requests-3070", "requests-3178",
    "django-12345", "django-23456",
    "astropy-7350", "astropy-8872", "astropy-9999",
]

for task_id in our_tasks:
    # SWE-bench uses format like "psf__requests-1142"
    swe_id = task_id.replace("-", "__", 1).replace("-", "-")
    # Try common formats
    for fmt in [task_id, task_id.replace("-", "__", 1), f"psf__{task_id}"]:
        if fmt in ds_by_id:
            row = ds_by_id[fmt]
            print(f"'{task_id}': commit='{row['base_commit']}', patch_len={len(row['patch'])}")
            break
    else:
        print(f"NOT FOUND: {task_id}")
```

### Step 3: Update TASKS registry

For each task found, update `setup_workspaces.py`:
- Replace placeholder `commit` hash with `row['base_commit']`
- Add `golden_patch` field from `row['patch']`

If a task ID is not found in SWE-bench verified, replace it with a verified task from the same repo. The task IDs in the registry were chosen to cover the right difficulty distribution — exact IDs are less important than having the right mix of FAST/STANDARD/DEEP tasks.

### Step 4: For non-SWE-bench tasks (PLANNING, REVIEW, REFACTOR, DEBUG)

These tasks have custom problem statements and no golden patches. They are evaluated differently:
- PLANNING: structural checklist (did the response reference project constraints?)
- REVIEW: memory accuracy (did the summary mention what was done?)
- REFACTOR: test suite pass + blast radius correctness
- DEBUG: modifier fired + diagnosis quality (manual review)

No commit hash update needed for these — they use the same `psf/requests` commit as tasks already verified.

---

## Pre-Run Verification Checklist

Run through this checklist before any evaluation runs. Fix everything that fails.

### Environment

```bash
# 1. SG CLI
sg --version
sg build --help
sg init --help
sg prepare --help    # may not exist yet if not built — note which commands exist

# 2. Claude Code CLI
claude --version
claude -p "say hello" --output-format json   # verify JSON output format
# Inspect the output — does it have messages[].usage.input_tokens?

# 3. Python deps
python3 -c "from datasets import load_dataset; print('datasets OK')"
python3 -c "import json, pathlib, subprocess, argparse; print('stdlib OK')"

# 4. Git
git --version

# 5. Disk space (40 repos × ~50MB each = ~2GB)
df -h .
```

### SG Pipeline Verification (Run Before Benchmark)

```bash
# Pick any single repo to verify the full pipeline works
mkdir /tmp/sg_test && cd /tmp/sg_test
git clone --depth 1 https://github.com/psf/requests.git
cd requests

# Build index
sg build
sg status   # should show function count > 0

# Run sg init
sg init    # answer: "Python HTTP library for making web requests"

# Verify context files created
ls .skeletongraph/
cat .skeletongraph/project.md   # should have your goal text

# Test query_context
sg prepare "fix the content-length bug in GET requests"
cat .skeletongraph/context.md   # should show FAST or STANDARD mode context
cat .skeletongraph/eval/hit_log.jsonl   # should have 1 entry

# Verify mode was correct
python3 -c "
import json
entry = json.loads(open('.skeletongraph/eval/hit_log.jsonl').read().strip().splitlines()[-1])
print('Mode:', entry.get('mode'))
print('Query type:', entry.get('query_type'))
print('Confidence:', entry.get('confidence'))
print('Tokens delivered:', entry.get('tokens_delivered'))
"
# Expected: mode=STANDARD or FAST, query_type=CODE_FIX
```

### Claude Code Hook Verification

```bash
# Verify hooks fire correctly
cd /tmp/sg_test/requests

# Check settings.json exists
cat .claude/settings.json

# Run a simple claude task and verify hit_log updates
> .skeletongraph/eval/hit_log.jsonl
claude -p "what does prepare_content_length do?" --output-format json > /tmp/test_out.json

# Check if hit_log was updated
cat .skeletongraph/eval/hit_log.jsonl
# If empty: hooks are not firing. Check .claude/settings.json hook paths are correct.
# Common issue: hook script paths must be absolute or relative to repo root.
```

### Cursor Manual Verification (Do Once)

```
1. Open /tmp/sg_test/requests in Cursor
2. Verify MCP skeletongraph appears in Settings → MCP Servers
3. Ask: "fix the content-length bug in GET requests"
4. Verify query_context was called (should appear in tool calls in chat)
5. Export chat to CSV — verify CSV has input/output token columns
6. Note the CSV column names for parse_cursor_csv.py
```

---

## Execution Order

### Phase 1: Setup (Do Once)

```bash
# From project root
cd /path/to/skeletongraph

# 1. Verify SWE-bench task IDs and update TASKS registry
python3 eval/scripts/setup_workspaces.py --dry-run
# Review output — does every task show a valid repo URL?

# 2. Download SWE-bench dataset and update commit hashes
# (see "Commit Hash Verification" above)

# 3. Create all workspaces
python3 eval/scripts/setup_workspaces.py --agents all

# 4. For each repo, run sg build + sg init
# This cannot be automated because sg init requires user input
for task_dir in eval/runs/claude_code/*/repo; do
    echo "=== $task_dir ==="
    cd "$task_dir"
    sg build
    # sg init will prompt — answer with 1-2 sentence project description
    sg init
    cd -
done
# Note: Claude Code repos and other agent repos share the same base repo
# You only need to sg build once per unique repo (not once per agent)
# Symlink or copy .skeletongraph/ after the first build
```

### Phase 2: Claude Code Automated Runs (Primary Benchmark)

```bash
# Run all 40 tasks (automated)
bash eval/scripts/run_claude_code.sh all

# Or run individually to verify first
bash eval/scripts/run_claude_code.sh requests-1142
# Check output: did it show FAST mode? Was reduction > 5x?

# If requests-1142 works, run remaining tasks
bash eval/scripts/run_claude_code.sh all --skip-baseline
# (if you already have baselines from a previous run)
```

### Phase 3: Manual Runs (Cursor, Antigravity, Copilot)

For each task in the priority subset (15 tasks for Cursor, 10 for Antigravity, 5 for Copilot):

1. Open `eval/runbooks/<agent>_<task_id>_runbook.md`
2. Follow the runbook exactly
3. Fill in token counts in the runbook
4. Run the comparison script after each task:

```bash
python3 eval/scripts/compare_patches.py \
  --task-id <task_id> \
  --run-dir eval/runs/<agent>/<task_id> \
  --output eval/runs/<agent>/<task_id>/patch_comparison.json
```

5. For Cursor, create `metrics.json` manually from the CSV:

```bash
python3 -c "
import json
from pathlib import Path

# Fill these in from runbook and Cursor CSV
metrics = {
    'task_id': 'requests-1142',
    'agent': 'cursor',
    'tier': 'B',
    'native': {
        'total_input_tokens': FILL_IN,
        'total_output_tokens': FILL_IN,
        'total_tokens': FILL_IN,
        'test_passed': True  # or False
    },
    'sg': {
        'total_input_tokens': FILL_IN,
        'total_output_tokens': FILL_IN,
        'total_tokens': FILL_IN,
        'test_passed': True,
        'sg_mode': FILL_FROM_HITLOG,
        'tokens_delivered': FILL_FROM_HITLOG,
        'sg_hit': FILL_FROM_HITLOG,
    },
    'reduction_ratio': NATIVE_TOTAL / SG_TOTAL,
    'success': True,  # compute manually
    'flags': []
}
Path('eval/runs/cursor/requests-1142/metrics.json').write_text(json.dumps(metrics, indent=2))
"
```

### Phase 4: Aggregation and Calibration

```bash
# Generate final report
python3 eval/scripts/aggregate_metrics.py

# Review report
cat eval/results/benchmark_report.md

# If benchmark passes (>5x reduction, correctness maintained):
# → Proceed to Enhanced mode (SLM routing) — build and test separately

# If benchmark has failures:
# → Follow the diagnosis checklist in the report
# → Fix the identified issue
# → Re-run only the failing tasks
```

---

## REVIEW and PLANNING Mode Evaluation

These modes cannot be auto-evaluated by patch comparison. Use this manual checklist.

### PLANNING Mode Tasks (10 tasks)

For each planning task, after running:

```bash
cat eval/runs/claude_code/<task_id>/sg_hitlog.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    print('Mode:', e.get('mode'))
    print('Tokens delivered:', e.get('tokens_delivered'))
    print('Code bodies in context:', 'NO' if e.get('mode') == 'PLANNING' else 'YES')
"
```

Then open `eval/runbooks/claude_code_<task_id>_runbook.md` and evaluate each checklist item:
- Did the agent reference actual project constraints? (from project.md)
- Were code bodies absent from the context? (PLANNING mode must have no code bodies)
- Did the agent give ≥ 2 distinct approaches?
- Was the response grounded in the actual codebase structure?

Mark each item and record in `eval/runs/claude_code/<task_id>/planning_eval.md`.

### REVIEW Mode Tasks (5 tasks)

These require running 2-4 "prior turn" prompts before the summary prompt:

```bash
# Example for review-session-01
cd eval/runs/claude_code/review-session-01/repo

# Clear hit log
> .skeletongraph/eval/hit_log.jsonl

# Run prior turns first (builds session memory)
claude -p "Fix the content-length bug in GET requests" --output-format json >> ../session_turns.jsonl
claude -p "Also fix the redirect auth header stripping" --output-format json >> ../session_turns.jsonl

# Now run the summary (should trigger REVIEW mode)
claude -p "summarize what we changed and why in this session" --output-format json > ../sg_output.json

# Verify REVIEW mode was used
tail -1 .skeletongraph/eval/hit_log.jsonl | python3 -c "
import json, sys
e = json.loads(sys.stdin.read())
print('Mode:', e.get('mode'))  # should be REVIEW
print('Tokens:', e.get('tokens_delivered'))  # should be ~1200
print('Code bodies loaded:', 'YES' if e.get('mode') != 'REVIEW' else 'NO')
"
```

Evaluate: Did the summary mention both prior-turn changes? Was REVIEW mode used?

---

## Session Memory Verification

Run this to verify session memory is accumulating correctly:

```bash
cd eval/runs/claude_code/requests-1142/repo

# After a completed SG run, check session files
cat .skeletongraph/session/current.md
# Should contain: Turn summary, files modified, key decisions

cat .skeletongraph/session/recent.md
# Should contain: Previous session summaries (if multiple runs done)

# Verify domain notes are being created
ls .skeletongraph/domain/
# If any "decided to..." statements were in agent responses,
# domain notes should have been created

# Verify compression would trigger at right threshold
python3 -c "
from pathlib import Path
recent = Path('.skeletongraph/session/recent.md')
if recent.exists():
    tokens_est = len(recent.read_text()) // 4  # rough token estimate
    print(f'recent.md estimated tokens: {tokens_est}')
    print(f'Compression threshold: 500 tokens')
    print(f'Will compress: {tokens_est > 500}')
"
```

---

## What the Numbers Mean

### Interpreting Reduction Ratio

- **>10x**: Pipeline is working very well. Agent is reading context.md and not exploring.
- **5-10x**: Pipeline is working. Within target range.
- **2-5x**: Pipeline is working but agent is doing some exploration after context. Check hit rate.
- **<2x**: Pipeline is not working. Agent is mostly ignoring context.md. Check hooks delivery.
- **<1x (worse than native)**: SG overhead is causing net increase. Check rules file size and schema overhead.

### Interpreting Hit Rate

- **>80%**: Excellent. Agent reads context.md and executes without further exploration.
- **60-80%**: Good. Occasional exploration needed (expected for MEDIUM confidence tasks).
- **<60%**: Poor. Agent is not trusting context.md. Check rules file directives.
- **0%**: hooks not firing or agent not finding context.md.

### Interpreting Mode Routing Accuracy

Check `eval/results/benchmark_report.md` → "Mode Routing Accuracy" table.
- FAST tasks getting STANDARD: acceptable (over-delivery, slight cost increase)
- STANDARD tasks getting FAST: bad (under-delivery, agent may explore more)
- Any task getting PLANNING when code needed: bad (agent gets no code bodies)
- CODE_FIX task getting PLANNING: investigate classifier bug

### Interpreting Correctness

- **SG pass rate ≥ native pass rate**: Success. Same quality, lower cost.
- **SG pass rate > native pass rate**: Very good. Better quality AND lower cost.
- **SG pass rate < native pass rate**: Failure. Do not publish results. Investigate.
  - Most likely cause: context.md is missing something the agent needed.
  - Check: was the task MISS confidence? (agent explored natively — context was wrong)
  - Check: was mode selection correct? (FAST for something needing STANDARD)

---

## Failure Diagnosis Quick Reference

Copy this into a terminal-friendly format for quick reference during runs:

```
SYMPTOM: Low reduction ratio (<2x)
CHECK 1: cat .skeletongraph/eval/hit_log.jsonl → is sg_mode populated?
         If empty → hooks not firing → check .claude/settings.json hook paths
CHECK 2: Does context.md exist after sg prepare?
CHECK 3: Does CLAUDE.md tell agent to read context.md?
FIX:     sg install → re-run

SYMPTOM: Agent makes 5+ tool calls after context delivery
CHECK 1: Is context.md complete? → cat .skeletongraph/context.md
CHECK 2: Was confidence LOW? → check hit_log confidence field
CHECK 3: Did agent read shadow files or real files?
FIX:     If LOW confidence → expected behavior
         If HIGH confidence miss → check prompt_builder L4 completeness

SYMPTOM: Test fails (sg regression)
CHECK 1: Was native test passing? → cat eval/runs/<agent>/<task>/native_test_result.txt
CHECK 2: What mode was used? Was it FAST when STANDARD needed?
CHECK 3: Did blast radius warn about the breaking callers?
FIX:     Check mode selection criteria → maybe tighten FAST threshold
         Check blast radius output in context.md → was it accurate?

SYMPTOM: Wrong mode selected
CHECK 1: What query type was classified? → hit_log query_type field
CHECK 2: What was the confidence level? → hit_log confidence field
EXAMPLE: CODE_FIX + HIGH → should be FAST. If getting STANDARD → confidence threshold issue
FIX:     Adjust classifier thresholds. Run calibrate_confidence.py after benchmark.

SYMPTOM: PLANNING mode for a coding task (agent gets no code bodies)
CHECK 1: Does the query look like a planning question? → review query text
CHECK 2: Were any entities found? → hit_log entity_match score
FIX:     If truly a coding task → entity extraction failed → check intent.py patterns
         If borderline → accept STANDARD as safer default
```

---

## After the Benchmark: What to Do With Results

### If benchmark passes (>5x reduction, ≥ native correctness)

1. Run `calibrate_confidence.py` to tune weights based on actual data
2. Write up results in `eval/results/benchmark_report.md` — it's auto-generated but add narrative
3. Consider building Enhanced mode (SLM routing for MEDIUM/LOW confidence)
4. Run the same benchmark on Enhanced mode — compare vs Universal

### If benchmark fails

Do NOT build Enhanced mode. Fix Universal first. Most likely fixes in order:
1. Hook delivery for Claude Code (most common failure)
2. Confidence threshold calibration (second most common)
3. L4 depth for specific task types (FAST getting tasks that need STANDARD)

---

## Files Reference Summary

| File | Purpose | Run When |
|---|---|---|
| `setup_workspaces.py` | Clone repos, generate runbooks | Once, before all runs |
| `run_claude_code.sh <task_id>` | Full automated run for 1 task | Per task (Claude Code only) |
| `run_claude_code.sh all` | Run all tasks sequentially | After verifying 1 task works |
| `parse_claude_logs.py` | Extract metrics from Claude Code JSON | Auto-called by run script |
| `compare_patches.py` | Score patches vs golden | Auto-called by run script; manual for GUI agents |
| `aggregate_metrics.py` | Generate final report | After all runs complete |

---

## Checklist Before Handing to Human

Before running anything:
- [ ] Commit hashes in TASKS registry updated from SWE-bench verified dataset
- [ ] Golden patches added to TASKS registry
- [ ] `claude --output-format json` format verified against parser expectations
- [ ] `sg` CLI commands verified (build, init, prepare, status, install)
- [ ] `.claude/settings.json` hook paths verified for your installation
- [ ] Disk space sufficient (~3GB for all repos across 5 agents)
- [ ] At least 1 full pipeline test run completed on /tmp test repo
