#!/usr/bin/env python3
"""
SkeletonGraph v3 — Evaluation Workspace Setup
Clones all repos, checks out commits, builds SG indexes, generates runbooks.

Usage:
    python eval/scripts/setup_workspaces.py
    python eval/scripts/setup_workspaces.py --agents claude_code cursor
    python eval/scripts/setup_workspaces.py --tasks requests-1142
    python eval/scripts/setup_workspaces.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────
# TASK REGISTRY — All 40 evaluation tasks
# ─────────────────────────────────────────────

tasks_file = Path(__file__).parent.parent / "tasks.json"
if not tasks_file.exists():
    print(f"Error: {tasks_file} not found.")
    sys.exit(1)
import json
TASKS = json.loads(tasks_file.read_text(encoding="utf-8"))

AGENTS = ["claude_code", "cursor", "copilot", "codex", "antigravity"]

AGENT_TIERS = {
    "claude_code": "A",
    "cursor": "B",
    "codex": "B",
    "antigravity": "C",
    "copilot": "D",
}

# ─────────────────────────────────────────────
# RUNBOOK GENERATOR
# ─────────────────────────────────────────────

def generate_runbook(task_id: str, task: dict, agent: str, run_dir: Path) -> None:
    tier = AGENT_TIERS[agent]
    mode = task["expected_mode"]
    tokens = task.get("expected_sg_tokens", "~3000")
    modifier = task.get("expected_modifier", "none")
    eval_type = task.get("eval_type", "correctness")
    checklist = task.get("eval_checklist", [])

    # Agent-specific instructions
    disable_sg = {
        "claude_code": "Remove or rename `.claude/settings.json` in the repo root",
        "cursor": "Settings → MCP Servers → skeletongraph → toggle OFF",
        "copilot": "SG not installed for Copilot — baseline is already native",
        "codex": "SG not installed for Codex by default — baseline is native",
        "antigravity": "Disable MCP in Antigravity settings → remove skeletongraph",
    }[agent]

    enable_sg = {
        "claude_code": "Run: `sg build && sg install claude_code` in the repo directory",
        "cursor": "Run: `sg build && sg install cursor` in the repo directory",
        "copilot": "Run: `sg build && sg install copilot` in the repo directory",
        "codex": "Run: `sg build && sg install codex` in the repo directory",
        "antigravity": "Run: `sg build && sg install antigravity` in the repo directory",
    }[agent]

    run_cmd_baseline = {
        "claude_code": f'claude -p "{task["problem"][:80]}..." --output-format json > ../native_output.json',
        "cursor": "Open repo in Cursor. New chat. Paste problem statement below.",
        "copilot": "Open repo in VS Code with Copilot. Open Copilot Chat. Paste prompt.",
        "codex": f'codex "{task["problem"][:80]}..."',
        "antigravity": "Open repo in Antigravity. New session. Paste prompt.",
    }[agent]

    run_cmd_sg = {
        "claude_code": f'claude -p "{task["problem"][:80]}..." --output-format json > ../sg_output.json',
        "cursor": "Open repo in Cursor. New chat. Paste SAME problem statement.",
        "copilot": "Open repo in VS Code. Open Copilot Chat. Paste SAME prompt.",
        "codex": f'codex "{task["problem"][:80]}..."',
        "antigravity": "Open repo in Antigravity. New session. Paste SAME prompt.",
    }[agent]

    token_fields = {
        "A": "Auto-extracted from conversation JSON. No manual input needed.",
        "B": "Export chat CSV from agent UI. Fill in fields below.",
        "C": "Read from UI manually. Fill in fields below.",
        "D": "N/A — Copilot does not expose token counts.",
    }[tier]

    checklist_md = "\n".join(f"- [ ] {item}" for item in checklist)

    runbook = f"""# Runbook: {task_id} — {agent.replace('_', ' ').title()}
**Dataset:** {task['dataset']}
**Expected Mode:** {mode}
**Expected Modifier:** {modifier}
**Expected SG Tokens:** ~{tokens}
**Eval Type:** {eval_type}
**Tier:** {tier} ({"full metrics" if tier == "A" else "session totals" if tier == "B" else "qualitative" if tier == "C" else "correctness only"})

---

## Problem Statement

Paste this EXACTLY — do not modify, do not add context:

```
{task['problem']}
```

---

## Prior Turns (for REVIEW tasks only)
{"N/A" if not task.get("prior_turns") else chr(10).join(f"{i+1}. {t}" for i, t in enumerate(task["prior_turns"]))}

---

## Setup (Run Once Per Task)

```bash
cd eval/runs/{agent}/{task_id}/
ls repo/   # verify repo is cloned
git -C repo log --oneline -1  # should show: {task['commit'][:12]}...
```

If repo not cloned, run: `python eval/scripts/setup_workspaces.py --tasks {task_id} --agents {agent}`

---

## Baseline Run (Native — No SG)

### Step 1: Disable SG
{disable_sg}

### Step 2: Verify baseline
```bash
git -C repo status  # should be clean
```

### Step 3: Run
{run_cmd_baseline}

### Step 4: Capture diff and trace
1. **Save all modified files** in the editor. Do not commit.
2. Export the session trace (Click "View Logs" / "Export Trace" in the chat panel).
3. Save the exported JSON as `eval/runs/{agent}/{task_id}/native_trace.json`.

```bash
cd repo
git diff > ../native.patch
git stash
```

---

## SkeletonGraph Run (v4 Pipeline)

### Step 1: Build + Install (auto-configures MCP + rules files)
```bash
cd repo
sg build                      # Index the codebase
sg install {agent}             # Auto-writes MCP config + agent rules
sg status                      # Verify: should show indexed functions
```

### Step 2: Select the correct model in your IDE
{enable_sg}

### Step 3: Run
{run_cmd_sg}

### Step 4: Capture diff
```bash
cd repo
git diff > ../sg.patch
cp .skeletongraph/eval/hit_log.jsonl ../sg_hitlog.jsonl 2>/dev/null || echo "No hit log"
git stash
```

---

## Comparison

```bash
cd eval/runs/{agent}/{task_id}/

# Compare patches
python ../../scripts/compare_patches.py \\
  --native native.patch \\
  --sg sg.patch \\
  --task-id {task_id} \\
  --agent {agent}

# For automated Claude Code runs, extract full metrics:
{"python ../../scripts/parse_claude_logs.py --native native_output.json --sg sg_output.json --hitlog sg_hitlog.jsonl" if agent == "claude_code" else "# Manual: fill in the fields above"}
```

---

## Verification & Record Results

### Option A: Qualitative Patch Comparison (recommended)
```bash
cd eval/runs/{agent}/{task_id}/

# Compare agent patches to the known-correct golden fix
diff native.patch golden.patch    # Baseline vs golden
diff sg.patch golden.patch        # SG vs golden

# Score: 1 = correct files + correct logic, 0 = wrong files or bad logic
python ../../../../eval/scripts/record_result.py --agent {agent} --task {task_id} --mode baseline --success 1
python ../../../../eval/scripts/record_result.py --agent {agent} --task {task_id} --mode sg --success 1
```

### Option B: Automated Test Harness (may fail on older codebases)
```bash
cd eval/runs/{agent}/{task_id}/
python ../../../../eval/scripts/swe_bench_harness.py
```

---

## Success Criteria

{"- [ ] Test passed (both runs): `" + task['test_cmd'] + "`" if task.get('test_cmd') else "- [ ] Patch matches golden fix (correct files + correct logic)"}
- [ ] SG mode was: **{mode}** (verify in sg_hitlog.jsonl)
- [ ] SG modifier was: **{modifier}** (verify in sg_hitlog.jsonl)

### Mode-Specific Checklist
{checklist_md if checklist_md else "- [ ] No specific checklist for this task"}

---

## Notes
_Fill in any anomalies, unexpected behavior, or observations during the run._

```
Baseline notes: 
SG notes:
Comparison notes:
```
"""

    runbook_path = run_dir / "RUNBOOK.md"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text(runbook, encoding="utf-8")
    print(f"  [OK] Runbook: {runbook_path}")


# ─────────────────────────────────────────────
# WORKSPACE SETUP
# ─────────────────────────────────────────────

def setup_task_for_agent(task_id: str, task: dict, agent: str, dry_run: bool = False) -> None:
    run_dir = Path(f"eval/runs/{agent}/{task_id}")
    repo_dir = run_dir / "repo"
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_url = f"https://github.com/{task['repo']}.git"

    if dry_run:
        print(f"  [DRY] Would clone {repo_url} -> {repo_dir}")
        print(f"  [DRY] Would checkout {task['commit']}")
        print(f"  [DRY] Would run: sg build in {repo_dir}")
        generate_runbook(task_id, task, agent, run_dir)
        return

    # Clone
    if not repo_dir.exists():
        print(f"  Cloning {task['repo']}...")
        result = subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [ERROR] Failed to clone {repo_url}")
            print(result.stderr)
            return
    else:
        print(f"  Repo exists, skipping clone")

    # Checkout commit
    result = subprocess.run(
        ["git", "checkout", task["commit"]],
        cwd=repo_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] Failed to checkout {task['commit']} in {repo_dir}. Run 'git stash' and try again.")
        print(f"  {result.stderr}")
        return

    # Write golden patch if available
    if task.get("golden_patch"):
        (run_dir / "golden.patch").write_text(task["golden_patch"], encoding="utf-8")
        
    # Write test patch if available
    if task.get("test_patch"):
        (run_dir / "swe_bench_tests.patch").write_text(task["test_patch"], encoding="utf-8")

    # Save task metadata
    (run_dir / "task.json").write_text(json.dumps({
        "task_id": task_id,
        "agent": agent,
        "repo": task["repo"],
        "commit": task["commit"],
        "expected_mode": task["expected_mode"],
        "expected_sg_tokens": task.get("expected_sg_tokens"),
        "test_cmd": task.get("test_cmd"),
        "dataset": task["dataset"],
        "eval_type": task.get("eval_type", "correctness"),
    }, indent=2), encoding="utf-8")

    # Generate runbook
    generate_runbook(task_id, task, agent, run_dir)

    # Note: sg build runs separately since it requires the sg CLI to be installed
    sg_note = run_dir / "SG_SETUP.md"
    sg_note.write_text(f"""# SG Setup Required

Run these commands before starting evaluation:

```bash
cd {repo_dir}
sg build                      # Index the codebase
sg install {agent}             # Auto-configure MCP + rules files
sg status                      # Verify index
```

To change model configuration:
```bash
sg config --show               # View current config
sg config --agent {agent}      # Re-apply agent preset
```
""", encoding="utf-8")

    print(f"  [OK] {agent}/{task_id}")


def main():
    parser = argparse.ArgumentParser(description="Setup SkeletonGraph evaluation workspaces")
    parser.add_argument("--agents", nargs="+", choices=AGENTS + ["all"], default=["all"])
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agents = AGENTS if "all" in args.agents else args.agents
    tasks = list(TASKS.keys()) if "all" in args.tasks else args.tasks

    # Validate tasks
    invalid = [t for t in tasks if t not in TASKS]
    if invalid:
        print(f"Unknown tasks: {invalid}")
        sys.exit(1)

    print(f"Setting up {len(tasks)} tasks × {len(agents)} agents = {len(tasks) * len(agents)} workspaces")
    if args.dry_run:
        print("DRY RUN — no files will be created\n")

    # Create directory structure
    for d in ["eval/dataset/code_fix/fast", "eval/dataset/code_fix/standard",
              "eval/dataset/planning", "eval/dataset/review",
              "eval/dataset/refactor", "eval/dataset/debug",
              "eval/results/raw", "eval/runbooks", "eval/scripts"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Setup each workspace
    for task_id in tasks:
        task = TASKS[task_id]
        print(f"\n[{task_id}] ({task['dataset']})")
        for agent in agents:
            setup_task_for_agent(task_id, task, agent, dry_run=args.dry_run)

    # Write master index
    index = {
        "tasks": tasks,
        "agents": agents,
        "total_workspaces": len(tasks) * len(agents),
        "by_dataset": {},
        "by_agent_tier": AGENT_TIERS,
    }
    for task_id in tasks:
        ds = TASKS[task_id]["dataset"]
        index["by_dataset"].setdefault(ds, []).append(task_id)

    Path("eval/dataset/index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"\n[OK] Setup complete.")
    print(f"  Runbooks generated: eval/runbooks/")
    print(f"  Master index: eval/dataset/index.json")
    print(f"\nNext step: For each repo, run:")
    print(f"  cd eval/runs/<agent>/<task_id>/repo && sg build && sg install <agent>")
    print(f"\nTo view/change model config:")
    print(f"  sg config --show")


if __name__ == "__main__":
    main()
