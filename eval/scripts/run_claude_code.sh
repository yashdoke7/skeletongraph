#!/bin/bash
# SkeletonGraph v3 — Automated Claude Code Evaluation Runner
# Runs baseline + SG for a single task, captures all metrics
#
# Usage:
#   bash eval/scripts/run_claude_code.sh requests-1142
#   bash eval/scripts/run_claude_code.sh all         # runs all tasks sequentially
#   bash eval/scripts/run_claude_code.sh all --skip-baseline  # SG only

set -euo pipefail

TASK_ID="${1:-}"
SKIP_BASELINE="${2:-}"
EVAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$EVAL_DIR/scripts"

if [[ -z "$TASK_ID" ]]; then
    echo "Usage: $0 <task_id|all> [--skip-baseline]"
    exit 1
fi

if [[ "$TASK_ID" == "all" ]]; then
    # Run all tasks sequentially
    python3 -c "
import json, sys
idx = json.load(open('eval/dataset/index.json'))
for t in idx['tasks']:
    print(t)
" | while read -r tid; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "TASK: $tid"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        bash "$0" "$tid" "$SKIP_BASELINE" || echo "FAILED: $tid (continuing...)"
    done
    echo ""
    echo "All tasks complete. Running aggregation..."
    python3 "$SCRIPTS_DIR/aggregate_metrics.py"
    exit 0
fi

RUN_DIR="$EVAL_DIR/runs/claude_code/$TASK_ID"
REPO_DIR="$RUN_DIR/repo"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: Repo not found at $REPO_DIR"
    echo "Run: python eval/scripts/setup_workspaces.py --tasks $TASK_ID --agents claude_code"
    exit 1
fi

# Get task config
PROBLEM=$(python3 -c "
import sys; sys.path.insert(0, 'eval/scripts')
from setup_workspaces import TASKS
t = TASKS.get('$TASK_ID')
if not t: sys.exit(1)
print(t['problem'])
")

TEST_CMD=$(python3 -c "
import sys; sys.path.insert(0, 'eval/scripts')
from setup_workspaces import TASKS
t = TASKS.get('$TASK_ID')
print(t.get('test_cmd') or '')
")

echo "Task: $TASK_ID"
echo "Run dir: $RUN_DIR"

# ─────────────────────────────────────────────
# BASELINE RUN
# ─────────────────────────────────────────────

if [[ "$SKIP_BASELINE" != "--skip-baseline" ]]; then
    echo ""
    echo "── BASELINE RUN ──────────────────────────"

    cd "$REPO_DIR"
    git reset --hard HEAD 2>/dev/null || true
    git clean -fdx 2>/dev/null || true

    # Disable SG by temporarily moving settings
    if [[ -f ".claude/settings.json" ]]; then
        cp ".claude/settings.json" "$RUN_DIR/settings_backup.json"
        rm ".claude/settings.json"
        echo "  SG disabled (settings.json moved)"
    else
        echo "  No .claude/settings.json found (already baseline)"
    fi

    echo "  Running claude -p ..."
    timeout 300 claude -p "$PROBLEM" \
        --output-format json \
        --max-turns 20 \
        2>&1 | tee "$RUN_DIR/native_output.json"

    # Capture diff
    git diff > "$RUN_DIR/native.patch"
    echo "  Native patch written: $(wc -l < "$RUN_DIR/native.patch") lines"

    # Run tests if available
    if [[ -n "$TEST_CMD" ]]; then
        echo "  Running tests..."
        if eval "$TEST_CMD" > "$RUN_DIR/native_test_output.txt" 2>&1; then
            echo "  ✓ Tests PASSED"
            echo "PASS" > "$RUN_DIR/native_test_result.txt"
        else
            echo "  ✗ Tests FAILED"
            echo "FAIL" > "$RUN_DIR/native_test_result.txt"
        fi
    fi

    # Reset repo
    git reset --hard HEAD 2>/dev/null || true
    git clean -fdx 2>/dev/null || true

    # Re-enable SG
    if [[ -f "$RUN_DIR/settings_backup.json" ]]; then
        cp "$RUN_DIR/settings_backup.json" ".claude/settings.json"
        echo "  SG re-enabled"
    fi

    echo "  Baseline run complete"
fi

# ─────────────────────────────────────────────
# SG RUN
# ─────────────────────────────────────────────

echo ""
echo "── SKELETONGRAPH RUN ─────────────────────"

cd "$REPO_DIR"
git reset --hard HEAD 2>/dev/null || true
git clean -fdx 2>/dev/null || true

# Verify SG is built
if ! sg status > /dev/null 2>&1; then
    echo "  SG index not found. Running sg build..."
    sg build
    echo "  Note: Run 'sg init' to populate project.md if not done"
fi

# Verify SG hooks are enabled
if [[ ! -f ".claude/settings.json" ]]; then
    echo "  WARNING: .claude/settings.json not found — SG hooks may not fire"
    echo "  Run: sg install to set up Claude Code hooks"
fi

# Clear previous hit log
mkdir -p ".skeletongraph/eval"
> ".skeletongraph/eval/hit_log.jsonl"

echo "  Running claude -p with SG enabled..."
timeout 300 claude -p "$PROBLEM" \
    --output-format json \
    --max-turns 20 \
    2>&1 | tee "$RUN_DIR/sg_output.json"

# Capture diff
git diff > "$RUN_DIR/sg.patch"
echo "  SG patch written: $(wc -l < "$RUN_DIR/sg.patch") lines"

# Copy SG metrics
if [[ -f ".skeletongraph/eval/hit_log.jsonl" ]]; then
    cp ".skeletongraph/eval/hit_log.jsonl" "$RUN_DIR/sg_hitlog.jsonl"
    echo "  SG hit log copied: $(wc -l < "$RUN_DIR/sg_hitlog.jsonl") entries"
else
    echo "  WARNING: No hit log found — SG may not have fired"
    touch "$RUN_DIR/sg_hitlog.jsonl"
fi

# Run tests if available
if [[ -n "$TEST_CMD" ]]; then
    echo "  Running tests..."
    if eval "$TEST_CMD" > "$RUN_DIR/sg_test_output.txt" 2>&1; then
        echo "  ✓ Tests PASSED"
        echo "PASS" > "$RUN_DIR/sg_test_result.txt"
    else
        echo "  ✗ Tests FAILED"
        echo "FAIL" > "$RUN_DIR/sg_test_result.txt"
    fi
fi

# Reset repo
git reset --hard HEAD 2>/dev/null || true
git clean -fdx 2>/dev/null || true

# ─────────────────────────────────────────────
# EXTRACT METRICS
# ─────────────────────────────────────────────

echo ""
echo "── EXTRACTING METRICS ─────────────────────"

python3 "$SCRIPTS_DIR/parse_claude_logs.py" \
    --task-id "$TASK_ID" \
    --run-dir "$RUN_DIR" \
    --output "$RUN_DIR/metrics.json"

python3 "$SCRIPTS_DIR/compare_patches.py" \
    --task-id "$TASK_ID" \
    --run-dir "$RUN_DIR" \
    --output "$RUN_DIR/patch_comparison.json"

echo ""
echo "── RESULTS SUMMARY ─────────────────────────"
python3 -c "
import json
from pathlib import Path

metrics_path = Path('$RUN_DIR/metrics.json')
patch_path = Path('$RUN_DIR/patch_comparison.json')

if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    print(f'  Native tokens:  {m.get(\"native_total_tokens\", \"N/A\"):>10,}')
    print(f'  SG tokens:      {m.get(\"sg_total_tokens\", \"N/A\"):>10,}')
    ratio = m.get('reduction_ratio')
    if ratio:
        flag = '✓' if ratio >= 5 else '⚠'
        print(f'  Reduction:      {ratio:>10.1f}x  {flag}')
    print(f'  SG mode:        {m.get(\"sg_mode\", \"N/A\"):>10}')
    print(f'  Hit rate:       {m.get(\"hit_rate\", \"N/A\")}')

if patch_path.exists():
    p = json.loads(patch_path.read_text())
    n_pass = '✓ PASS' if p.get('native_test_passed') else '✗ FAIL' if p.get('native_test_passed') is False else '? N/A'
    s_pass = '✓ PASS' if p.get('sg_test_passed') else '✗ FAIL' if p.get('sg_test_passed') is False else '? N/A'
    print(f'  Native correct: {n_pass}')
    print(f'  SG correct:     {s_pass}')
    if p.get('sg_regression'):
        print(f'  ⚠ REGRESSION: SG failed where native passed')
" 2>/dev/null || echo "  (metrics extraction failed — check raw files)"

echo ""
echo "✓ Task complete: $TASK_ID"
echo "  Results in: $RUN_DIR/"
