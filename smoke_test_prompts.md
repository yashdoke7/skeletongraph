# Isolated Smoke Test Workflow

## Universal Prompt
For every agent, you will use this exact prompt:
> Fix the trailing slash routing issue in Blueprints. Ensure strict_slashes config is respected.

## Execution Steps per Agent

Because the repos are isolated, you will CD into each agent's specific folder before running.

### 1. Claude Code
```powershell
cd public_eval_runs/runs/claude_code/flask__flask-smoke/flask
# Run Native:
claude
# Run SG:
skeletongraph build
claude --mcp ... (your mcp config)
```

### 2. Copilot
```powershell
cd public_eval_runs/runs/copilot/flask__flask-smoke/flask
# Open in VS Code:
code .
# Run Native and SG in Copilot chat.
```

### 3. Cursor
```powershell
cd public_eval_runs/runs/cursor/flask__flask-smoke/flask
# Open in Cursor:
cursor .
# Run Native and SG in Composer.
```

### 4. Antigravity
```powershell
cd public_eval_runs/runs/antigravity/flask__flask-smoke/flask
# Run Native and SG via your standard execution script.
```

### 5. Codex
```powershell
cd public_eval_runs/runs/codex/flask__flask-smoke/flask
# Run Native and SG via CLI or IDE extension.
```

## Parsing
Once the runs are complete, populate the `native_trace.json` and `sg_trace.json` inside `benchmark_traces/<agent>/flask__flask-smoke/` and run the aggregation:

```powershell
skeletongraph eval-benchmark --dataset swe-bench-verified --traces-dir benchmark_traces/claude_code
```
