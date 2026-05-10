# SkeletonGraph

**Graph-based context assembly and model routing for AI coding workflows.**

SkeletonGraph indexes a codebase into function-level structure, dependency
edges, tests, and project metadata. It then builds compact context packets for
IDE agents and CLI model runs, so the model starts with the code it most likely
needs instead of burning turns on broad exploration.

SkeletonGraph has two product surfaces:

- **SG IDE**: MCP context server for Cursor, Claude Code, Copilot, Codex,
  Antigravity, Windsurf, and other agentic IDEs.
- **SG CLI**: terminal pipeline for route, prepare, dry-run, provider execution,
  and cost-aware model selection.

## Why SkeletonGraph

Most coding agents spend expensive turns discovering the repo:

```text
search -> read file -> read neighbor -> read tests -> realize the target
```

SkeletonGraph moves that work into a deterministic graph pipeline:

```text
prompt -> classify task -> find target nodes -> expand graph -> assemble packet
```

The goal is not only lower token cost. The useful product outcomes are:

- fewer exploratory file reads
- faster first useful answer
- better target/test/blast-radius context
- transparent routing reasons
- lower model overkill for routine tasks
- reusable packets for IDEs, CLIs, and other agents

## Install

```bash
pip install skeletongraph
```

For provider-backed CLI execution:

```bash
pip install "skeletongraph[llm]"
```

## Quick Start: SG IDE

Use this path when you already work inside Cursor, Claude Code, Copilot, Codex,
Antigravity, or another MCP-capable coding environment.

```bash
cd your-project
sg init
sg build
sg doctor
```

`sg init` writes the MCP config and the agent instruction file for the selected
IDE. SG IDE does not require an API key. Your IDE subscription/model still does
the reasoning and editing; SkeletonGraph supplies the context.

Supported IDE setup targets include:

| IDE | Integration | Model switching |
| --- | --- | --- |
| Cursor | MCP + rules | manual in IDE |
| Claude Code | MCP + `CLAUDE.md` | `/model` command |
| GitHub Copilot | MCP + instructions | manual in IDE |
| Codex | MCP + `AGENTS.md` | manual in agent |
| Antigravity | MCP + rules | manual in IDE |
| Windsurf | MCP + rules | manual in IDE |

## Quick Start: SG CLI

Use this path when you want a terminal-first context and model-routing pipeline.

```bash
cd your-project
sg build
sg route "fix the auth token validation bug"
sg prepare "fix the auth token validation bug" --out .skeletongraph/context.md
sg run "fix the auth token validation bug" --dry-run
```

`sg route`, `sg prepare`, and `sg run --dry-run` do not need an API key.

To call a provider:

```bash
sg config --cli-provider anthropic
$env:ANTHROPIC_API_KEY = "..."
sg run "fix the auth token validation bug" --execute
```

To test locally without a paid provider key:

```bash
ollama pull qwen3-coder:latest
ollama serve
sg config --cli-provider local
sg run "fix the auth token validation bug" --dry-run
sg run "fix the auth token validation bug" --execute
```

Local execution is intended for cheap pipeline testing. Use provider models for
quality benchmarks unless the benchmark is specifically for local models.

## Model Routing

SkeletonGraph separates IDE-facing model labels from CLI provider model names.

For IDEs, model tiers are recommendations:

| Tier | Typical use |
| --- | --- |
| SLM | docs, explanations, simple lookup |
| MLM | normal coding, debugging, tests, review |
| LLM | architecture, broad migrations, low-confidence tasks |

For CLI execution, SkeletonGraph can route to provider model names:

```bash
sg config --cli-provider anthropic
sg config --cli-provider openai
sg config --cli-provider google
sg config --cli-provider local
```

Dynamic routing uses task mode, confidence, candidate count, token size, and
complexity. Code-changing work keeps an MLM floor by default so cost savings do
not come from making weak models edit code unsafely.

## CLI Reference

| Command | Purpose | API key |
| --- | --- | --- |
| `sg init` | configure project, IDE, MCP, rules | no |
| `sg build` | index source files and graph | no |
| `sg doctor` | check index, routing, provider readiness | no |
| `sg route "task"` | show task mode, tier, model route | no |
| `sg prepare "task"` | create/copy context packet | no |
| `sg run "task" --dry-run` | plan routed execution | no |
| `sg run "task" --execute` | call configured provider/local model | provider or local |
| `sg query "task"` | inspect assembled context | no |
| `sg config` | configure IDE and CLI models | no |
| `sg status` | show index status | no |
| `sg metrics` | show logged metrics | no |

Provider output from `sg run --execute` is written under:

```text
.skeletongraph/runs/
```

Automatic patch application is intentionally not enabled yet. The next release
milestone is safe diff parsing, approval gates, `sg verify`, and `sg runs`.

## Python API

```python
from skeletongraph.engine import SGEngine

engine = SGEngine(project_root=".")
result = engine.query("fix the content-length bug", delivery="cli")

print(result.context_text)
print(result.query_mode)
print(result.model_tier)
print(result.recommended_model)
print(result.routing_reason)
```

## Architecture

```text
src/skeletongraph/
  parser/       AST extraction
  graph/        dependency graph and ranking
  storage/      .skeletongraph persistence
  retrieval/    classification, resolution, model routing
  assembly/     context packet construction
  session/      memory and dedup
  server/       MCP server
  llm/          LiteLLM wrapper for optional CLI execution
  cli/          Click commands
  engine.py     unified query pipeline
```

## Evaluation

The architecture/pipeline blueprint and evaluation plan are in:

```text
docs/blueprint.md
docs/evaluation.md
```

SkeletonGraph should be evaluated on both quality and cost:

- target recall and packet completeness
- missed tests/callers
- first useful answer latency
- file reads after SG context
- pass rate
- cost per passing task
- dynamic routing overkill/underpower rate
- IDE compliance with SG-first context usage

Cost savings are only meaningful when reported with pass rate.

## Current Status

SG IDE is a context pipeline. SG CLI now has route, prepare, dry-run execution,
provider/local configuration, and provider output logging. The remaining work
before a stronger CLI release is safe patch apply, verification, run history,
and benchmark-backed routing results.

## License

MIT
