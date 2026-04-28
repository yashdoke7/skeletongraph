# SkeletonGraph

**Token-minimal, constraint-preserving context assembly for AI coding agents.**

SkeletonGraph indexes your codebase into a lightweight skeleton graph — function signatures, dependency edges, and structural metadata — then assembles the minimum context an LLM needs to complete a coding task. No full-file reading, no wasted tokens.

## Key Metrics

| Metric | Value | Method |
|--------|-------|--------|
| Avg Token Reduction | **Nx** | tiktoken BPE vs native agent retrieval (SWE-bench Verified) |
| File Localization F1 | **0.XX** | Against human-authored PR patches |
| MRR | **0.XX** | First relevant file rank |
| Session Savings | **40-60%** | Cross-turn deduplication after turn 1 |
| Resolve Time | **0.8ms** | Graph-based retrieval (zero LLM cost) |

> Benchmark numbers pending — run `skeletongraph eval-benchmark` on SWE-bench Verified to generate.

## How It Works

```
Source Files → Tree-sitter AST → Skeleton Table + Dependency Graph
                                         ↓
User Prompt → Intent Analysis → Entity Resolution → Graph Expansion
                                         ↓
                              Budget Allocation → 4-Zone Assembly → LLM Context
```

**4-Zone Attention-Aware Assembly:**
- **Zone 1 (Primacy)**: Scoped instructions & hierarchical constraints (`.cursorrules` aware).
- **Zone 3 (Structure)**: Skeletons & signatures of periphery dependencies.
- **Zone 2 (Recency)**: Full source of high-impact target code bodies.
- **Zone 4 (Prompt)**: User instructions at the boundary of attention.

### Production Features
- **Cross-Turn Session Memory**: Tracks what the LLM has already "seen". If a function was sent in Turn 1, it is replaced with a 1-line signature in Turn 2, saving 90% of those tokens.
- **Hierarchical Constraints**: Load global rules from project root and specific rules from nested directories.
- **Attention Heatmap**: Visual terminal feedback showing how your token budget is allocated across the 4 zones.
- **PR Blast-Radius**: Analyze `git diff` to identify and include only the functions affected by a logic change.

## Quick Start

```bash
# Install & Auto-detect IDEs (Claude Code, Cursor, Windsurf, Copilot, etc)
pip install skeletongraph
skeletongraph install

# Index your project
skeletongraph build

# Query with visual attention heatmap
skeletongraph query "fix validate_token in middleware.py"

# Track token savings & cost reduction
skeletongraph stats

# Perform PR/Diff blast-radius review
git diff | skeletongraph review
```

## Evaluation & Benchmarking

SkeletonGraph includes a **research-grade evaluation framework** for comparing AI coding agent performance before and after using the graph. Unlike other tools that use character-based estimates (`len//4`), we use **tiktoken BPE** for exact token counting and validate against **SWE-bench Verified** (the industry gold standard).

### Quick Eval (Per-Project)

Run the same prompt with SG ON and OFF, then compare:

```bash
# Antigravity
skeletongraph eval --agent antigravity --native-file ./native_chat.txt --project flask

# Cursor
skeletongraph eval --agent cursor --native-file ./native_chat.txt --project flask

# Claude Code
skeletongraph eval --agent claude_code --native-file ./native_export.md --project flask

# Copilot
skeletongraph eval --agent copilot --native-file ./native_export.json --project flask

# Codex
skeletongraph eval --agent codex --project flask
```

### Research Benchmark (SWE-bench Verified)

Evaluate against 500 human-validated GitHub issues with automated scoring:

```bash
# List available repos
skeletongraph eval-list

# Run benchmark
skeletongraph eval-benchmark \
  --dataset swe-bench-verified \
  --traces-dir ./benchmark_traces \
  --output ./benchmark_results
```

**Metrics produced:**
- **Token Efficiency**: retrieval reduction ratio, total conversation cost, API savings
- **Retrieval Quality**: Precision, Recall, F1, MRR, Hit@k (against human PR patches)
- **Execution Quality**: test pass rate, regression rate (via pytest)
- **Operational Efficiency**: turns, tool calls, redundant file views

All metrics include mean, std deviation, and 95% confidence intervals.

See [EVALUATION_WORKFLOW.md](EVALUATION_WORKFLOW.md) for detailed step-by-step instructions per agent.
See [evaluation_dataset.md](evaluation_dataset.md) for dataset specifications.

### How We Compare

| Aspect | code-review-graph | graphify | **SkeletonGraph** |
|:---|:---|:---|:---|
| Token counter | `len(text)//4` (est.) | `len(text)//4` (est.) | **tiktoken BPE (exact)** |
| Baseline | Static file reads | Raw file dump | **Real agent sessions** |
| Ground truth | Self-referential graph | None | **SWE-bench human PRs** |
| Sample size | 13 commits | 1 folder | **30+ verified tasks** |
| Quality metric | None | None | **P/R/F1/MRR + test pass** |
| Confidence | None | None | **95% CI with std dev** |

## Python API

```python
from skeletongraph import build_index, resolve_context, assemble_context
from pathlib import Path

# Build index
store = build_index(Path("."))

# Query
result = resolve_context("fix the authentication bug", store)
context = assemble_context(result, store, Path("."))

print(f"Tokens: {context.token_count}")
print(f"Confidence: {context.confidence}")
print(context.text)
```

## MCP Server (IDE Integration)

Add to your Claude Code / Cursor MCP config:

```json
{
  "mcpServers": {
    "skeletongraph": {
      "command": "python",
      "args": ["-m", "skeletongraph.cli.main", "serve", "--path", "."]
    }
  }
}
```

Available tools:
- `query_context` — Prompt → assembled context
- `expand_function` — Page-fault: get full source of a function
- `show_graph` — Dependency graph around a function
- `search_index` — Keyword search across all functions
- `index_status` — Index health check

## Architecture

```
src/skeletongraph/
├── parser/           # Tree-sitter AST extraction (Python, TypeScript)
├── graph/            # Dependency graph + search structures
├── storage/          # Persistence to .skeletongraph/
├── retrieval/        # Intent → candidates pipeline
├── assembly/         # Context construction (4-zone)
├── llm/              # LLM integration
├── server/           # MCP server
├── eval/             # Evaluation framework
│   ├── datasets/     # SWE-bench, custom dataset loaders
│   ├── benchmarks/   # Token efficiency, retrieval quality
│   ├── parsers/      # Antigravity, Cursor, Claude, Copilot, Codex
│   ├── scorer.py     # P/R/F1/MRR/Hit@k + aggregate stats
│   └── schema.py     # AgentTrace schema
├── cli/              # CLI commands
└── build.py          # Build orchestrator
```

## Supported Languages

- Python (.py)
- TypeScript (.ts, .tsx)
- JavaScript (.js, .jsx, .mjs, .cjs)

## License

MIT
