# SkeletonGraph

**Token-minimal, constraint-preserving context assembly for AI coding agents.**

SkeletonGraph indexes your codebase into a lightweight skeleton graph — function signatures, dependency edges, and structural metadata — then assembles the minimum context an LLM needs to complete a coding task. No full-file reading, no wasted tokens.

## Key Metrics (Production Hardened)

| Metric | Value | Description |
|--------|-------|-------------|
| Avg Token Reduction | **2.5×** | vs raw file reading (avg 1000 tokens → 400) |
| Success Rate (Recall) | **100%** | on golden dataset (auth, logic, cross-file cases) |
| Session Savings | **40-60%** | saved after turn 1 via cross-turn deduplication |
| Resolve Time | **0.8ms** | graph-based retrieval (zero LLM cost) |
| Multi-Turn Deduplication | **✅ YES** | remembers what the LLM read before |
| Scoped Constraints | **✅ YES** | hierarchical directory-level rules |

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

### 🧠 Production Features
- **Cross-Turn Session Memory**: Tracks what the LLM has already "seen". If a function was sent in Turn 1, it is replaced with a 1-line signature in Turn 2, saving 90% of those tokens.
- **Hierarchical Constraints**: Load global rules from project root and specific rules from nested directories (e.g. `services/auth/.skeletongraph/constraints.md`).
- **Attention Heatmap**: Visual terminal feedback `[██████░░░]` showing how your token budget is allocated across the 4 zones.
- **PR Blast-Radius**: Analyze `git diff` to identify and include only the functions affected by a logic change.

## Quick Start

# Install & Auto-detect IDEs (Claude, Cursor, Windsurf, etc)
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
│   ├── ast_extractor.py
│   ├── edge_extractor.py
│   ├── node_kinds.py
│   ├── skeleton.py
│   └── languages/    # Language-specific rules
├── graph/            # Dependency graph + search structures
│   ├── dependency.py # BFS traversal algorithms
│   ├── bloom.py      # Probabilistic existence check
│   └── inverted_index.py
├── storage/          # Persistence to .skeletongraph/
│   ├── dirty.py      # Incremental change tracking
│   └── local.py      # Atomic JSON serialization
├── retrieval/        # Intent → candidates pipeline
│   ├── intent.py     # Entity extraction + task classification
│   ├── budget.py     # Elastic token budget
│   └── resolver.py   # Graph-based context retrieval
├── assembly/         # Context construction
│   └── zone_assembler.py
├── llm/              # LLM integration
│   ├── provider.py   # LiteLLM abstraction
│   └── summarizer.py # Batch function summarization
├── server/           # MCP server
│   └── mcp.py
├── cli/              # CLI commands
│   └── main.py
└── build.py          # Build orchestrator
```

## Supported Languages

- Python (.py)
- TypeScript (.ts, .tsx)
- JavaScript (.js, .jsx, .mjs, .cjs)

## License

MIT
