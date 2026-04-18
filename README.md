# SkeletonGraph

**Token-minimal, constraint-preserving context assembly for AI coding agents.**

SkeletonGraph indexes your codebase into a lightweight skeleton graph — function signatures, dependency edges, and structural metadata — then assembles the minimum context an LLM needs to complete a coding task. No full-file reading, no wasted tokens.

## Key Metrics (on test fixture)

| Metric | Value |
|--------|-------|
| Avg Token Reduction | **2.0×** vs raw file reading |
| Coverage Score | **90%** of expected functions included |
| Constraint Preservation | **100%** — constraints never dropped |
| High Confidence Rate | **100%** — correct entity resolution |
| Resolve Time | **0.1ms** — zero LLM cost for retrieval |

## How It Works

```
Source Files → Tree-sitter AST → Skeleton Table + Dependency Graph
                                         ↓
User Prompt → Intent Analysis → Entity Resolution → Graph Expansion
                                         ↓
                              Budget Allocation → 4-Zone Assembly → LLM Context
```

**4-Zone Attention-Aware Assembly:**
- **Zone 1** (top): Project constraints → primacy effect
- **Zone 3** (middle): Structural context (signatures, relationships)
- **Zone 2** (above prompt): Target code bodies → recency effect
- **Zone 4** (bottom): User prompt → strongest attention

## Quick Start

```bash
pip install skeletongraph

# Index your project
skeletongraph build

# Query
skeletongraph query "fix validate_token in middleware.py" --verbose

# Incremental update after code changes
skeletongraph update

# Generate LLM summaries (optional, requires API key)
skeletongraph summarize --model gemini/gemini-2.0-flash
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
