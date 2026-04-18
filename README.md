# 🦴 SkeletonGraph

> **Token-minimal, constraint-preserving context assembly for AI coding agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SkeletonGraph parses your codebase into function-level skeletons and a dependency graph, then assembles the **minimum context** an LLM needs to handle any coding task — typically **4-10× fewer tokens** than reading raw files.

## How It Works

```
Your Codebase                    SkeletonGraph                         LLM Context
┌──────────────┐                ┌─────────────────┐                 ┌────────────────────┐
│ 500 files     │  Tree-sitter  │ Skeleton Table   │   Retrieve +   │ Zone 1: Constraints│
│ 5000 functions│ ──────────► │ Dependency Graph │ ──Assemble──► │ Zone 2: Target Code│
│ ~500K tokens  │   + Summary   │ Inverted Index   │   (~735 tok)   │ Zone 3: Skeletons  │
│               │               │ Bloom Filter     │                │ Zone 4: Prompt     │
└──────────────┘                └─────────────────┘                 └────────────────────┘
                                     ~20K tokens                       4-10× reduction
```

## Quick Start

```bash
pip install skeletongraph

# Build index for your project
skeletongraph build

# Query your codebase
skeletongraph query "what does validate_token depend on?"

# Start MCP server for IDE integration
skeletongraph serve

# Standalone chat mode
skeletongraph chat --model gpt-4o-mini
```

## IDE Integration

```bash
# Claude Code
skeletongraph install --platform claude-code

# Google Antigravity  
skeletongraph install --platform antigravity

# Cursor
skeletongraph install --platform cursor

# GitHub Copilot
skeletongraph install --platform copilot
```

## Key Features

- **Skeleton-First Retrieval**: Function signatures + 1-line summaries = 25× cheaper than full bodies
- **Dependency Graph**: Blast-radius analysis — know what breaks before you change it
- **Constraint Zones**: HierMem-inspired attention-aware assembly — constraints never get lost
- **Dynamic Budget**: Elastic token allocation — expands for complex tasks, compresses for simple ones
- **Output Modes**: TERSE / STANDARD / DETAILED — control output token usage
- **Zero-LLM Retrieval**: 70%+ of queries resolved with pure graph traversal, no LLM calls
- **Incremental Updates**: < 1 second on file save (function-level dirty tracking)
- **Multi-Language**: Python, TypeScript/JavaScript (Phase 1). Go, Rust (Phase 2).

## Research

SkeletonGraph builds on [HierMem](https://github.com/.../llm-hiermem), extending its constraint-preserving zone architecture from conversational memory to local file context management.

**Paper**: Coming soon.

## License

MIT
