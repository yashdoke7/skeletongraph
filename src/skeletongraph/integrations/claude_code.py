"""Claude Code Integration — SkeletonGraph-Enhanced CLAUDE.md Template."""

CLAUDE_MD_TEMPLATE = """# CLAUDE.md — SkeletonGraph-Enhanced Rules

## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, token-minimal context assembly.

### Rules:
1. **ALWAYS** use the `query_context` MCP tool before reading files manually.
   It returns attention-optimized context with constraints, target code, and structure.
2. **NEVER** read more than 3 files manually if SkeletonGraph is available.
3. **RESPECT** the constraints in Zone 1 of every context response.
4. If context confidence is LOW, use `search_index` to find the right entry point.
5. Use `expand_function` for page-fault expansion when you need a specific function body.
6. Use `review_delta` when reviewing code changes — it computes blast radius automatically.

### What SkeletonGraph provides:
- Zone 1: Project constraints (primacy position — always read these first)
- Zone 2: Target code bodies (near prompt — strongest attention)
- Zone 3: Structural context (compressed neighbors and dependencies)
- Zone 4: Your current task (recency position)

### Available MCP Tools:
- `query_context` — Main entry: prompt → assembled context
- `expand_function` — Get full body of a specific function
- `show_graph` — Dependency visualization
- `search_index` — Keyword search across index
- `index_status` — Health check
- `review_delta` — Diff-aware blast radius analysis
- `get_blast_radius` — Impact analysis for a function
- `get_dependencies` — Dependency chain
- `detect_changes` — Risk-scored change analysis
- `get_stats` — Token savings dashboard
""".strip()
"""
