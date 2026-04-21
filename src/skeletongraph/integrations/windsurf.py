"""
Windsurf Integration — SkeletonGraph-Enhanced .windsurfrules Template.
"""

WINDSURF_RULES_TEMPLATE = """# Windsurf Rules — SkeletonGraph-Enhanced

## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, token-minimal context assembly.

### Rules:
1. **ALWAYS** use the `query_context` MCP tool before reading files manually.
2. **NEVER** read more than 3 files manually if SkeletonGraph is available.
3. **RESPECT** the constraints in Zone 1 of every context response.
4. Use `review_delta` when reviewing code changes.

### Available Tools:
query_context, expand_function, show_graph, search_index, index_status,
review_delta, get_blast_radius, get_dependencies, detect_changes, get_stats
""".strip()
"""
