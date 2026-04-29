"""
Kiro Integration — SkeletonGraph-Enhanced rules.
"""

KIRO_RULES_TEMPLATE = """# Kiro Rules — SkeletonGraph-Enhanced

## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, token-minimal context assembly. Use it before broad native search/read operations, but keep native tools available as bounded fallback.

### Rules:
1. Start with the `query_context` MCP tool before broad native search or large file reads.
2. Prefer `search_index`, `expand_function`, and `view_file_range` before full-file reads.
3. **RESPECT** the constraints in Zone 1 of every context response.
4. Use native tools only when SG confidence is LOW, the file is unindexed, or a small verification read is needed.

### Available Tools:
query_context, expand_function, show_graph, search_index, index_status,
view_file_range, grep_codebase, review_delta, get_blast_radius, get_dependencies,
detect_changes, get_stats
""".strip()
