"""Codex CLI Integration — SkeletonGraph-Enhanced AGENTS.md Template."""

CODEX_AGENTS_TEMPLATE = """# AGENTS.md — SkeletonGraph-Enhanced Rules

## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, graph-powered context assembly.

### Rules:
1. **Before searching or reading files** for a code task, call the `query_context` MCP tool with the user's prompt. It returns a pre-assembled context with:
   - Exact function bodies you need to edit (Zone 2 — target code)
   - Structural context: neighbors, callers, dependencies (Zone 3)
   - Project constraints (Zone 1)
   - Related test files and file structure
2. **RESPECT** the constraints in Zone 1 of every context response — read them first.
3. Use `expand_context` only if you need full bodies of specific functions that were returned as skeletons.
4. Use your normal shell/cat/grep tools for any additional detail — do NOT call other SG tools.
5. If `query_context` confidence is LOW, fall back to your native search tools.
""".strip()
