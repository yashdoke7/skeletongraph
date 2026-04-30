"""Claude Code Integration — SkeletonGraph-Enhanced CLAUDE.md Template."""

CLAUDE_MD_TEMPLATE = """# CLAUDE.md — SkeletonGraph-Enhanced Rules

## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, graph-powered context assembly.
It builds a structural map of the codebase (call graphs, dependencies, blast radius)
and returns precisely the code you need — no blind grep through every file.

### Rules:
1. **Before searching or reading files** for a code task, call the `query_context` MCP tool with the user's prompt. It returns a pre-assembled context with:
   - Exact function bodies you need to edit (Zone 2 — target code)
   - Structural context: neighbors, callers, dependencies (Zone 3)
   - Project constraints (Zone 1)
   - Related test files and file structure
2. **RESPECT** the constraints in Zone 1 of every context response — read them first.
3. Use `expand_context` only if you need full bodies of specific functions that were returned as skeletons.
4. Use your normal Glob/Grep/Read tools for any additional detail — do NOT call other SG tools.
5. If `query_context` confidence is LOW, fall back to your native search tools.

### What the context gives you:
- Zone 1: Project constraints (primacy position — always read first)
- Zone 2: Target code bodies (near prompt — strongest attention)
- Zone 3: Structural context (compressed neighbors and dependencies)
- Zone 4: Your current task (recency position)
""".strip()
