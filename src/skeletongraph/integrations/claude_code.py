"""
Claude Code Integration Templates.
"""

CLAUDE_MD_TEMPLATE = """# Codebase Navigation with SkeletonGraph

This repository is indexed with **SkeletonGraph**. 

## CRITICAL: Context Hunting Workflow
When asked to investigate a bug, implement a feature, or explore the code, you **MUST** prioritize using the SkeletonGraph MCP tools to resolve context FIRST, before reverting to standard `grep` or `glob`.

1. Run the `resolve_context` MCP tool with the user's natural language prompt.
2. The tool will return a highly optimized **Zone Assembly** (Constraints, Target Skeletons, Structural Neighbours).
3. If the confidence is HIGH or MEDIUM, base your solution heavily on this context. 
4. If you need to see the *entire body* of a neighbor function (which may have been compressed to a signature), you can use your native `read_file` tool *only on those specific exact lines*.

## CRITICAL: Memory and Planning Preservation
**SkeletonGraph is a search optimizer, NOT an agentic brain.**
- You must **continue** to use your internal Scratchpad, memory files, and multi-turn planning logic exactly as normal.
- Do **not** assume `resolve_context` remembers previous turns. You are responsible for preserving constraints and maintaining conversational state.
- If SkeletonGraph returns 'No Match', log this in your scratchpad and fall back gracefully to your native `grep` mechanics.
"""
