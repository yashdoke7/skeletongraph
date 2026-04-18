"""
Cursor Integration Templates.
"""

CURSOR_RULES_TEMPLATE = """# SkeletonGraph Context Instructions

This workspace uses [SkeletonGraph] as its primary codebase indexer.

## Your Context Strategy
When a user asks a complex multi-file architectural question or requests a bug fix, you should minimize blindly using natural-language codebase retrieval or `grep`. 

Instead, use the `resolve_context` MCP tool. 
- Pass the user's direct prompt to it.
- It will return an optimized 'Skeleton Map' showing exact dependencies, blast-radius graphs, and full function bodies of the target intent.

### Retaining Your Cognitive Loop
**IMPORTANT**: SkeletonGraph provides context. It does not plan. 
You must retain your internal step-by-step thinking loop. Continue using your scratchpad capabilities. If the graph yields an empty result, gracefully fall back to your native file reading capabilities without dropping your thought process.
"""
