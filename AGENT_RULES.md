# SkeletonGraph Agent Rules

**These rules are MANDATORY for all IDE agents using SkeletonGraph.**

## Core Rule: Always Use SkeletonGraph

When you need to understand codebase context, you MUST:
1. Use `query_context` or `get_retrieval_context` tool from SkeletonGraph MCP server
2. Never manually read files for context retrieval—only for editing
3. Never make assumptions about code structure—always query the index first

**Exception**: You may read files directly for:
- Viewing code you're about to edit
- Validating SG's context before making changes
- Understanding syntax/AST of a specific file (use SG for cross-file context)

## Rule 1: Project Metadata Must Stay Current

When you detect a change in project structure, tech stack, or constraints:

### Updates to `.skeletongraph/project.md`:
- If adding a new language/framework: Update the project description
- If discovering a new constraint: Add it to constraints section
- If architecture changes: Note it in decisions section
- When phase changes (e.g., refactoring starts): Update phase field

### Updates to `.skeletongraph/architecture.md`:
- After modifying module structure: Update architecture diagram
- After adding/removing major components: Update component descriptions
- If design patterns change: Document new patterns

**Implementation**:
```
# Don't just edit code—update metadata too
1. Make code changes
2. Update .skeletongraph/project.md if needed
3. Update .skeletongraph/architecture.md if needed
4. Call report_completion() to rebuild index
```

## Rule 2: Comment Hygiene (Docstring-First)

SkeletonGraph is docstring-first. Comments are the source of truth for summaries.

### When adding a new function:
```python
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user against database and return True if successful.
    
    Also logs failed attempts for security auditing.
    """
    # implementation
```
✅ Good: Clear docstring as first thing

### When summary is inaccurate:
- Update the function docstring, NOT just SkeletonGraph config
- Then call `report_completion()` to rebuild index with new docstring

### Never:
- ❌ Create function without docstring
- ❌ Leave "TODO: add docstring" placeholders
- ❌ Update SkeletonGraph summary config instead of actual docstring

## Rule 3: Trigger Index Rebuild After Completion

After finishing a task, ALWAYS call `report_completion`:

```python
# At the end of your agent workflow
tool.call("report_completion", {
    "task": "description of what you accomplished",
    "files_modified": ["path/to/file1.py", "path/to/file2.py"],
    "metadata_updated": True  # if you changed project.md/architecture.md
})
```

This triggers:
- Index rebuild with new code changes
- Docstring extraction for new functions
- BM25 corpus update
- Embeddings refresh (if enabled)

**Without this**, next query will use stale index.

## Rule 4: Never Override Agent Instructions with Metadata

SkeletonGraph metadata (project.md, architecture.md) provides CONTEXT, not INSTRUCTIONS.

**Hierarchy**:
1. Your agent instructions (highest priority)
2. `.skeletongraph/project.md` constraints (followed when reasonable)
3. Index signals (used for context selection only)

**Example**:
- If project.md says "No SQL database", follow it
- If project.md says "Always use async/await", follow it UNLESS your task explicitly requires sync code
- If project.md says "No external deps", but task requires `numpy`, use agent judgment (note the decision)

## Rule 5: BM25 Fallback Behavior

When you query SkeletonGraph and get "no direct match":

1. **SG will automatically try BM25 fallback** (semantic search over docstrings)
2. You don't need to do anything—just interpret results normally
3. If BM25 results are poor:
   - Check if docstrings are accurate (Rule 2)
   - Add missing docstrings
   - Call `report_completion()` to rebuild

## Rule 6: Config Is Autonomous

These SkeletonGraph configs are AUTO-ENABLED:
- `enable_bm25_fallback: true` — fallback to semantic search
- `summary_use_docstrings: true` — docstrings are primary summaries
- `auto_rebuild_on_completion: true` — index rebuilds after tasks
- `enable_embeddings: true` (if available) — semantic embeddings for context

**You do NOT need to change these.** They are designed for your workflow.

## Breaking These Rules

What happens if you ignore these:

| Rule | If Broken | Impact |
|------|-----------|--------|
| Use SG for context | Read files manually | Slow, context misses, expensive LLM calls |
| Metadata current | Don't update project.md | Agent loses architectural context, poor decisions |
| Docstring hygiene | Skip docstrings | SG summaries become useless, fallback activates |
| Report completion | Forget to rebuild | Next query uses stale index, context misses |
| Config changes | Disable BM25/docstrings | Loses key features, manual LLM summarization costs |

## Implementation Checklist

When you start work on a SkeletonGraph project:

- [ ] Read `.skeletongraph/project.md` to understand scope/constraints
- [ ] Use `query_context("your question")` to get ranked context, not manual reads
- [ ] When modifying code: update docstrings (not SkeletonGraph config)
- [ ] When discovering patterns: update `.skeletongraph/architecture.md`
- [ ] At task end: call `report_completion()` to rebuild index
- [ ] Never edit `.skeletongraph/config.json` to "fix" summaries—fix docstrings instead

## Example: Correct Workflow

```
Task: Add user authentication module

1. Query: "How does current auth system work?"
   → Use get_retrieval_context() tool
   → Read returned context (don't search manually)

2. Code: Add new functions with clear docstrings
   def verify_password(hash, pwd):
       """Compare password against stored bcrypt hash."""

3. Metadata: Update project.md
   - Add constraint: "Auth module must use bcrypt"
   - Update description: "Now includes JWT token validation"

4. Architecture: Update architecture.md
   - Add Auth module to component diagram
   - Document new auth flow

5. Complete: Call report_completion()
   → Index rebuilds with new docstrings
   → Next query will find new functions

6. Verify: Query "authentication" again
   → Should return your new functions
   → Docstrings are summaries (not LLM generated)
```

## Questions?

If a rule conflicts with your task:
1. Document the conflict in a comment
2. Proceed with task but note the exception
3. Update project.md with the constraint violation (if intentional)
