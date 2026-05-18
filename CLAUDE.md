## SkeletonGraph (SG) — context assistant

SG is active for this project. Follow these rules every session:

1. **Call `sg_overview` first** — at session start, before reading any files.
   It is the project briefing: project purpose, important structure, constraints,
   recent decisions/turns, and index status.
2. **Use `sg_search` as a task-context assembler, not as grep.**
   Ask for the whole task/symptom once. For coding/debug tasks it returns likely
   edit targets, imports/prelude, helper bodies, graph neighbors, and likely tests.
   Do not split one task into many symbol searches unless confidence is LOW/MISS
   or the target is absent.
3. **Use `sg_get` / `sg_expand` only for exact follow-up.**
   Expand a specific FQN only when you are about to edit it and the body was not
   already in `sg_search`. Do not read MCP `content.txt` result files.
4. **Check `sg_constraint` before proposing changes** — see project rules that must not be violated.
5. **Use `sg_log` to review recent session turns** — avoids re-reading history.

Available MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
