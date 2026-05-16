## SkeletonGraph (SG) — context assistant

SG is active for this project. Follow these rules every session:

1. **Call `sg_overview` first** — at session start, before reading any files.
   It shows the top functions by PageRank, active constraints, and recent turns.
2. **Use `sg_search` instead of grep/glob** — hybrid BM25 + graph centrality search.
3. **Use `sg_get` / `sg_expand` instead of reading full files** — token-efficient retrieval.
4. **Check `sg_constraint` before proposing changes** — see project rules that must not be violated.
5. **Use `sg_log` to review recent session turns** — avoids re-reading history.

Available MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
