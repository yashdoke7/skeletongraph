# SkeletonGraph Complete Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SkeletonGraph v4 Pipeline                      │
└─────────────────────────────────────────────────────────────────────┘

     CLI FLOW                          IDE FLOW (Cursor/Claude)
     ─────────                         ──────────────────────
        │                                      │
        ├─ python -m skeletongraph             ├─ MCP Server (sg server)
        │  build --path . --auto-infer         │  (listening on stdout)
        │                                      │
        └──→ CLI Pipeline                      └──→ IDE Agent (Cursor/Claude)
            (see below)                            │
                                                   ├─ query_context()
                                                   ├─ get_retrieval_context()
                                                   └─ pack_context()


═══════════════════════════════════════════════════════════════════════

## CLI FLOW (Deterministic, No LLM for Context)

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: INIT (First Build Only)                                    │
└─────────────────────────────────────────────────────────────────────┘

   src/skeletongraph/cli/main.py [sg build --path . --auto-infer]
   │
   ├─→ src/skeletongraph/cli/init.py [run_init()]
   │   │
   │   ├─ Auto-infer project metadata (LLM: Claude Sonnet 4.6)
   │   │  └─→ _infer_metadata_with_llm()
   │   │      Generates: goal, constraints, phase, decisions
   │   │
   │   ├─ Create .skeletongraph/project.md (L0 Project DNA)
   │   ├─ Create .skeletongraph/architecture.md (L1 Architecture)
   │   ├─ Create .skeletongraph/config.json (with flags enabled):
   │   │  ├─ enable_bm25_fallback: true
   │   │  ├─ summary_use_docstrings: true
   │   │  ├─ auto_rebuild_on_completion: true
   │   │  └─ enable_embeddings: true
   │   │
   │   └─ Setup directories:
   │      ├─ .skeletongraph/session/ (conversation logs)
   │      ├─ .skeletongraph/domain/ (skeleton index)
   │      └─ .skeletongraph/eval/ (evaluation results)


┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: DISCOVER & PARSE (Every Build)                            │
└─────────────────────────────────────────────────────────────────────┘

   src/skeletongraph/cli/main.py [build()]
   │
   ├─→ src/skeletongraph/build.py [discover_files()]
   │   Find all supported files (.py, .ts, .java, etc)
   │   └─ Use .skeletongraphignore + .gitignore
   │
   └─→ src/skeletongraph/build.py [build_index()]
       │
       ├─ For each file:
       │  │
       │  ├─→ src/skeletongraph/parser/ast_extractor.py
       │  │   [extract_file()] → tree-sitter AST parsing
       │  │   └─ Extract FQN, docstrings, signatures, body
       │  │      (Docstring-First: docstrings are primary)
       │  │
       │  └─→ src/skeletongraph/parser/edge_extractor.py
       │      [extract_edges()] → caller/callee relationships
       │      └─ Build dependency graph
       │
       ├─ Aggregate into IndexStore:
       │  │
       │  ├─ skeleton_table: {FQN → SkeletonCore}
       │  │  └─ Each has: docstring (primary), summary (fallback)
       │  │
       │  ├─ inverted_index: {keyword → set(FQNs)}
       │  │  └─ For entity matching
       │  │
       │  ├─ bm25_model: Lazy-loaded BM25 corpus
       │  │  └─ Built from docstrings + signatures + body keywords
       │  │     (Auto-invalidated on index changes)
       │  │
       │  ├─ embeddings: Optional EmbeddingStore
       │  │  └─ Incremental semantic search (if available)
       │  │
       │  ├─ dependency_graph: DependencyGraph
       │  │  └─ For blast-radius/dependency-chain traversal
       │  │
       │  └─ config: SGConfig
       │     └─ Flags loaded from .skeletongraph/config.json


┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: QUERY (CLI: sg query "search term")                       │
└─────────────────────────────────────────────────────────────────────┘

   src/skeletongraph/cli/main.py [query_command()]
   │
   └─→ src/skeletongraph/retrieval/resolver.py
       [resolve_context(prompt, store, enable_bm25_fallback=True)]
       │
       ├─ TIER 1: Entity Matching (Fast)
       │  │  
       │  ├─→ tokenize_query(prompt)
       │  │   └─ Lighter stop-word filtering (keeps domain terms)
       │  │
       │  ├─→ InvertedIndex.lookup(tokens)
       │  │   └─ Find FQNs directly matching tokens
       │  │   └─ match_source = "entity"
       │  │
       │  └─ Result: Found? → Return candidates
       │     Not found? → Fallback to Tier 2
       │
       ├─ TIER 2: BM25 Fallback (Semantic)
       │  │ (Only if no entity match)
       │  │
       │  ├─→ build_bm25_corpus() [if first time]
       │  │   └─ Corpus = {FQN → space-separated tokens}
       │  │     (Includes docstrings + signatures + body)
       │  │
       │  ├─→ BM25Model.search(prompt, top_k=5)
       │  │   └─ Rank by semantic relevance
       │  │   └─ match_source = "bm25"
       │  │
       │  └─ Result: Ranked candidates (no entity match)
       │
       ├─ TIER 3: SLM Fallback (Entity Resolution)
       │  │ (Optional, if enable_slm_fallback=True)
       │  │
       │  └─→ SLM extracts FQNs from prompt
       │      └─ Fuzzy match to skeleton table
       │      └─ match_source = "slm"
       │
       └─ RANKING: Multi-Signal Confidence
          │
          ├─→ src/skeletongraph/retrieval/confidence.py
          │   [score_candidates()]
          │   │
          │   ├─ entity_match: 1.0 (entity), 0.5 (bm25), 0.3 (slm)
          │   ├─ coverage: % of prompt terms in docstring
          │   ├─ ambiguity: How many matches (fewer = higher)
          │   ├─ dependency_depth: Cross-file reach
          │   └─ cross_file_signals: Imports/references
          │
          └─ Final Score: Weighted average → sorted candidates


┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: ASSEMBLY (Pack Context for LLM)                           │
└─────────────────────────────────────────────────────────────────────┘

   src/skeletongraph/assembly/prompt_builder.py
   [assemble_prompt(candidates, prompt, cfg, project_root)]
   │
   ├─ ZONE 1: Metadata (Static)
   │  ├─ .skeletongraph/project.md (goal, constraints, phase)
   │  ├─ .skeletongraph/architecture.md (key decisions)
   │  └─ Top-level imports, package structure
   │
   ├─ ZONE 2: High-Value Bodies (Dynamic)
   │  ├─ Top 3-5 functions ranked by confidence
   │  ├─ Full source code (or focused extraction if >200 lines)
   │  ├─ Summary from _pick_summary():
   │  │  ├─ Docstring (primary, docstring-first)
   │  │  ├─ Stored summary (fallback)
   │  │  └─ Empty string (if no docstring/summary)
   │  └─ Match source + confidence score
   │
   ├─ ZONE 3: Related Context (Graph Traversal)
   │  ├─ Callers/callees (blast_radius depth)
   │  ├─ Dependency chain (dep_depth)
   │  ├─ Test files (if load_tests=True)
   │  └─ Skeletons only (not full bodies)
   │
   ├─ ZONE 4: Fallback Context (If budget allows)
   │  └─ Keyword matches (inverted index)
   │
   └─ ZONE 5: Error Context (If previous run failed)
       └─ .skeletongraph/session/error_followup.json
          (Only error signal, no full context re-send)


┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: CLI OUTPUT                                                 │
└─────────────────────────────────────────────────────────────────────┘

   sg query "authenticate" --path .
   │
   ├─ Prints ranked context with:
   │  ├─ Function FQN + docstring
   │  ├─ Match source (entity/bm25/slm)
   │  ├─ Confidence score
   │  └─ Source code (if Tier 1)
   │
   └─ Outputs full assembled prompt
      (User copies into LLM manually)


═══════════════════════════════════════════════════════════════════════

## IDE FLOW (Interactive, Cursor/Claude Code)

┌─────────────────────────────────────────────────────────────────────┐
│ STARTUP: MCP Server Registration                                    │
└─────────────────────────────────────────────────────────────────────┘

   python -m skeletongraph server
   │
   └─→ src/skeletongraph/server/mcp.py [main()]
       │
       ├─ Start stdio MCP server
       ├─ Register tools with IDE:
       │  ├─ query_context(prompt: str) → str
       │  ├─ get_retrieval_context(prompt: str, detail_level: str) → dict
       │  └─ pack_context(prompt: str) → json
       │
       └─ Listening for tool calls from IDE


┌─────────────────────────────────────────────────────────────────────┐
│ IDE FIRST CHAT: Initialization                                     │
└─────────────────────────────────────────────────────────────────────┘

   User opens Cursor/Claude Code in project with .skeletongraph/
   │
   ├─ IDE loads .skeletongraph/config.json
   ├─ IDE reads .skeletongraph/project.md (if exists)
   │  └─ If MISSING: Agent should infer from codebase in first chat
   │
   ├─ IDE agent instructions loaded: cursor.instructions.md
   │  └─ Enforces: Query SG first, docstring-first, update metadata
   │
   └─ User types first prompt
      "Add user authentication feature"


┌─────────────────────────────────────────────────────────────────────┐
│ IDE AGENT WORKFLOW (Per Chat)                                      │
└─────────────────────────────────────────────────────────────────────┘

   Agent (Cursor/Claude) processes user prompt
   │
   ├─ STEP 1: Query SkeletonGraph
   │  │
   │  └─→ tool.call("query_context", {"prompt": "existing auth?"})
   │      │
   │      └─→ src/skeletongraph/server/mcp.py [query_context_handler()]
   │          │
   │          ├─ Load .skeletongraph/ if first time
   │          ├─ Run resolver pipeline (same as CLI Phase 3)
   │          ├─ Run assembler (same as CLI Phase 4)
   │          └─ Return assembled context to IDE
   │
   ├─ STEP 2: Understand Context
   │  ├─ Agent reads returned docstrings (docstring-first)
   │  ├─ Understands project constraints from project.md
   │  └─ Sees architecture patterns from architecture.md
   │
   ├─ STEP 3: Generate Code
   │  ├─ Agent writes new functions with clear docstrings
   │  ├─ Agent follows project constraints
   │  └─ Agent maintains architecture patterns
   │
   ├─ STEP 4: Update Metadata (If Needed)
   │  ├─ If discovering new constraint:
   │  │  └─ Agent edits .skeletongraph/project.md
   │  │
   │  ├─ If changing architecture:
   │  │  └─ Agent edits .skeletongraph/architecture.md
   │  │
   │  └─ If discovering new pattern:
   │     └─ Agent updates .skeletongraph/session/conversation_log.json
   │
   ├─ STEP 5: Report Completion
   │  │
   │  └─→ tool.call("report_completion", {
   │        "task": "Added JWT token validation",
   │        "files_modified": ["auth/jwt.py"],
   │        "metadata_updated": true
   │      })
   │      │
   │      └─→ src/skeletongraph/build.py [update_index()]
   │          ├─ Rebuild index from modified files
   │          ├─ Extract new docstrings
   │          ├─ Invalidate BM25 model (will rebuild on next query)
   │          ├─ Update embeddings incrementally
   │          └─ Write conversation log entry
   │
   └─ STEP 6: Next Chat Uses Fresh Index
      └─ Any new query sees updated docstrings + metadata


┌─────────────────────────────────────────────────────────────────────┐
│ IDE AGENT INSTRUCTIONS                                              │
└─────────────────────────────────────────────────────────────────────┘

   src/skeletongraph/cursor.instructions.md (etc for other IDEs)
   │
   ├─ MANDATORY RULES:
   │  ├─ Rule 1: Always use query_context() for codebase questions
   │  ├─ Rule 2: Write docstrings for every new function
   │  ├─ Rule 3: Update .skeletongraph/project.md if constraints change
   │  ├─ Rule 4: Call report_completion() after significant changes
   │  └─ Rule 5: Never disable BM25 fallback
   │
   ├─ DOCSTRING-FIRST WORKFLOW:
   │  ├─ def authenticate_user():  # Added docstring FIRST
   │  │   """Authenticate user against database."""
   │  │   ...
   │  │
   │  └─ If summary wrong: Update docstring (not SG config)
   │
   └─ METADATA AUTO-UPDATE:
       ├─ Agent discovers: "Must use async/await"
       ├─ Agent adds to project.md constraints
       ├─ Agent calls report_completion()
       └─ Next agent sees updated constraints


═══════════════════════════════════════════════════════════════════════

## KEY FILES & PURPOSES

┌─────────────────────────────────────────────────────────────────────┐
│ SKELETON GRAPH CORE (src/skeletongraph/)                            │
└─────────────────────────────────────────────────────────────────────┘

BUILD PIPELINE:
├─ build.py
│  ├─ discover_files(): Find all source files
│  ├─ build_index(): Parse + extract + aggregate
│  ├─ update_index(): Incremental rebuild after changes
│  └─ _summary_needs_refresh(): Check if docstring→summary stale
│
├─ parser/ast_extractor.py
│  └─ extract_file(): tree-sitter AST → FileSkeleton
│     (Extracts: docstrings, signatures, line ranges)
│
├─ parser/edge_extractor.py
│  └─ extract_edges(): caller/callee relationships
│
├─ parser/skeleton.py
│  └─ Data classes: SkeletonCore, FileSkeleton, etc
│
├─ graph/inverted_index.py
│  ├─ InvertedIndex: {keyword → set(FQNs)}
│  └─ build_bm25_corpus(): {FQN → space-separated tokens}
│     (For BM25 semantic search)
│
├─ graph/bm25.py
│  └─ BM25Model: Lightweight semantic search
│     (Lazy-loaded, invalidated on index change)
│
├─ graph/embeddings.py (Optional)
│  └─ EmbeddingStore: Semantic embeddings (if available)
│     (Incremental updates on index changes)
│
├─ storage/local.py
│  └─ IndexStore: In-memory index representation
│     ├─ skeleton_table, inverted_index, bm25_model, embeddings
│     └─ Persisted to .skeletongraph/domain/
│
├─ config.py
│  └─ SGConfig: Central configuration
│     ├─ enable_bm25_fallback: true
│     ├─ summary_use_docstrings: true
│     ├─ auto_rebuild_on_completion: true
│     └─ enable_embeddings: true


RETRIEVAL PIPELINE:
├─ retrieval/resolver.py
│  ├─ resolve_context(): Multi-tier retrieval
│  │  ├─ Tier 1: Entity matching (fast)
│  │  ├─ Tier 2: BM25 fallback (semantic)
│  │  ├─ Tier 3: SLM fallback (optional)
│  │  └─ Returns: Ranked candidates
│  │
│  └─ _bm25_fallback(): Semantic search when no entity match
│     └─ Auto-lazy-loads BM25 model on first use
│
├─ retrieval/confidence.py
│  └─ score_candidates(): 5-factor confidence scoring
│     ├─ entity_match (1.0 entity, 0.5 bm25, 0.3 slm)
│     ├─ coverage, ambiguity, dependency_depth, cross_file
│     └─ Returns: weighted score for ranking
│
├─ assembly/prompt_builder.py
│  ├─ assemble_prompt(): 5-zone context assembly
│  │  ├─ Zone 1: Metadata (.skeletongraph/project.md)
│  │  ├─ Zone 2: High-value bodies (top ranked functions)
│  │  ├─ Zone 3: Related context (graph traversal)
│  │  ├─ Zone 4: Fallback (keyword matches)
│  │  └─ Zone 5: Error context (if previous run failed)
│  │
│  └─ _pick_summary(): Docstring-first picker
│     ├─ Docstring (primary)
│     ├─ Stored summary (fallback)
│     └─ Empty string (both missing)


CLI ENTRY POINT:
├─ cli/main.py
│  ├─ build: Discover → parse → index
│  ├─ query: Resolve → assemble → output
│  ├─ server: Start MCP server
│  └─ init: Create project metadata (with --auto-infer)
│
├─ cli/init.py
│  ├─ run_init(): Setup project.md + architecture.md
│  ├─ _infer_metadata_with_llm(): Auto-infer from codebase
│  │  └─ Uses Claude Sonnet 4.6 by default
│  │
│  └─ _detect_phase(): Infer phase from git history


IDE/MCP INTERFACE:
├─ server/mcp.py
│  ├─ query_context_handler(): Tool for IDE queries
│  ├─ get_retrieval_context_handler(): Detailed context
│  └─ pack_context_handler(): Minimal packet assembly
│
└─ [IDE-specific files]:
   ├─ cursor.instructions.md: Cursor-specific rules
   ├─ claude-code.instructions.md: Claude Code rules
   ├─ copilot.instructions.md: GitHub Copilot rules
   ├─ codex.instructions.md: Codex rules
   └─ antigravity.instructions.md: Antigravity rules


═══════════════════════════════════════════════════════════════════════

## CONVERSATION LOGGING (IDE Session Tracking)

┌─────────────────────────────────────────────────────────────────────┐
│ .skeletongraph/session/conversation_log.json                        │
└─────────────────────────────────────────────────────────────────────┘

{
  "project": "skeletongraph",
  "start_time": "2026-05-13T10:30:00Z",
  "conversations": [
    {
      "id": "conv_1",
      "timestamp": "2026-05-13T10:30:05Z",
      "user_prompt": "Add user authentication",
      "agent_action": "Query codebase structure",
      "discovered_patterns": [
        "Uses async/await everywhere",
        "Docstring-first convention",
        "Constraint: No SQL database"
      ],
      "files_modified": ["auth/user.py", "auth/jwt.py"],
      "metadata_updated": {
        "project_md": {"added_constraint": "async/await mandatory"},
        "architecture_md": {"added_module": "Auth module with JWT"}
      },
      "summary": "Implemented user auth with JWT tokens",
      "reason": "User requested feature"
    },
    {
      "id": "conv_2",
      "timestamp": "2026-05-13T10:35:10Z",
      "user_prompt": "Where is password validation?",
      "context_retrieved": {
        "tool": "query_context",
        "query": "password validation",
        "results": ["verify_password_hash (docstring: Compare password with stored bcrypt hash)"]
      },
      "agent_action": "Referenced existing function",
      "summary": "Found existing password validation in auth/hash.py",
      "reason": "User asked for existing code"
    }
  ],
  "learned_constraints": [
    "async/await mandatory",
    "Docstring-first approach",
    "No external deps without approval"
  ],
  "learned_architecture": [
    "Auth module with JWT tokens",
    "Async request handlers",
    "Bcrypt password hashing"
  ]
}

Purpose:
├─ Prevent agent from re-learning same patterns
├─ Provide context for future chats in same session
├─ Track why decisions were made
├─ Detect if agent is looping (same query twice)
└─ Improve agent reasoning for next query
```

Now provide this prompt to Gemini Nano to generate architecture diagram:
