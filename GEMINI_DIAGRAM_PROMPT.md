# Gemini Nano/Pro Prompt: Generate SkeletonGraph Architecture Diagram

Use this prompt with Gemini Nano/Pro to generate SVG/Mermaid architecture diagram:

```
You are an expert software architect visualizing a code retrieval pipeline.

## Task
Generate a Mermaid diagram showing SkeletonGraph's complete pipeline architecture.

## Requirements
1. Show TWO parallel flows: CLI (left) and IDE (right)
2. For each flow, show:
   - Data flow direction (arrows)
   - Which files handle each phase
   - When BM25/docstrings are used
   - Where metadata is updated
   - Where LLM inference happens

3. CLI Flow:
   - Entry: python -m skeletongraph build/query
   - Phase 1: INIT (cli/init.py) → Auto-infer with LLM
   - Phase 2: DISCOVER & PARSE (build.py, ast_extractor.py) → Extract docstrings
   - Phase 3: QUERY (resolver.py) → Entity match → BM25 fallback
   - Phase 4: ASSEMBLY (prompt_builder.py) → Docstring-first summaries
   - Phase 5: OUTPUT → Ranked context to user

4. IDE Flow:
   - Entry: MCP Server (server/mcp.py) listening
   - Chat 1: IDE agent (cursor.instructions.md) → Query SkeletonGraph
   - Resolver: Same as CLI but returns to IDE
   - Agent learns: Updates project.md, architecture.md
   - Agent completes: Calls report_completion() → update_index()
   - Chat N: Index rebuilt, new docstrings extracted, cycle repeats

5. Data Storage:
   - Show .skeletongraph/ directory structure:
     ├─ config.json (with flags)
     ├─ project.md (goal, constraints, phase)
     ├─ architecture.md (decisions)
     ├─ domain/ (skeleton index)
     └─ session/ (conversation_log.json)

6. Key Components:
   - Docstring Extraction (parser/ast_extractor.py)
   - Inverted Index (graph/inverted_index.py)
   - BM25 Fallback (graph/bm25.py, lazy-loaded)
   - Confidence Scoring (retrieval/confidence.py)
   - Prompt Assembly (assembly/prompt_builder.py)
   - Metadata Updates (build.py update_index)

7. Color coding:
   - Blue: Data storage
   - Green: CLI operations
   - Orange: IDE operations
   - Red: LLM inference points
   - Purple: Index rebuild triggers

## Output Format
Generate a Mermaid diagram (can be embedded in markdown as ````mermaid ... ````)

## Example Structure (just to show style):
```
graph LR
    subgraph CLI["CLI Flow"]
        A["python -m sg build<br/>--auto-infer"]
        B["cli/init.py<br/>LLM: Infer metadata"]
        C["build.py<br/>discover_files()"]
    end
    
    subgraph Shared["Shared Core"]
        ...
    end
    
    subgraph IDE["IDE Flow (Cursor)"]
        ...
    end
    
    A -->|First build| B
    B -->|Creates| ConfigDB["config.json<br/>project.md<br/>architecture.md"]
    ...
```

Generate the full diagram now, making it comprehensive but readable.
```

---

## Implementation: Conversation Logging

Now add conversation logging to track IDE agent discoveries:
