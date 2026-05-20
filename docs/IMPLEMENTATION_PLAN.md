# Implementation Plan — Post-Foundation Work

**For**: a Sonnet-tier model executing after the Opus foundation pass.

**Prerequisite reading**: `docs/RESEARCH_PLAN.md` (the thesis, the metric
catalog, and the anti-drift checklist). Every task here exists to support that
plan; if a task does not, skip it.

**Repo conventions** (from MEMORY):
- Work in main repo dir, never `.claude/worktrees/*`
- Never auto-commit; provide commands for the user to run manually
- Plan-first for any change touching >2 files
- All Windows-safe (no POSIX-only assumptions)

---

## Task index

| # | Task | Files | Est. effort |
|---|---|---|---|
| 1 | Wire Aider repo-map arm | `eval/backends/aider_map.py`, `eval/agent/tools.py` | 2-3h |
| 2 | Wire Hybrid-RAG arm | `eval/backends/hybrid.py`, `eval/agent/tools.py` | 3-4h |
| 3 | SG ablation toggles | `src/skeletongraph/config.py`, `src/skeletongraph/retrieval/resolver.py`, `src/skeletongraph/retrieval/ranker.py`, `eval/agent/tools.py` | 4-6h |
| 4 | Multi-language tree-sitter adapters | `src/skeletongraph/parser/*`, `pyproject.toml` | 1-2 days |
| 5 | ContextBench block annotations | `eval/scripts/extract_tasks.py`, `eval/agent/run_agent.py` | 4-6h |
| 6 | Repo cleanup (delete + gitignore) | root | 30min |

Do them in this order. 1, 2, 3 are gating for tier-0 sanity. 4 is for tool
viability. 5 is gating for tier-5 ContextBench run. 6 is hygiene.

---

## Task 1 — Aider repo-map arm

**Goal**: a working `aider` arm in `tools._retrieve` returning ranked file
paths exactly like the `bm25` arm does, but using Aider's published repo-map.

**Approach**: depend on the published `aider-chat` PyPI package — `pip install
aider-chat`. Use their `RepoMap` class directly so the comparison is faithful
to the deployed system. Document the commit hash in the paper.

### File: `eval/backends/aider_map.py` (NEW)

```python
"""Aider repo-map baseline.

Uses the published aider-chat package's RepoMap class. We give it the same
query + repo and extract the ranked file list it would have surfaced.

This is the canonical repo-map baseline. Document the aider-chat version
in the paper's reproducibility appendix.
"""
from __future__ import annotations
from pathlib import Path
from typing import List


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked file paths (forward-slash) for the query.

    Aider's RepoMap ranks files by PageRank over the tree-sitter symbol graph
    given a set of "mentioned" identifiers. We feed it tokenized query terms.
    """
    try:
        from aider.repomap import RepoMap
    except ImportError as e:
        raise RuntimeError(
            "aider-chat not installed. Run: pip install aider-chat"
        ) from e

    rm = RepoMap(
        map_tokens=4096,           # token budget for the repo-map text
        root=str(repo),
        main_model=None,           # we don't call a model here — just rank
        io=_NullIO(),              # avoid stdout chatter
        verbose=False,
    )

    # Collect all source files Aider would consider.
    all_files = [str(p) for p in repo.rglob("*")
                 if p.is_file() and _is_source(p)]
    chat_files: List[str] = []     # files already "open" — none in our case
    mentioned_idents = set(_tokenize_query(query))

    ranked_text = rm.get_repo_map(
        chat_files=chat_files,
        other_files=all_files,
        mentioned_fnames=set(),
        mentioned_idents=mentioned_idents,
    )
    # ranked_text contains file headers + symbol snippets. Extract files in
    # order of first appearance and return.
    return _extract_files_in_order(ranked_text, repo)[:k]


def _tokenize_query(q: str) -> List[str]:
    import re
    return [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", q) if len(t) >= 3]


def _is_source(p: Path) -> bool:
    return p.suffix in {".py", ".js", ".ts", ".go", ".java", ".rs",
                         ".rb", ".cpp", ".c", ".cs", ".kt", ".php"}


def _extract_files_in_order(map_text: str, repo: Path) -> List[str]:
    """RepoMap output has file paths as section headers. Parse them out."""
    import re
    files: List[str] = []
    seen = set()
    for line in (map_text or "").splitlines():
        # Aider's format: "path/to/file.py:" or full path lines
        m = re.match(r"^([\w./\\-]+\.[A-Za-z]+):?\s*$", line.strip())
        if not m:
            continue
        rel = m.group(1).replace("\\", "/")
        # Normalize: drop any "repo/" prefix; keep relative path
        try:
            abs_p = (repo / rel).resolve()
            if abs_p.is_file():
                rel = str(abs_p.relative_to(repo)).replace("\\", "/")
        except (OSError, ValueError):
            pass
        if rel not in seen:
            seen.add(rel)
            files.append(rel)
    return files


class _NullIO:
    """Aider's RepoMap takes an IO object; we suppress its output."""
    def tool_output(self, *a, **k): pass
    def tool_warning(self, *a, **k): pass
    def tool_error(self, *a, **k): pass
    def read_text(self, fname):
        from pathlib import Path
        return Path(fname).read_text(encoding="utf-8", errors="replace")
```

### File: `eval/agent/tools.py` — replace the placeholder

The existing `_retrieve` already has the dispatch — change `if backend ==
"aider":` from `noqa` to a real import + call, mirroring the `bm25` branch.

### Acceptance criteria
- `pip install aider-chat` is added to `pyproject.toml` `[project.optional-dependencies] eval`
- Running `python -m eval.agent.run_agent --task-id astropy__astropy-8707 --arm aider --model qwen-7b` produces a non-empty `first_search_hits`
- `embeddings_used` is `None` (not SG; correct)
- Verify against pinned aider-chat version; record it in `eval/REPRODUCIBILITY.md` (new file with version pins for all baselines)

---

## Task 2 — Hybrid-RAG arm

**Goal**: a strong industry-default baseline. BM25 + sentence-transformers
embeddings + cross-encoder rerank. Same `retrieve(query, repo, k)` API as
the other backends.

**Stack**:
- BM25: existing `eval/backends/bm25_flat.py` (already there)
- Embeddings: `sentence-transformers` `all-MiniLM-L6-v2` (same model SG uses, for fair comparison)
- Rerank: `sentence-transformers` `cross-encoder/ms-marco-MiniLM-L-6-v2`

### File: `eval/backends/hybrid.py` (NEW)

```python
"""Hybrid-RAG baseline: BM25 ∪ dense → cross-encoder rerank.

The deployed industry default (Augment, Voyage, Cohere). Same retrieval recipe
that SG's curator chooses among; here we run it monolithically without query
classification — that's the controlled experiment.

Per-repo index is built lazily on first call and cached on disk under
.hybrid_index/ for subsequent searches in the same workspace.
"""
from __future__ import annotations
from pathlib import Path
from typing import List


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Return up to k ranked file paths."""
    idx = _ensure_index(repo)
    # 1. BM25 candidates (top 30)
    bm25_hits = idx.bm25_search(query, top_k=30)
    # 2. Dense candidates (top 30)
    dense_hits = idx.dense_search(query, top_k=30)
    # 3. Union → cross-encoder rerank → top k
    candidates = list({*bm25_hits, *dense_hits})
    if not candidates:
        return []
    reranked = idx.rerank(query, candidates, top_k=k)
    return reranked


def _ensure_index(repo: Path):
    """Lazy per-repo index. Cache under <repo>/.hybrid_index/."""
    # Implementation: build chunked file index, embed with MiniLM, save .npz.
    # Reuse pattern from src/skeletongraph/graph/embeddings.py (and its bug
    # fix: temp file must end in .npz).
    # Each "candidate" is a FILE path (chunked content concatenated for the
    # cross-encoder input).
    ...
```

Full implementation: see existing `src/skeletongraph/graph/embeddings.py` for
the embed-and-save pattern. Cross-encoder is a 1-line `CrossEncoder.predict`.

### Acceptance criteria
- First call on a fresh repo builds `.hybrid_index/` in ~30s; subsequent calls <1s
- Embeddings save with `.npz`-terminated temp (don't repeat the SG bug)
- Returns 5-10 files for typical queries
- Same dependency entry pattern in `pyproject.toml`

---

## Task 3 — SG ablation toggles (the paper's intellectual core)

**Goal**: three SG configurations with one component disabled each, callable
via the `sg-nograph`, `sg-norerank`, `sg-nosummary` arms.

**Approach**: add three boolean flags to `SGConfig`, wire them through the
relevant code paths, and have `tools._retrieve` instantiate `SGEngine` with
the right flag set.

### File: `src/skeletongraph/config.py` — add to SGConfig

```python
# ── Ablation toggles (eval only) ──────────────────────────────────────
# These default to True (full SG); the eval harness flips them off one at
# a time for the stage-2 ablation. NEVER ship them False in production.
enable_graph_expansion: bool = True   # Tier-2/Tier-3 graph neighbor inclusion
enable_centrality_rerank: bool = True # PageRank/hub-score reranking
enable_summaries: bool = True         # Use stored summaries in retrieval response
```

Also add to `save_config()`'s data dict so they round-trip.

### File: `src/skeletongraph/retrieval/resolver.py`

In `resolve_context()`, gate the graph-expansion blocks on
`config.enable_graph_expansion`. When False:
- Skip `blast_radius()` and `dependency_chain()` calls
- Return only the direct entity matches (Tier 1 only)

Search the file for `blast_radius` and `dependency_chain` — wrap each call site
in `if config.enable_graph_expansion:`.

### File: `src/skeletongraph/retrieval/ranker.py`

In `Ranker.score()`, gate the connectivity (hub) signal:
```python
# Signal 2: Connectivity (hub score)
if weights.connectivity > 0 and self._centrality_enabled:
    hub = self._hub_scores.get(fqn, 0.0)
    score += hub * self.weights.connectivity
```

Add `centrality_enabled: bool = True` to `Ranker.__init__`, pass through from
`SGEngine` (which reads `config.enable_centrality_rerank`).

### File: SG search response builder (in mcp.py or wherever summaries get attached)

When `config.enable_summaries` is False, omit the summary text from the
returned context — return only the FQN + signature + line range, no Tier-2
summary, no Tier-3 docstring.

### File: `eval/agent/tools.py` — wire the ablation backends

Replace the `NotImplementedError` block:

```python
if backend in ("sg-nograph", "sg-norerank", "sg-nosummary"):
    from skeletongraph.engine import SGEngine
    from skeletongraph.config import SGConfig
    cfg = SGConfig()
    if backend == "sg-nograph":
        cfg.enable_graph_expansion = False
    elif backend == "sg-norerank":
        cfg.enable_centrality_rerank = False
    elif backend == "sg-nosummary":
        cfg.enable_summaries = False
    engine = SGEngine(project_root=repo, config=cfg)
    res = engine.heuristic_query(query, top_n=k)
    return [c.skeleton.fqn for c in res.candidates]
```

Check that `SGEngine.__init__` accepts a `config=` kwarg; if not, add it
(merge with the project-local config).

### Acceptance criteria
- All three ablation backends run end-to-end on at least one task without raising
- Behavior visibly differs from full SG: sg-nograph has fewer total candidates;
  sg-norerank has different rank order; sg-nosummary returns no summary fields
- A unit test in `tests/test_ablations.py` asserts each flag changes the
  ResolverResult by at least one observable field

---

## Task 4 — Multi-language tree-sitter adapters

**Goal**: parser support for 10 languages so SG is publishable as a tool, not
just a Python-only research artifact. Per RESEARCH_PLAN §9: paper stays
Python-only; tool ships v0.2 with 10 languages.

**Target languages** (top-10 by real-world agent usage):
1. Python (already supported)
2. TypeScript
3. JavaScript
4. Go
5. Java
6. Rust
7. C++
8. C
9. Ruby
10. PHP
11. (bonus) Kotlin or C#

**Approach**: tree-sitter has pip-installable grammars for all of these. The
parser needs a language-aware adapter that extracts: functions/methods, classes,
signatures, docstrings, and an FQN convention.

### Files

- `src/skeletongraph/parser/languages/__init__.py` — registry
- `src/skeletongraph/parser/languages/python.py` — refactor existing parser
- `src/skeletongraph/parser/languages/typescript.py` — new
- `src/skeletongraph/parser/languages/javascript.py` — new
- (etc. for each language)
- `src/skeletongraph/parser/dispatcher.py` — file extension → language adapter

### Adapter interface

```python
class LanguageAdapter(Protocol):
    extensions: Set[str]                # {".ts", ".tsx"} etc
    tree_sitter_lang: str               # "typescript"

    def parse_file(self, source: bytes, path: str) -> List[SkeletonCore]: ...
    def fqn_for(self, file_path: str, qualifier: List[str], name: str) -> str: ...
    def is_exported(self, node) -> bool: ...
```

The existing `SkeletonCore` dataclass already has fields that map cleanly to
any language (fqn, signature, file_path, line range, docstring, complexity).

### Dependencies

Add to pyproject.toml:
```toml
tree-sitter-python = "^0.20"
tree-sitter-typescript = "^0.20"
tree-sitter-javascript = "^0.20"
tree-sitter-go = "^0.20"
tree-sitter-java = "^0.20"
tree-sitter-rust = "^0.20"
tree-sitter-cpp = "^0.20"
tree-sitter-c = "^0.20"
tree-sitter-ruby = "^0.20"
tree-sitter-php = "^0.20"
```

### Acceptance criteria
- `sg build` on a small TypeScript fixture (e.g. a tiny Express app) produces
  a non-empty index with functions, classes, and FQNs
- Same for each of the 10 languages — 10 small fixtures live under
  `tests/fixtures/multilang/<lang>/`
- Existing Python tests still pass
- README updated with the supported-languages table

### What NOT to do
- Don't try to match language idioms perfectly (e.g. JS arrow functions vs
  function declarations — extract both as functions; don't fight the AST)
- Don't try to resolve cross-file types — SG's contribution isn't type
  resolution
- Don't add summarization for non-Python languages in v0.2 — docstring
  extraction yes, LLM summaries no (cost)

---

## Task 5 — ContextBench block annotations

**Goal**: backfill block- and line-level gold annotations into our task
dataset so `run_agent.py` can compute the ContextBench usage-drop metric
properly (not just at file granularity).

**Source**: ContextBench's released gold-context annotations cover an overlap
subset with SWE-bench Verified. Download their annotations JSON, join on
`instance_id`, augment `eval/datasets/stage0.jsonl`.

### Files

- `eval/scripts/extract_contextbench.py` (NEW) — download + join script
- `eval/agent/run_agent.py` — extend `_consolidation_metrics` to use block-
  level annotations when present (fall back to file-level when not)

### Schema additions to each task record in stage0.jsonl

```json
{
  "task_id": "astropy__astropy-8707",
  "gold_files": [...],
  // NEW:
  "gold_blocks": [
    {"path": "astropy/io/fits/header.py",
     "ast_kind": "method",
     "qualified_name": "Header.fromstring",
     "start_line": 482, "end_line": 530}
  ],
  "gold_lines": {  // line-precise gold byte ranges
    "astropy/io/fits/header.py": [[482, 530], [612, 615]]
  }
}
```

### Acceptance criteria
- For every task with a ContextBench annotation: `gold_blocks` and `gold_lines`
  populated
- `_consolidation_metrics()` reports `consolidation_gap_blocks` and
  `consolidation_gap_lines` when available, in addition to `gap_files`
- A test verifies the join logic on at least 3 known-overlap tasks

---

## Task 6 — Repo cleanup

Files visible in the repo screenshot that DON'T belong in a public/professional
research repo:

| File/dir | Why | Action |
|---|---|---|
| `.cursor/` | IDE-local config (Cursor) | `git rm -r`, add to .gitignore |
| `AGENT_RULES.md` | Cursor/Claude agent rules — not project docs | `git rm`, move to `.cursor/` (local-only) |
| `CLAUDE.md` | Claude Code project memory | `git rm`, move to `.claude/` (local-only) |
| `CURSOR_SETUP.md` | Cursor IDE setup notes | `git rm` if no public value, else `docs/internal/` |
| `GEMINI_DIAGRAM_PROMPT.md` | Test prompt for diagram generation | `git rm` — not project docs |
| `test_docstring_extraction.py` (root) | Test file at root | `git mv` to `tests/` |
| `test_docstring_first.py` (root) | Test file at root | `git mv` to `tests/` |
| `mcp.json` | MCP config example | keep, but rename to `mcp.example.json` |

### .gitignore additions

```gitignore
# IDE-local files (machine-specific)
.cursor/
.claude/
.idea/
.vscode/

# Project-local agent guidance (not for the public repo)
AGENT_RULES.md
CLAUDE.md
CURSOR_SETUP.md
GEMINI_DIAGRAM_PROMPT.md

# Eval intermediates
eval/datasets/_agent_work/
eval/results/agent/*.json
!eval/results/agent/SUMMARY.md
```

### Git commands to clean

```cmd
git rm -r --cached .cursor
git rm --cached AGENT_RULES.md CLAUDE.md CURSOR_SETUP.md GEMINI_DIAGRAM_PROMPT.md
git mv test_docstring_extraction.py tests/test_docstring_extraction.py
git mv test_docstring_first.py tests/test_docstring_first.py
git mv mcp.json mcp.example.json
git add .gitignore
git commit -m "chore: clean repo of IDE-local and agent-specific files"
```

(The user runs commits; never auto-commit per MEMORY.)

---

## Execution order for Sonnet's session

1. Task 6 (cleanup) — clears noise before deeper work
2. Task 3 (SG ablations) — paper's intellectual core; needs to work for tier-2
3. Task 1 (Aider arm) — quick win, gets the baseline ready
4. Task 2 (Hybrid-RAG arm) — strongest baseline
5. After 1-3 land: re-run 6-arm local 7B smoke on 30 tasks, validate ordering
6. Task 4 (multi-language) — biggest lift, can run in parallel with paper writing
7. Task 5 (ContextBench annotations) — gating for tier-5

---

## Open questions for the user (only if blockers — defer otherwise)

- Aider arm: pin to a specific `aider-chat` version, or HEAD? (Default: pin to current stable.)
- Hybrid-RAG: use `sentence-transformers` cross-encoder, or a faster alternative? (Default: SBERT cross-encoder.)
- Multi-language: include Kotlin OR C#? (Default: C#, more relevant to SWE-bench-style benchmarks.)
- ContextBench annotations: which subset to backfill — Lite (500) or Full (1136)? (Default: Lite, fits in tier-5 budget.)
