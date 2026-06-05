"""Per-arm tool surfaces + system prompts (systems comparison, final v2).

The OLD harness gave every arm the identical 5 tools and varied only the
search_code backend — a controlled retrieval ablation that nerfs the pipelines
(SG/cbmem/aider) down to a bare ranker. For final v2 we compare SYSTEMS: each
pipeline exposes the tools that ARE its contribution, holding only the model,
tasks, and edit/submit/verify substrate common.

  baseline (bm25/grep/hybrid/none/…) — search_code/list_files + read_file + edit/submit
  sg     — + read_symbol (fetch one function) + expand (callers/callees)
  cbmem  — its graph tools (search_graph/trace/snippet/architecture)   [phase 3]
  aider  — repo-map INJECTED into the prompt, no search tool           [phase 4]

build_profile(arm, repo) -> (tool_schemas, system_prompt).
Anything not yet natively wired falls back to the baseline profile, so existing
arms are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make eval/backends importable (aider repo-map injection) regardless of launch.
_EVAL_DIR = str(Path(__file__).resolve().parent.parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from .tools import TOOL_SCHEMAS, READ_SYMBOL_SCHEMA, EXPAND_SCHEMA

# Baseline schemas keyed by name so profiles can compose subsets.
_BY_NAME = {s["function"]["name"]: s for s in TOOL_SCHEMAS}


BASELINE_PROMPT = """You are an autonomous software engineer fixing a bug in a \
repository. You can only see the code through your tools.

Available tools: search_code, list_files, read_file, edit_file, submit.

Process:
1. Use search_code to locate the code relevant to the issue.
2. read_file to inspect the relevant code.
3. edit_file to make the minimal correct fix.
4. Call submit when done.

Rules:
- Make the smallest change that correctly fixes the issue.
- Do NOT run or write tests — the test environment is not available.
- Do NOT explain at length; act through tools.
- When the fix is complete, call submit.
"""

# SG's NATIVE pipeline: structural search, then fetch the function (read_symbol)
# or follow the call graph (expand) — not whole-file reads.
SG_PROMPT = """You are an autonomous software engineer fixing a bug in a \
repository. You can only see the code through your tools.

Available tools: search_code, read_symbol, expand, read_file, edit_file, submit.

Process:
1. search_code(query) — locate relevant functions/classes (ranked file::symbol).
2. read_symbol(fqn) — read ONE function/class body from a result (cheap). Prefer
   this over read_file; only read_file when you need the whole file.
3. expand(fqn) — see a function's callers and callees to follow the real flow.
4. edit_file to make the minimal correct fix; submit when done.

Rules:
- Make the smallest change that correctly fixes the issue.
- Do NOT run or write tests — the test environment is not available.
- Do NOT explain at length; act through tools.
- When the fix is complete, call submit.
"""

# SG native tool surface = baseline 5 + read_symbol + expand.
_SG_SCHEMAS = [
    _BY_NAME["search_code"], READ_SYMBOL_SCHEMA, EXPAND_SCHEMA,
    _BY_NAME["list_files"], _BY_NAME["read_file"],
    _BY_NAME["edit_file"], _BY_NAME["submit"],
]


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


# ── cbmem native tools (the real MCP graph surface, via CLI) ─────────────────
_CBMEM_SCHEMAS = [
    _fn("cbmem_search",
        "Search the code knowledge graph for relevant functions/classes. Returns "
        "ranked file::symbol plus a qualified_name to pass to cbmem_snippet.",
        {"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
    _fn("cbmem_trace",
        "Trace a function's call path: its callers and callees (BFS up to depth 5).",
        {"function_name": {"type": "string"},
         "direction": {"type": "string", "description": "in | out | both"}},
        ["function_name"]),
    _fn("cbmem_snippet",
        "Fetch the source code of one function by its qualified_name (from a search "
        "result) — cheaper than reading the whole file.",
        {"qualified_name": {"type": "string"}}, ["qualified_name"]),
    _fn("cbmem_arch",
        "Codebase overview: languages, packages, entry points, hotspots, clusters.",
        {}, []),
    _BY_NAME["read_file"], _BY_NAME["edit_file"], _BY_NAME["submit"],
]

CBMEM_PROMPT = """You are an autonomous software engineer fixing a bug in a \
repository. You explore the code through a knowledge-graph memory.

Available tools: cbmem_search, cbmem_trace, cbmem_snippet, cbmem_arch, read_file, edit_file, submit.

Process:
1. cbmem_search(query) — find relevant functions/classes (ranked file::symbol).
2. cbmem_snippet(qualified_name) — read a specific function (cheap); cbmem_trace to
   follow callers/callees; cbmem_arch for a high-level map. read_file for a whole file.
3. edit_file to make the minimal correct fix; submit when done.

Rules: smallest correct change; do NOT run or write tests; act through tools; submit when done.
"""

# ── aider native: PageRank repo-map INJECTED into the prompt, no search tool ──
_AIDER_SCHEMAS = [
    _BY_NAME["list_files"], _BY_NAME["read_file"],
    _BY_NAME["edit_file"], _BY_NAME["submit"],
]

_AIDER_PROMPT_HEAD = """You are an autonomous software engineer fixing a bug in a \
repository. Below is a repository map (the most important symbols and signatures,
ranked by importance) to orient you. Use read_file to open the files you need.

Available tools: list_files, read_file, edit_file, submit.

Process: use the repository map to decide which files to read, read_file them,
edit_file the minimal correct fix, then submit. Do NOT run or write tests.
"""


def build_profile(arm: str, repo: Path):
    """Return (tool_schemas, system_prompt) for this arm. Each pipeline uses its
    NATIVE tools; baselines (bm25/grep/hybrid/none) keep the standard 5."""
    if arm == "sg":
        return (_SG_SCHEMAS, SG_PROMPT)
    if arm == "cbmem":
        return (_CBMEM_SCHEMAS, CBMEM_PROMPT)
    if arm == "aider":
        try:
            from backends.aider_repomap import get_map_text
            mp = get_map_text(Path(repo))
        except Exception:
            mp = ""
        prompt = _AIDER_PROMPT_HEAD + (
            "\n\n--- REPOSITORY MAP ---\n" + mp + "\n--- END MAP ---\n"
            if mp else "\n\n(repository map unavailable — use list_files/read_file)\n")
        return (_AIDER_SCHEMAS, prompt)
    return (TOOL_SCHEMAS, BASELINE_PROMPT)


# Superset of every tool name any profile can expose — used by react.py to
# validate text-form tool calls regardless of which profile is active.
ALL_TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS} | {
    "read_symbol", "expand",
    "cbmem_search", "cbmem_trace", "cbmem_snippet", "cbmem_arch"}
