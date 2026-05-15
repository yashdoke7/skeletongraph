"""
SLM Entity Extractor — the brain of the v4 pipeline.

The SLM (Small Language Model) interprets user prompts by reading pre-processed
graph metadata (function summaries, file map, session context). It does NOT
traverse the graph or read files — code handles that after.

Pipeline position:
  User prompt → [SLM Extractor] → structured retrieval plan → graph expansion → assembly

Cost: ~$0.0001-$0.0005 per call (Gemini Flash / Haiku).
Latency: ~200ms.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Data Types ──────────────────────────────────────────────────────────


@dataclass
class SLMEntity:
    """An entity identified by the SLM."""
    fqn: str                    # e.g., "models.py::PreparedRequest.prepare_content_length"
    role: str = "target"        # "target" | "related" | "context"


@dataclass
class SLMResult:
    """Structured output from the SLM entity extraction."""
    mode: str = ""                              # e.g., "debug_investigate"
    entities: List[SLMEntity] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)  # Domain terms, not code names
    files: List[str] = field(default_factory=list)
    reasoning: str = ""                         # SLM's explanation (included in main LLM prompt)
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class SLMToolCall:
    """A planned tool call for context expansion."""
    call_type: str                  # "function" | "file" | "range" | "directory"
    target: str                     # FQN or path
    start_line: int = 0
    end_line: int = 0
    include_neighbors: bool = False


@dataclass
class SLMPlan:
    """Structured plan of tool calls from the SLM."""
    tool_calls: List[SLMToolCall] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


# ── System Prompt ───────────────────────────────────────────────────────


_EXTRACT_SYSTEM = """You are a code retrieval planner. Given a user's coding request and a project's function index, identify which code entities are relevant.

Return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "mode": "<one of: retrieval_fast, debug_targeted, debug_investigate, build_guided, build_greenfield, refactor, explain, architecture, review, test, document, migrate>",
  "entities": [
    {"fqn": "file/path.py::ClassName.method_name", "role": "target|related|context"}
  ],
  "concepts": ["concept1", "concept2"],
  "files": ["file1.py", "file2.py"],
  "reasoning": "One sentence explaining your selection"
}

Rules:
- "target" = the function the user wants to change/fix/understand (max 3)
- "related" = functions closely connected to the target (max 5)
- "context" = functions that provide background (max 5)
- Match natural language to function SUMMARIES, not just names
- "concepts" = domain terms from the prompt that aren't code names
- If unsure, include MORE candidates rather than fewer
- If the user references recent work ("fix it", "that thing"), use session context to resolve"""


_PLAN_SYSTEM = """You are a tool planner for context expansion.

Given a user request and a function index, propose a small list of tool calls
that will gather the most important context for a coding task.

Return ONLY valid JSON (no markdown):
{
    "reasoning": "one short sentence",
    "tool_calls": [
        {"type": "function", "target": "path/file.py::Class.method", "include_neighbors": false},
        {"type": "file", "target": "path/file.py"},
        {"type": "range", "target": "path/file.py", "start_line": 10, "end_line": 60}
    ]
}

Rules:
- Max 5 tool_calls total
- Only use targets that appear in the function index or file list
- Prefer function-level expansions over whole files
- If unsure, return an empty tool_calls list
"""


# ── SLM Prompt Builder ──────────────────────────────────────────────────


def build_slm_prompt(
    user_prompt: str,
    project_summary: str,
    file_map: str,
    function_index: str,
    session_context: str = "",
    retry_note: str = "",
) -> str:
    """Build the user-message portion of the SLM prompt.

    Each segment is sized to keep total input under ~6000 tokens.
    """
    parts = []

    # Segment 1: User request (always first, highest attention)
    parts.append(f"## User's Request\n{user_prompt}")

    # Segment 2: Session context (anaphora resolution)
    if session_context:
        parts.append(f"## Recent Session\n{session_context}")

    # Segment 3: Project context
    if project_summary:
        parts.append(f"## Project\n{project_summary}")

    # Segment 4: File structure
    if file_map:
        parts.append(f"## Files\n{file_map}")

    # Segment 5: Function index (the main knowledge base)
    parts.append(f"## Function Index\n{function_index}")

    # Retry note (if retrying after failed attempt)
    if retry_note:
        parts.append(f"## Retry Note\n{retry_note}")

    return "\n\n".join(parts)


def build_function_index(
    skeleton_table: Dict,
    summaries: Dict[str, str],
    pagerank_scores: Optional[Dict[str, float]] = None,
    max_entries: int = 500,
    prefilter_fqns: Optional[Set[str]] = None,
) -> str:
    """Build the function index segment for the SLM prompt.

    For small projects (<500 functions): include all.
    For larger projects: use prefilter_fqns (from PageRank + session/file hints).

    Format: "file.py::Class.method — one-line summary"
    """
    entries = []

    # Determine which FQNs to include
    if prefilter_fqns is not None:
        fqns = list(prefilter_fqns)
    else:
        fqns = list(skeleton_table.keys())

    # Sort by PageRank (most important first) if available
    if pagerank_scores:
        fqns.sort(key=lambda f: pagerank_scores.get(f, 0.0), reverse=True)

    for fqn in fqns[:max_entries]:
        sk = skeleton_table.get(fqn)
        if sk is None:
            continue
        summary = ""
        if hasattr(sk, "docstring") and sk.docstring:
            summary = sk.docstring.strip().splitlines()[0]
        if not summary:
            summary = summaries.get(fqn, "")
        short_name = fqn.split("::")[-1] if "::" in fqn else fqn
        file_display = sk.file_path if hasattr(sk, "file_path") else ""

        if summary:
            entries.append(f"{file_display}::{short_name} — {summary}")
        else:
            # No summary — show signature instead
            sig = sk.signature if hasattr(sk, "signature") else short_name
            entries.append(f"{file_display}::{short_name} — {sig}")

    return "\n".join(entries)


def build_file_map(file_skeletons: Dict, file_summaries: Optional[Dict[str, str]] = None) -> str:
    """Build the file structure segment.

    Format: "path/file.py — description (N functions)"
    """
    lines = []
    for file_path, sks in sorted(file_skeletons.items()):
        n_funcs = len(sks) if isinstance(sks, list) else len(getattr(sks, "all_skeletons", []))
        desc = ""
        if file_summaries and file_path in file_summaries:
            desc = f" — {file_summaries[file_path]}"
        lines.append(f"{file_path}{desc} ({n_funcs} functions)")
    return "\n".join(lines)


def build_project_summary(sg_dir) -> str:
    """Load project.md and return first ~300 tokens."""
    project_md = sg_dir / "project.md"
    if project_md.exists():
        text = project_md.read_text(encoding="utf-8")
        # Cap at ~300 tokens (~1200 chars)
        return text[:1200]
    return ""


def build_session_context(sg_dir) -> str:
    """Load current.md for anaphora resolution."""
    current_md = sg_dir / "session" / "current.md"
    if current_md.exists():
        text = current_md.read_text(encoding="utf-8")
        # Cap at ~150 tokens (~600 chars)
        return text[:600]
    return ""


def slm_plan_tools(
    prompt: str,
    store,
    sg_dir,
    config,
    session_fqns: Optional[Set[str]] = None,
) -> SLMPlan:
    """Plan a small list of expansion tool calls using an SLM."""
    try:
        from ..llm.provider import complete, LLMConfig
    except ImportError:
        return SLMPlan(success=False, error="litellm not installed")

    start = time.time()

    prefilter = None
    if len(store.skeleton_table) > 500:
        prefilter = prefilter_for_slm(prompt, store, session_fqns=session_fqns)

    project_summary = build_project_summary(sg_dir)
    file_map = build_file_map(store.file_skeletons)
    function_index = build_function_index(
        store.skeleton_table,
        store.summaries._store if hasattr(store, "summaries") else {},
        pagerank_scores=getattr(store, "pagerank_scores", None),
        max_entries=400,
        prefilter_fqns=prefilter,
    )
    session_context = build_session_context(sg_dir)

    user_prompt = build_slm_prompt(
        user_prompt=prompt,
        project_summary=project_summary,
        file_map=file_map,
        function_index=function_index,
        session_context=session_context,
    )

    plan = SLMPlan()
    try:
        resp = complete(
            user_prompt,
            system=_PLAN_SYSTEM,
            config=LLMConfig(
                model=config.get_cli_model_for_tier("slm"),
                temperature=0.0,
                max_tokens=250,
                timeout=config.slm_timeout,
                max_retries=0,
                api_base=config.get_cli_api_base(),
            ),
        )
        plan.raw_response = resp.text
        plan.input_tokens = resp.input_tokens
        plan.output_tokens = resp.output_tokens
        plan.cost_usd = resp.cost
        plan.latency_ms = (time.time() - start) * 1000

        payload = json.loads(resp.text)
        plan.reasoning = str(payload.get("reasoning", ""))
        calls = payload.get("tool_calls", [])
        for raw in calls[:5]:
            call_type = str(raw.get("type", ""))
            target = str(raw.get("target", ""))
            if not call_type or not target:
                continue
            plan.tool_calls.append(
                SLMToolCall(
                    call_type=call_type,
                    target=target,
                    start_line=int(raw.get("start_line", 0) or 0),
                    end_line=int(raw.get("end_line", 0) or 0),
                    include_neighbors=bool(raw.get("include_neighbors", False)),
                )
            )
    except Exception as e:
        plan.success = False
        plan.error = str(e)

    return plan


# ── Pre-Filter for Large Projects ───────────────────────────────────────


def prefilter_for_slm(
    prompt: str,
    store,
    session_fqns: Optional[Set[str]] = None,
    max_candidates: int = 300,
) -> Set[str]:
    """Pre-filter function index for large projects (>500 functions).

    Combines:
    1. Top PageRank hub functions (top 100)
    2. Functions in mentioned files
    3. Recently modified functions from session

    All pure code — zero LLM cost, ~2ms.
    """
    candidates: Set[str] = set()

    # 1. Top hub functions by PageRank
    if hasattr(store, "pagerank_scores") and store.pagerank_scores:
        top_hubs = sorted(
            store.pagerank_scores.keys(),
            key=lambda f: store.pagerank_scores[f],
            reverse=True,
        )[:100]
        candidates.update(top_hubs)

    # 2. Functions in files mentioned in prompt
    import re
    file_pattern = re.compile(r'[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cs|rb|php)')
    mentioned = file_pattern.findall(prompt)
    for file_ref in mentioned:
        for fqn, sk in store.skeleton_table.items():
            if hasattr(sk, "file_path") and sk.file_path.endswith(file_ref):
                candidates.add(fqn)

    # 3. Session context (recently modified)
    if session_fqns:
        candidates.update(session_fqns)

    # Cap at max_candidates, sorted by PageRank
    if len(candidates) > max_candidates and hasattr(store, "pagerank_scores") and store.pagerank_scores:
        candidates = set(
            sorted(candidates, key=lambda f: store.pagerank_scores.get(f, 0), reverse=True)
            [:max_candidates]
        )

    return candidates


# ── SLM Call ────────────────────────────────────────────────────────────


def slm_extract(
    prompt: str,
    store,
    sg_dir,
    config,
    session_fqns: Optional[Set[str]] = None,
    retry_note: str = "",
) -> SLMResult:
    """Call the SLM to extract entities from a user prompt.

    Args:
        prompt: Raw user prompt.
        store: Loaded IndexStore (has skeleton_table, summaries, etc.)
        sg_dir: Path to .skeletongraph directory.
        config: SGConfig with slm_model, slm_timeout, etc.
        session_fqns: Recently modified FQNs from session.
        retry_note: Additional context if this is a retry.

    Returns:
        SLMResult with extracted entities, mode, concepts.
    """
    start = time.time()

    try:
        from ..llm.provider import complete, LLMConfig
    except ImportError:
        return SLMResult(
            success=False,
            error="litellm not installed. Run: pip install skeletongraph[llm]"
        )

    # Determine if we need pre-filtering
    n_functions = len(store.skeleton_table)
    if n_functions > 500:
        prefilter = prefilter_for_slm(prompt, store, session_fqns, max_candidates=300)
    else:
        prefilter = None  # Send all functions

    # Build segments
    project_summary = build_project_summary(sg_dir)
    session_context = build_session_context(sg_dir)
    file_map = build_file_map(
        store.file_skeletons,
        file_summaries=getattr(store, "file_summaries", None),
    )

    # Access summaries store
    summaries_dict = {}
    if hasattr(store, "summaries"):
        summaries_dict = store.summaries._store if hasattr(store.summaries, "_store") else {}

    function_index = build_function_index(
        store.skeleton_table,
        summaries_dict,
        pagerank_scores=getattr(store, "pagerank_scores", None),
        max_entries=config.slm_max_fqns_in_prompt,
        prefilter_fqns=prefilter,
    )

    user_msg = build_slm_prompt(
        user_prompt=prompt,
        project_summary=project_summary,
        file_map=file_map,
        function_index=function_index,
        session_context=session_context,
        retry_note=retry_note,
    )

    # Call SLM
    try:
        llm_config = LLMConfig(
            model=config.get_cli_model_for_tier("slm"),
            temperature=0.0,    # Deterministic extraction
            max_tokens=300,     # Structured JSON, ~100-200 tokens
            timeout=config.slm_timeout,
            max_retries=1,
            api_base=config.get_cli_api_base(),
        )
        resp = complete(user_msg, system=_EXTRACT_SYSTEM, config=llm_config)
    except Exception as e:
        logger.warning("SLM call failed: %s", e)
        return SLMResult(
            success=False,
            error=str(e),
            latency_ms=(time.time() - start) * 1000,
        )

    latency = (time.time() - start) * 1000

    # Parse response
    result = _parse_slm_response(resp.text)
    result.raw_response = resp.text
    result.input_tokens = resp.input_tokens
    result.output_tokens = resp.output_tokens
    result.cost_usd = resp.cost
    result.latency_ms = latency

    logger.info(
        "SLM extracted %d entities in %.0fms (mode=%s, cost=$%.4f)",
        len(result.entities), latency, result.mode, result.cost_usd,
    )

    return result


def _parse_slm_response(text: str) -> SLMResult:
    """Parse the SLM's JSON response into an SLMResult.

    Handles: raw JSON, JSON in markdown fences, partial/malformed JSON.
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        import re
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return SLMResult(success=False, error=f"Failed to parse SLM JSON: {text[:200]}")
        else:
            return SLMResult(success=False, error=f"No JSON in SLM response: {text[:200]}")

    # Extract fields
    entities = []
    for e in data.get("entities", []):
        if isinstance(e, dict) and "fqn" in e:
            entities.append(SLMEntity(
                fqn=e["fqn"],
                role=e.get("role", "target"),
            ))

    return SLMResult(
        mode=data.get("mode", ""),
        entities=entities,
        concepts=data.get("concepts", []),
        files=data.get("files", []),
        reasoning=data.get("reasoning", ""),
        success=True,
    )


# ── Turn Summarization ──────────────────────────────────────────────────


def slm_summarize_turn(
    agent_response: str,
    config,
    max_summary_tokens: int = 50,
) -> str:
    """Use SLM to compress an agent's turn response into 1-2 sentences.

    Only called when response is long (>500 chars) and config.enable_slm_turn_summary.
    Cost: ~$0.0001 per call.
    """
    try:
        from ..llm.provider import complete, LLMConfig
    except ImportError:
        return ""

    system = "Summarize this coding agent response in 1-2 sentences. Focus on: what was changed, what was decided, what files were modified. Be concise."

    # Cap input to ~800 tokens
    truncated = agent_response[:3200]

    try:
        resp = complete(
            truncated,
            system=system,
            config=LLMConfig(
                model=config.slm_model,
                temperature=0.0,
                max_tokens=max_summary_tokens,
                timeout=3,
                max_retries=0,
            ),
        )
        return resp.text.strip()
    except Exception:
        return ""


# ── Batch Summarization (for sg build / sg update) ──────────────────────


def batch_summarize_functions(
    bodies: List[str],
    fqns: List[str],
    config,
    batch_size: int = 10,
) -> Dict[str, str]:
    """Summarize multiple function bodies using SLM.

    Used during `sg build` (auto-summarize top 20%) and `sg summarize` (all).
    Cost: ~$0.0002 per function with Flash.
    """
    try:
        from ..llm.provider import complete, LLMConfig
    except ImportError:
        logger.warning("litellm not installed — skipping summarization")
        return {}

    system = (
        "Summarize this function in ONE sentence. "
        "Describe WHAT it does, not HOW. "
        "Max 15 words. No code. No markdown."
    )

    results: Dict[str, str] = {}
    summary_model = getattr(config, "summary_model", None) or config.slm_model
    llm_config = LLMConfig(
        model=summary_model,
        temperature=0.0,
        max_tokens=40,
        timeout=5,
        max_retries=1,
    )

    for i, (body, fqn) in enumerate(zip(bodies, fqns)):
        if not body.strip():
            continue
        try:
            # Cap body at ~1000 tokens
            truncated = body[:4000]
            resp = complete(truncated, system=system, config=llm_config)
            summary = resp.text.strip().rstrip(".")
            if summary:
                results[fqn] = summary
        except Exception as e:
            logger.debug("Failed to summarize %s: %s", fqn, e)

        # Log progress
        if (i + 1) % batch_size == 0:
            logger.info("Summarized %d/%d functions", i + 1, len(bodies))

    return results
