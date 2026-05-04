"""
MCP (Model Context Protocol) Server for IDE integration.

Exposes SkeletonGraph as tools that IDE agents (Claude Code, Cursor, etc.) can call:
  - query_context: Main entry — prompt → assembled context
  - expand_function: Page-fault — request full body of a specific function
  - show_graph: Visualize dependencies for a function
  - search_index: Keyword search across all indexed functions
  - index_status: Check index health and stats
  - review_delta: Diff-aware context assembly (blast radius)
  - get_blast_radius: Compute blast radius for a function
  - get_dependencies: Show dependency chain for a function
  - detect_changes: Risk-scored change impact analysis
  - get_stats: Token savings dashboard

Protocol: JSON-RPC over stdio (stdin/stdout) per MCP spec.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..storage.local import IndexStore, load_index
from ..retrieval.resolver import resolve_context
from ..assembly.zone_assembler import assemble_context
from ..retrieval.session import Session
from ..config import SGConfig, load_config
from ..metrics.metrics_logger import MetricsLogger
from ..eval.token_counter import measure_text_tokens


# ── Tool Registry ──────────────────────────────────────────────────────

_TOOLS: Dict[str, Dict[str, Any]] = {}

_TOOL_PROFILES: Dict[str, Optional[set]] = {
    "full": None,  # All tools exposed (legacy, high schema overhead)
    "compact": {"query_context", "expand_context"},  # Recommended: one-shot + page-fault
    "minimal": {"query_context"},  # Absolute minimum: one-shot only
}


def _required_parameter_names(parameters: dict) -> List[str]:
    """Infer required MCP parameters while keeping defaulted fields optional."""
    required = []
    for name, schema in parameters.items():
        if schema.get("required") is False or "default" in schema:
            continue
        description = str(schema.get("description", "")).lower()
        if "optional" in description or "default:" in description:
            continue
        required.append(name)
    return required


def tool(name: str, description: str, parameters: dict):
    """Decorator to register an MCP tool."""
    def decorator(func: Callable):
        properties = {
            key: {k: v for k, v in schema.items() if k != "required"}
            for key, schema in parameters.items()
        }
        _TOOLS[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": _required_parameter_names(parameters),
            },
            "handler": func,
        }
        return func
    return decorator


# ── Server State ───────────────────────────────────────────────────────

_server_state: Dict[str, Any] = {
    "store": None,
    "project_root": None,
    "session": None,
    "config": None,
    "metrics": None,
    "tool_profile": None,
}


def _get_store() -> IndexStore:
    return _server_state["store"]


def _get_root() -> Path:
    return _server_state["project_root"]


def _get_session() -> Session:
    return _server_state["session"]


def _get_tool_profile() -> str:
    profile = _server_state.get("tool_profile")
    if not profile:
        config: SGConfig = _server_state.get("config")
        profile = config.mcp_tool_profile if config else "full"
    return str(profile).lower()


def _get_allowed_tools() -> Optional[set]:
    profile = _get_tool_profile()
    return _TOOL_PROFILES.get(profile, None)


def _resolve_skeleton(store: IndexStore, fqn: str) -> tuple[Optional[Any], str, List[str]]:
    """Resolve a skeleton by FQN with a few fuzzy fallbacks."""
    matches: List[str] = []
    sk = store.get_skeleton(fqn)
    if sk:
        return sk, sk.fqn, matches

    alt_fqn = f"src/{fqn}" if not fqn.startswith("src/") else fqn.replace("src/", "", 1)
    sk = store.get_skeleton(alt_fqn)
    if sk:
        return sk, sk.fqn, matches

    target_name = fqn.split("::")[-1]
    candidates = [
        s for s in store.skeleton_table.values()
        if s.fqn.endswith(f"::{target_name}") or s.fqn.endswith(f".{target_name}")
    ]
    if candidates:
        candidates = sorted(candidates, key=lambda x: len(x.file_path))
        matches = [c.fqn for c in candidates]
        sk = candidates[0]
        return sk, sk.fqn, matches

    return None, fqn, matches


# ── Tool Definitions ───────────────────────────────────────────────────

@tool(
    name="query_context",
    description=(
        "Main entry point. Takes a natural language prompt and returns "
        "attention-optimized, layered context with constraints, target code, "
        "structural context, and the task. Uses session memory to avoid "
        "re-sending code the agent already has."
    ),
    parameters={
        "prompt": {"type": "string", "description": "The user's task or question"},
        "budget": {
            "type": "integer",
            "description": "Model context limit in tokens (default: 128000)",
        },
        "detail_level": {
            "type": "string",
            "description": "Context detail level: compact | full (default: compact)",
        },
        "top_n": {
            "type": "integer",
            "description": "Max number of structural skeletons to include (default: 50)",
        },
        "previous_response": {
            "type": "string",
            "description": (
                "Optional. Your previous response to the user. "
                "Used only for token accounting -- does not affect context assembly."
            ),
        },
    },
)
def query_context_tool(params: dict) -> dict:
    store = _get_store()
    root = _get_root()
    session = _get_session()
    config: SGConfig = _server_state.get("config")
    metrics: MetricsLogger = _server_state["metrics"]

    prompt = params["prompt"]
    budget = int(params.get("budget", 128_000))
    profile = _get_tool_profile()
    if "top_n" in params:
        top_n = int(params.get("top_n", 50))
    else:
        top_n = 20 if profile in ("compact", "minimal") else 50
    detail_level = params.get("detail_level")
    if not detail_level:
        if profile == "minimal":
            detail_level = "minimal"
        elif profile == "compact":
            detail_level = "compact"
        else:
            detail_level = config.default_detail_level if config else "compact"
    previous_response = params.get("previous_response", "")

    t0 = time.perf_counter()

    # If agent passed previous_response, retroactively record it
    # onto the LAST turn in the session. This gives us L2 data.
    if previous_response and session._turns:
        last_turn = session._turns[-1]
        last_turn.response_text = previous_response
        last_turn.response_tokens = measure_text_tokens(previous_response)

    result = resolve_context(prompt, store, session=session, top_n=top_n)

    # ── v3 pipeline: classifier → prompt_builder ─────────────────────
    try:
        from ..retrieval.classifier import classify_query
        from ..assembly.prompt_builder import assemble as v3_assemble

        n_files = len({c.skeleton.file_path for c in result.candidates})
        target_fqns = {c.skeleton.fqn for c in result.candidates}
        classification = classify_query(
            intent=result.intent,
            confidence=result.confidence_score,
            target_fqns=target_fqns,
            n_files_involved=n_files,
        )

        v3_result = v3_assemble(
            classification=classification,
            resolver_result=result,
            store=store,
            project_root=root,
            session=session,
        )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        session.save(root)

        # Log metrics
        if metrics:
            files_involved = list({c.skeleton.file_path for c in result.candidates})
            metrics.log_skeleton_query(
                prompt=prompt,
                sg_tokens=v3_result.token_count,
                native_tokens_estimated=int(v3_result.reduction_ratio * v3_result.token_count) if v3_result.reduction_ratio > 0 else 0,
                reduction_ratio=v3_result.reduction_ratio,
                confidence=v3_result.confidence_level,
                entities_matched=result.entities_matched,
                zone_breakdown=v3_result.layer_breakdown,
                session_dedup_count=v3_result.session_dedup_count,
                session_tokens_saved=0,
                files_involved=files_involved,
                duration_ms=duration_ms,
            )

        response = {
            "context": v3_result.text,
            "token_count": v3_result.token_count,
            "confidence": v3_result.confidence_level,
            "mode": v3_result.mode.value,
            "query_type": v3_result.query_type.value,
            "modifiers": v3_result.modifiers,
            "extended_thinking": v3_result.extended_thinking,
            "layers_loaded": v3_result.layers_loaded,
            "layer_breakdown": v3_result.layer_breakdown,
            "reduction_ratio": v3_result.reduction_ratio,
        }

        if v3_result.session_dedup_count > 0:
            response["session_dedup"] = {
                "bodies_skipped": v3_result.session_dedup_count,
            }

        if v3_result.warning:
            response["warning"] = v3_result.warning

        return response

    except Exception as e:
        # Fallback to v2 zone_assembler if v3 fails
        import logging
        logging.getLogger(__name__).warning(f"v3 pipeline failed, falling back to zone_assembler: {e}")

    # ── v2 fallback path ─────────────────────────────────────────────
    assembled = assemble_context(
        result, store, root,
        model_context_limit=budget,
        detail_level=detail_level,
        session=session,
    )

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Save session after each query
    session.save(root)

    # Log metrics
    if metrics:
        files_involved = list({c.skeleton.file_path for c in result.candidates})
        metrics.log_skeleton_query(
            prompt=prompt,
            sg_tokens=assembled.token_count,
            native_tokens_estimated=int(assembled.reduction_ratio * assembled.token_count) if assembled.reduction_ratio > 0 else 0,
            reduction_ratio=assembled.reduction_ratio,
            confidence=assembled.confidence,
            entities_matched=assembled.entities_matched,
            zone_breakdown=assembled.zone_breakdown,
            session_dedup_count=assembled.session_dedup_count,
            session_tokens_saved=assembled.session_tokens_saved,
            files_involved=files_involved,
            duration_ms=duration_ms,
        )

    response = {
        "context": assembled.text,
        "token_count": assembled.token_count,
        "confidence": assembled.confidence,
        "confidence_reason": assembled.confidence_reason,
        "entities_matched": assembled.entities_matched,
        "zone_breakdown": assembled.zone_breakdown,
        "reduction_ratio": assembled.reduction_ratio,
    }

    if assembled.attention_map:
        response["attention_map"] = [
            {
                "zone": z.zone_name,
                "tokens": z.token_count,
                "attention": z.attention_level,
                "bar": z.bar,
            }
            for z in assembled.attention_map
        ]

    if assembled.session_dedup_count > 0:
        response["session_dedup"] = {
            "bodies_skipped": assembled.session_dedup_count,
            "tokens_saved": assembled.session_tokens_saved,
        }

    if assembled.warning:
        response["warning"] = assembled.warning

    return response


@tool(
    name="expand_function",
    description=(
        "Page-fault expansion: get the full body of a specific function. "
        "Use when the skeleton/summary wasn't enough and you need the "
        "complete implementation."
    ),
    parameters={
        "fqn": {"type": "string", "description": "Fully qualified name of the function"},
    },
)
def expand_function_tool(params: dict) -> dict:
    store = _get_store()
    root = _get_root()

    fqn = params["fqn"]
    sk, resolved_fqn, matches = _resolve_skeleton(store, fqn)
    if sk:
        fqn = resolved_fqn

    if not sk:
        suggestion = matches[:3] if matches else "None"
        return {"error": f"Function not found: {fqn}. Did you mean one of: {suggestion}"}

    file_path = root / sk.file_path
    if not file_path.exists():
        return {"error": f"File not found: {sk.file_path}"}

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(lines[sk.line_start - 1:sk.line_end])
    tokens = len(body) // 4
    
    metrics: MetricsLogger = _server_state.get("metrics")
    if metrics:
        metrics.log_tool_usage("expand_function", tokens, [sk.file_path], 0)

    return {
        "fqn": fqn,
        "file": sk.file_path,
        "lines": f"{sk.line_start}-{sk.line_end}",
        "signature": sk.signature,
        "body": body,
        "token_estimate": len(body) // 4,
    }


@tool(
    name="expand_context",
    description=(
        "Bundle expansion for multiple functions in a single call. "
        "Optionally include neighbors, file outlines, and test coverage."
    ),
    parameters={
        "fqns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of fully qualified names to expand",
        },
        "include_body": {
            "type": "boolean",
            "description": "Include full function bodies (default: true)",
        },
        "include_neighbors": {
            "type": "boolean",
            "description": "Include 1-hop callers/callees (default: false)",
        },
        "include_outline": {
            "type": "boolean",
            "description": "Include file-level outline (default: false)",
        },
        "include_tests": {
            "type": "boolean",
            "description": "Include tests that reference targets (default: false)",
        },
        "max_tokens": {
            "type": "integer",
            "description": "Max tokens to return (default: 4000)",
        },
    },
)
def expand_context_tool(params: dict) -> dict:
    store = _get_store()
    root = _get_root()

    fqns = params.get("fqns") or []
    include_body = params.get("include_body", True)
    include_neighbors = params.get("include_neighbors", False)
    include_outline = params.get("include_outline", False)
    include_tests = params.get("include_tests", False)
    max_tokens = int(params.get("max_tokens", 4000))

    parts: List[str] = []
    tokens_used = 0
    resolved: List[str] = []
    errors: List[str] = []

    def _try_add(text: str) -> bool:
        nonlocal tokens_used
        if not text:
            return False
        cost = measure_text_tokens(text)
        if tokens_used + cost > max_tokens:
            return False
        parts.append(text)
        tokens_used += cost
        return True

    # Bodies
    if include_body and fqns:
        _try_add("=== EXPANDED TARGETS ===")
        for fqn in fqns:
            sk, resolved_fqn, matches = _resolve_skeleton(store, fqn)
            if not sk:
                errors.append(
                    f"Function not found: {fqn}. Did you mean one of: {matches[:3] if matches else 'None'}"
                )
                continue
            resolved.append(resolved_fqn)
            file_path = root / sk.file_path
            if not file_path.exists():
                errors.append(f"File not found: {sk.file_path}")
                continue
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            body = "\n".join(lines[sk.line_start - 1:sk.line_end])
            header = f"# {sk.file_display} - {resolved_fqn.split('::')[-1]}"
            if not _try_add(f"{header}\n{body}"):
                break

    # Neighbors
    if include_neighbors and fqns:
        _try_add("\n=== NEIGHBORS (1-hop) ===")
        neighbor_fqns: List[str] = []
        for fqn in resolved or fqns:
            deps = store.graph.dependency_chain(fqn, max_depth=1)
            callers = store.graph.blast_radius(fqn, max_depth=1)
            neighbor_fqns.extend([k for k in deps.keys() if k != fqn])
            neighbor_fqns.extend([k for k in callers.keys() if k != fqn])
        for nfqn in sorted(set(neighbor_fqns)):
            sk = store.get_skeleton(nfqn)
            if not sk:
                continue
            if not _try_add(sk.signature):
                break

    # File outlines
    if include_outline and fqns:
        _try_add("\n=== FILE OUTLINES ===")
        files = {store.get_skeleton(f).file_path for f in resolved or fqns if store.get_skeleton(f)}
        for fp in sorted(files):
            fs = store.file_skeletons.get(fp)
            if not fs:
                continue
            outline_lines = [f"- {fp}"]
            if fs.summary:
                outline_lines.append(f"  # {fs.summary}")
            for sk in fs.all_skeletons[:50]:
                outline_lines.append(f"  {sk.signature}")
            if not _try_add("\n".join(outline_lines)):
                break

    # Test coverage
    if include_tests and fqns:
        _try_add("\n=== TEST COVERAGE ===")
        for fqn in resolved or fqns:
            tests = store.graph.test_coverage(fqn)
            if tests:
                line = f"- {fqn}: {', '.join(tests[:10])}"
                if not _try_add(line):
                    break

    metrics: MetricsLogger = _server_state.get("metrics")
    if metrics:
        metrics.log_tool_usage("expand_context", tokens_used, [str(r).split("::")[0] for r in resolved], 0)

    return {
        "context": "\n\n".join(parts) if parts else "",
        "token_count": tokens_used,
        "resolved": resolved,
        "errors": errors,
    }


@tool(
    name="show_graph",
    description="Show dependency relationships for a function (callers and callees).",
    parameters={
        "fqn": {"type": "string", "description": "Fully qualified name"},
        "depth": {"type": "integer", "description": "Max traversal depth (default: 2)"},
    },
)
def show_graph_tool(params: dict) -> dict:
    store = _get_store()
    fqn = params["fqn"]
    depth = int(params.get("depth", 2))

    # Forward deps (what this calls)
    deps = store.graph.dependency_chain(fqn, max_depth=depth)

    # Reverse deps (what calls this)
    callers = store.graph.blast_radius(fqn, max_depth=depth)

    return {
        "fqn": fqn,
        "calls": {k: v for k, v in deps.items()},
        "called_by": {k: v for k, v in callers.items()},
        "total_dependencies": len(deps),
        "total_callers": len(callers),
    }


@tool(
    name="search_index",
    description="Keyword search across all indexed functions, classes, and methods.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Max results (default: 10)"},
    },
)
def search_index_tool(params: dict) -> dict:
    store = _get_store()
    query = params["query"]
    top_k = int(params.get("top_k", 10))

    results = store.inverted_index.search(query, top_k=top_k)
    items = []
    for fqn, score in results:
        sk = store.get_skeleton(fqn)
        if sk:
            items.append({
                "fqn": fqn,
                "file": sk.file_path,
                "signature": sk.signature,
                "kind": sk.kind.value,
                "score": score,
            })

    return {"query": query, "results": items, "total": len(items)}


@tool(
    name="index_status",
    description="Check index health, stats, and build metadata.",
    parameters={},
)
def index_status_tool(params: dict) -> dict:
    store = _get_store()
    session = _get_session()

    response = {
        "version": store.meta.version,
        "files": store.meta.total_files,
        "functions": store.meta.total_functions,
        "edges": store.meta.total_edges,
        "languages": store.meta.languages,
        "build_time": f"{store.meta.build_duration_seconds:.2f}s",
        "summaries": store.summaries.count,
        "pending_summaries": store.summaries.pending_count,
    }

    if session.turn_count > 0:
        s = session.stats
        response["session"] = {
            "turns": s.total_turns,
            "tokens_used": s.total_sg_tokens,
            "tokens_saved": s.total_saved_tokens,
            "reduction_ratio": round(s.reduction_ratio, 1),
        }

    return response


@tool(
    name="view_file_range",
    description="View a specific line range of a file. Returns raw code.",
    parameters={
        "file_path": {"type": "string", "description": "Relative path to file"},
        "start": {"type": "integer", "description": "1-indexed start line"},
        "end": {
            "type": "integer",
            "description": "Optional. 1-indexed end line (inclusive)",
        },
    },
)
def view_file_range_tool(params: dict) -> dict:
    root = _get_root()
    path = params["file_path"]
    start = max(1, int(params.get("start", 1)))
    end = max(start, int(params.get("end", start + 100)))

    full_path = root / path
    if not full_path.exists():
        return {"error": f"File not found: {path}"}

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        content = "\n".join(lines[start - 1 : end])
        
        metrics: MetricsLogger = _server_state.get("metrics")
        if metrics:
            metrics.log_tool_usage("view_file_range", len(content) // 4, [path], 0)
            
        return {"file": path, "range": f"{start}-{end}", "content": content}
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="view_file_outline",
    description="Get structural overview of a file (imports, classes, constants).",
    parameters={
        "file_path": {"type": "string", "description": "Relative path to file"},
    },
)
def view_file_outline_tool(params: dict) -> dict:
    store = _get_store()
    path = params["file_path"]

    fsk = store.file_skeletons.get(path)
    if not fsk:
        return {"error": f"File not indexed: {path}"}

    return {
        "file": path,
        "lines": fsk.total_lines,
        "imports": fsk.imports,
        "exports": fsk.exports,
        "classes": [{"name": c.fqn.split("::")[-1], "methods": len(c.methods)} for c in fsk.classes],
        "functions": [f.fqn.split("::")[-1] for f in fsk.functions],
        "constants": getattr(fsk, "constants", []),
    }


@tool(
    name="grep_codebase",
    description="Search codebase text. Returns matching files and lines.",
    parameters={
        "pattern": {"type": "string", "description": "Regex pattern"},
        "glob": {
            "type": "string",
            "description": "Glob filter (default: '*')",
        },
    },
)
def grep_codebase_tool(params: dict) -> dict:
    root = _get_root()
    pattern = params["pattern"]
    glob_pattern = params.get("glob", "*")
    
    import re
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    results = []
    
    # We only grep files that are known to the index (avoids node_modules etc.)
    store = _get_store()
    from fnmatch import fnmatch
    
    for path in store.file_skeletons.keys():
        if not fnmatch(path, glob_pattern) and not fnmatch(Path(path).name, glob_pattern):
            continue
            
        full_path = root / path
        if not full_path.exists():
            continue
            
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            matches = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    matches.append({"line": i + 1, "content": line.strip()})
                    if len(matches) > 20: # Cap per file
                        break
            if matches:
                results.append({"file": path, "matches": matches})
                if len(results) >= 10: # Cap total files
                    break
        except Exception:
            continue
            
    return {"pattern": pattern, "results": results}


@tool(
    name="review_delta",
    description=(
        "Diff-aware context assembly for code review. Parses git diff, "
        "computes blast radius for every changed function, and assembles "
        "a single 4-zone review context with risk scores."
    ),
    parameters={
        "target": {
            "type": "string",
            "description": "Git diff target (default: HEAD). Examples: HEAD, main, HEAD~3",
        },
    },
)
def review_delta_tool(params: dict) -> dict:
    store = _get_store()
    root = _get_root()
    target = params.get("target", "HEAD")

    from ..retrieval.detect_changes import detect_changes
    analysis = detect_changes(root, store, diff_target=target)

    return {
        "risk_summary": analysis.risk_summary,
        "total_blast_radius": analysis.total_blast_radius,
        "files_to_review": analysis.files_to_review[:20],
        "changed_functions": [
            {"fqn": fn.fqn, "file": fn.file_path, "type": fn.change_type}
            for fn in analysis.changed_functions
        ],
        "affected_files": [
            {
                "file": af.file_path,
                "risk_score": af.risk_score,
                "risk_reason": af.risk_reason,
                "affected_count": len(af.affected_fqns),
                "distance": af.distance,
            }
            for af in analysis.affected_files[:20]
        ],
    }


@tool(
    name="get_blast_radius",
    description=(
        "Compute the blast radius for a specific function — "
        "everything that could be affected if this function changes."
    ),
    parameters={
        "fqn": {"type": "string", "description": "Fully qualified name"},
        "depth": {"type": "integer", "description": "Max depth (default: 2)"},
    },
)
def get_blast_radius_tool(params: dict) -> dict:
    store = _get_store()
    fqn = params["fqn"]
    depth = int(params.get("depth", 2))

    affected = store.graph.blast_radius(fqn, max_depth=depth)
    items = []
    for affected_fqn, dist in sorted(affected.items(), key=lambda x: x[1]):
        sk = store.get_skeleton(affected_fqn)
        if sk:
            items.append({
                "fqn": affected_fqn,
                "file": sk.file_path,
                "distance": dist,
                "is_exported": sk.is_exported,
            })

    return {
        "source": fqn,
        "affected": items,
        "total_affected": len(items),
    }


@tool(
    name="get_dependencies",
    description="Show the dependency chain for a function — everything it depends on.",
    parameters={
        "fqn": {"type": "string", "description": "Fully qualified name"},
        "depth": {"type": "integer", "description": "Max depth (default: 2)"},
    },
)
def get_dependencies_tool(params: dict) -> dict:
    store = _get_store()
    fqn = params["fqn"]
    depth = int(params.get("depth", 2))

    deps = store.graph.dependency_chain(fqn, max_depth=depth)
    items = []
    for dep_fqn, dist in sorted(deps.items(), key=lambda x: x[1]):
        sk = store.get_skeleton(dep_fqn)
        if sk:
            items.append({
                "fqn": dep_fqn,
                "file": sk.file_path,
                "signature": sk.signature,
                "distance": dist,
            })

    return {
        "source": fqn,
        "dependencies": items,
        "total": len(items),
    }


@tool(
    name="detect_changes",
    description="Risk-scored change impact analysis. Shows which files need attention.",
    parameters={
        "target": {
            "type": "string",
            "description": "Git diff target (default: HEAD)",
        },
    },
)
def detect_changes_tool(params: dict) -> dict:
    return review_delta_tool(params)


@tool(
    name="get_stats",
    description="Get token savings statistics for this session.",
    parameters={},
)
def get_stats_tool(params: dict) -> dict:
    session = _get_session()
    store = _get_store()

    return {
        "index": {
            "files": store.meta.total_files,
            "functions": store.meta.total_functions,
            "edges": store.meta.total_edges,
        },
        "session": session.stats.to_dict(),
    }


# ── JSON-RPC Protocol ─────────────────────────────────────────────────

def _handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return _json_rpc_response(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "skeletongraph",
                "version": "0.1.0",
            },
        })

    elif method == "tools/list":
        allowed = _get_allowed_tools()
        tools_list = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in _TOOLS.values()
            if allowed is None or t["name"] in allowed
        ]
        return _json_rpc_response(req_id, {"tools": tools_list})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        allowed = _get_allowed_tools()
        if allowed is not None and tool_name not in allowed:
            return _json_rpc_error(req_id, -32601, f"Tool disabled by profile: {tool_name}")

        if tool_name not in _TOOLS:
            return _json_rpc_error(req_id, -32601, f"Tool not found: {tool_name}")

        try:
            result = _TOOLS[tool_name]["handler"](tool_args)
            return _json_rpc_response(req_id, {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2)},
                ],
            })
        except Exception as e:
            return _json_rpc_error(req_id, -32603, str(e))

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "ping":
        return _json_rpc_response(req_id, {})

    return _json_rpc_error(req_id, -32601, f"Unknown method: {method}")


def _json_rpc_response(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _json_rpc_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def start_server(
    store: IndexStore,
    project_root: Path,
    port: int = 3500,
) -> None:
    """Start the MCP server over stdio.

    Reads JSON-RPC messages from stdin, dispatches to tool handlers,
    writes responses to stdout.
    """
    _server_state["store"] = store
    _server_state["project_root"] = project_root
    _server_state["session"] = Session.load(project_root)
    _server_state["config"] = load_config(project_root)
    _server_state["metrics"] = MetricsLogger(project_root)
    _server_state["tool_profile"] = _server_state["config"].mcp_tool_profile if _server_state["config"] else "full"

    # Stdio mode: read line-delimited JSON-RPC
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = _handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
def main():
    """CLI entry point for the MCP server."""
    import argparse
    parser = argparse.ArgumentParser(description="SkeletonGraph MCP Server")
    parser.add_argument("--path", default=".", help="Project root directory")
    parser.add_argument("--port", type=int, default=3500, help="Management port")
    args = parser.parse_args()

    project_root = Path(args.path).resolve()
    from ..storage.local import load_index
    
    # Must be completely silent so we don't break JSON-RPC over stdio
    store = load_index(project_root)
    
    if store is None:
        sys.exit(1)

    start_server(store, project_root, port=args.port)


if __name__ == "__main__":
    main()
