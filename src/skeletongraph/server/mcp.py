"""
MCP (Model Context Protocol) Server for IDE integration.

Exposes SkeletonGraph as tools that IDE agents (Claude Code, Cursor, etc.) can call:
  - query_context: Main entry — prompt → assembled context
  - expand_function: Page-fault — request full body of a specific function
  - show_graph: Visualize dependencies for a function
  - search_index: Keyword search across all indexed functions
  - index_status: Check index health and stats

Protocol: JSON-RPC over stdio (stdin/stdout) per MCP spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..storage.local import IndexStore, load_index
from ..retrieval.resolver import resolve_context
from ..assembly.zone_assembler import assemble_context


# ── Tool Registry ──────────────────────────────────────────────────────

_TOOLS: Dict[str, Dict[str, Any]] = {}


def tool(name: str, description: str, parameters: dict):
    """Decorator to register an MCP tool."""
    def decorator(func: Callable):
        _TOOLS[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": parameters,
            },
            "handler": func,
        }
        return func
    return decorator


# ── Tool Definitions ───────────────────────────────────────────────────

@tool(
    name="query_context",
    description=(
        "Retrieve assembled context for a coding task. Returns token-minimal, "
        "constraint-preserving context with full target code + structural neighbors."
    ),
    parameters={
        "prompt": {"type": "string", "description": "Natural language coding task"},
        "budget": {"type": "integer", "description": "Token budget limit", "default": 128000},
    },
)
def query_context_tool(
    store: IndexStore, project_root: Path, prompt: str, budget: int = 128000,
) -> dict:
    result = resolve_context(prompt, store)
    assembled = assemble_context(
        result, store, project_root, model_context_limit=budget,
    )
    return {
        "context": assembled.text,
        "token_count": assembled.token_count,
        "confidence": assembled.confidence,
        "confidence_reason": assembled.confidence_reason,
        "entities_matched": assembled.entities_matched,
        "zone_breakdown": assembled.zone_breakdown,
        "reduction_ratio": assembled.reduction_ratio,
        "warning": assembled.warning,
    }


@tool(
    name="expand_function",
    description="Page-fault: retrieve the full source code of a specific function by FQN.",
    parameters={
        "fqn": {"type": "string", "description": "Fully qualified name (e.g., 'auth/middleware.py::validate_token')"},
    },
)
def expand_function_tool(
    store: IndexStore, project_root: Path, fqn: str,
) -> dict:
    sk = store.get_skeleton(fqn)
    if sk is None:
        # Try fuzzy match
        matches = store.search(fqn.split("::")[-1] if "::" in fqn else fqn, top_k=3)
        if matches:
            return {
                "error": f"FQN '{fqn}' not found. Did you mean: {', '.join(matches)}",
                "suggestions": matches,
            }
        return {"error": f"FQN '{fqn}' not found"}

    # Read full body
    file_path = project_root / sk.file_path
    body = ""
    if file_path.exists():
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(lines[sk.line_start - 1:sk.line_end])

    summary = store.summaries.get(fqn) or ""

    return {
        "fqn": fqn,
        "file_path": sk.file_path,
        "line_range": f"{sk.line_start}-{sk.line_end}",
        "signature": sk.signature,
        "kind": sk.kind.value,
        "summary": summary,
        "source": body,
        "token_count": len(body) // 4,
    }


@tool(
    name="show_graph",
    description="Show dependency graph around a function: callers, callees, and test coverage.",
    parameters={
        "fqn": {"type": "string", "description": "FQN to center the graph on"},
        "depth": {"type": "integer", "description": "Max traversal depth", "default": 2},
    },
)
def show_graph_tool(
    store: IndexStore, project_root: Path, fqn: str, depth: int = 2,
) -> dict:
    if not store.graph.has_node(fqn):
        return {"error": f"FQN '{fqn}' not in graph"}

    callers = store.graph.blast_radius(fqn, max_depth=depth)
    callees = store.graph.dependency_chain(fqn, max_depth=depth)
    tests = store.graph.test_coverage(fqn)

    return {
        "fqn": fqn,
        "callers": {k: v for k, v in sorted(callers.items(), key=lambda x: x[1])},
        "callees": {k: v for k, v in sorted(callees.items(), key=lambda x: x[1])},
        "tests": tests,
        "total_blast_radius": len(callers),
        "total_dependencies": len(callees),
    }


@tool(
    name="search_index",
    description="Search the function index by keyword. Returns top matching FQNs with signatures.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "description": "Max results", "default": 10},
    },
)
def search_index_tool(
    store: IndexStore, project_root: Path, query: str, top_k: int = 10,
) -> dict:
    results = store.inverted_index.search(query, top_k=top_k)
    entries = []
    for fqn, score in results:
        sk = store.get_skeleton(fqn)
        if sk:
            entries.append({
                "fqn": fqn,
                "signature": sk.signature,
                "file": sk.file_display,
                "kind": sk.kind.value,
                "score": round(score, 2),
            })
    return {"results": entries, "total": len(entries)}


@tool(
    name="index_status",
    description="Check index health: file count, function count, edge count, build time.",
    parameters={},
)
def index_status_tool(
    store: IndexStore, project_root: Path,
) -> dict:
    return {
        "status": store.status_summary(),
        "version": store.meta.version,
        "files": store.meta.total_files,
        "functions": store.meta.total_functions,
        "edges": store.meta.total_edges,
        "languages": store.meta.languages,
        "build_time": f"{store.meta.build_duration_seconds:.2f}s",
    }


# ── JSON-RPC Server ───────────────────────────────────────────────────

def _handle_request(
    request: dict,
    store: IndexStore,
    project_root: Path,
) -> dict:
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return _jsonrpc_response(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "skeletongraph", "version": "0.1.0"},
        })

    elif method == "tools/list":
        tools_list = []
        for name, tool_def in _TOOLS.items():
            tools_list.append({
                "name": tool_def["name"],
                "description": tool_def["description"],
                "inputSchema": tool_def["inputSchema"],
            })
        return _jsonrpc_response(req_id, {"tools": tools_list})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        if tool_name not in _TOOLS:
            return _jsonrpc_error(req_id, -32601, f"Unknown tool: {tool_name}")

        handler = _TOOLS[tool_name]["handler"]
        try:
            result = handler(store, project_root, **tool_args)
            return _jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            })
        except Exception as e:
            return _jsonrpc_error(req_id, -32000, str(e))

    elif method == "notifications/initialized":
        return None  # No response for notifications

    else:
        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def _jsonrpc_response(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def start_server(
    store: IndexStore,
    project_root: Path,
    port: int = 3500,
) -> None:
    """Start the MCP server, reading JSON-RPC from stdin, writing to stdout.

    For stdio transport (used by Claude Code, Cursor, etc.):
    Each message is a JSON object, one per line.
    """
    import sys

    # Stdio mode
    sys.stderr.write(f"SkeletonGraph MCP server started. Index: {store.status_summary()}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = _handle_request(request, store, project_root)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
