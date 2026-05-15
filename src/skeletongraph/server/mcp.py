"""
MCP server: exactly 6 canonical tools for IDE agent integration.

Tools (in order of typical use):
  sg_overview   — project skeleton + constraints + session digest (call first each session)
  sg_search     — hybrid search: BM25 + graph centrality (PREFERRED over native grep)
  sg_get        — get a specific function by FQN (signature + summary)
  sg_expand     — full function body / file view on demand
  sg_constraint — view or propose project constraints
  sg_log        — read recent session log entries

Protocol: JSON-RPC over stdio per MCP spec.
No module-level state — all state lives in MCPServer instance.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import SGConfig, load_config
from ..engine import SGEngine
from ..session.log import append_log, format_log_digest, read_log

logger = logging.getLogger(__name__)


# ── Use-SG reminder (injected into tool descriptions and sg_overview) ────

_USE_SG_REMINDER = (
    "SkeletonGraph (SG) is your PREFERRED context source for this repo. "
    "Call sg_overview at session start, sg_search instead of native grep/glob, "
    "sg_get/sg_expand instead of reading full files, and sg_constraint to see "
    "project rules before proposing changes."
)


# ── Tool schema registry ─────────────────────────────────────────────────


def _make_tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


_TOOL_SCHEMAS = [
    _make_tool(
        "sg_overview",
        (
            "SESSION START — call this first every session. Returns:\n"
            "  • Project skeleton: top-N central functions by PageRank\n"
            "  • Zone 1 constraints: project rules you must not violate\n"
            "  • Session digest: last 5 turns (what was changed)\n"
            "  • Index stats: file count, function count, staleness\n\n"
            + _USE_SG_REMINDER
        ),
        {
            "top_n": {
                "type": "integer",
                "description": "Number of central functions to include (default: 20)",
                "default": 20,
            },
            "include_session": {
                "type": "boolean",
                "description": "Include session digest (default: true)",
                "default": True,
            },
        },
        [],
    ),
    _make_tool(
        "sg_search",
        (
            "PREFERRED over native grep/glob. Hybrid search: BM25 keyword + "
            "call-graph centrality signal. Returns ranked function list with "
            "signatures and one-line summaries.\n\n"
            "Use this instead of grep or file search when looking for functions, "
            "classes, or code related to a concept."
        ),
        {
            "query": {
                "type": "string",
                "description": "Natural language or keyword query",
            },
            "top_n": {
                "type": "integer",
                "description": "Max results to return (default: 15)",
                "default": 15,
            },
            "file_filter": {
                "type": "string",
                "description": "Optional: restrict to files matching this substring",
                "default": "",
            },
        },
        ["query"],
    ),
    _make_tool(
        "sg_get",
        (
            "Get a specific function or class by fully-qualified name (FQN). "
            "Returns signature, docstring, summary, callers, and callees.\n\n"
            "FQN format: 'path/to/file.py::ClassName.method_name'\n"
            "Use sg_search first if you don't know the exact FQN."
        ),
        {
            "fqn": {
                "type": "string",
                "description": "Fully-qualified name of the function/class",
            },
            "include_callers": {
                "type": "boolean",
                "description": "Include 1-hop callers (default: true)",
                "default": True,
            },
        },
        ["fqn"],
    ),
    _make_tool(
        "sg_expand",
        (
            "PREFERRED over reading full files. Expand a function, class, file, "
            "or line range on demand. Respects token budget.\n\n"
            "target formats:\n"
            "  - FQN: 'src/file.py::MyClass.my_method' (function body)\n"
            "  - File path: 'src/file.py' (full file, token-capped)\n"
            "  - Range: 'src/file.py:42-80' (specific lines)"
        ),
        {
            "target": {
                "type": "string",
                "description": "FQN, file path, or file:start-end range",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget (default: 4000)",
                "default": 4000,
            },
        },
        ["target"],
    ),
    _make_tool(
        "sg_constraint",
        (
            "View or propose project constraints (coding rules, architectural "
            "decisions, style requirements).\n\n"
            "action=view  — list all constraints (confirmed + proposed)\n"
            "action=propose — add a new proposal for human confirmation\n\n"
            "Constraints are injected into every sg_overview (Zone 1). "
            "Proposals require `sg constraint confirm <id>` to promote."
        ),
        {
            "action": {
                "type": "string",
                "enum": ["view", "propose"],
                "description": "view: list constraints. propose: add a new proposal.",
            },
            "text": {
                "type": "string",
                "description": "Constraint text (required when action=propose)",
                "default": "",
            },
            "provenance": {
                "type": "string",
                "description": "Source hint for the proposal (default: 'model-proposed')",
                "default": "model-proposed",
            },
        },
        ["action"],
    ),
    _make_tool(
        "sg_log",
        (
            "Read recent session log entries. Shows what was changed in past turns "
            "without re-reading the full session context."
        ),
        {
            "last_n": {
                "type": "integer",
                "description": "Number of recent entries to return (default: 10)",
                "default": 10,
            },
            "session_id": {
                "type": "string",
                "description": "Specific session ID (default: most recent session)",
                "default": "",
            },
        },
        [],
    ),
]


# ── MCPServer ────────────────────────────────────────────────────────────


class MCPServer:
    """MCP server instance. No module-level state.

    One instance per serve() call. All state is on self.
    """

    def __init__(self, project_root: Path, config: Optional[SGConfig] = None):
        self._root = project_root
        self._sg_dir = project_root / ".skeletongraph"
        self._config = config or load_config(project_root)
        self._engine = SGEngine(project_root, self._config)
        self._session_id = str(uuid.uuid4())[:8]
        self._turn_index = 0

    # ── JSON-RPC dispatch ────────────────────────────────────────────────

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = {"tools": _TOOL_SCHEMAS}
            elif method == "tools/call":
                result = self._handle_tool_call(params)
            elif method == "notifications/initialized":
                return {}  # no response needed for notifications
            else:
                return self._error(req_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.exception("Error handling method %s", method)
            return self._error(req_id, -32603, str(e))

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _handle_initialize(self, params: Dict) -> Dict:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "skeletongraph",
                "version": "0.1.0",
            },
        }

    def _handle_tool_call(self, params: Dict) -> Dict:
        name = params.get("name", "")
        args = params.get("arguments", {})

        handlers = {
            "sg_overview": self._tool_overview,
            "sg_search": self._tool_search,
            "sg_get": self._tool_get,
            "sg_expand": self._tool_expand,
            "sg_constraint": self._tool_constraint,
            "sg_log": self._tool_log,
        }
        handler = handlers.get(name)
        if not handler:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

        t0 = time.time()
        try:
            text = handler(args)
            elapsed = (time.time() - t0) * 1000
            logger.debug("Tool %s completed in %.0fms", name, elapsed)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return {"content": [{"type": "text", "text": f"Tool error: {e}"}], "isError": True}

        return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    # ── Tool: sg_overview ────────────────────────────────────────────────

    def _tool_overview(self, args: Dict) -> str:
        top_n = int(args.get("top_n", 20))
        include_session = bool(args.get("include_session", True))

        parts = [_USE_SG_REMINDER, ""]

        # Index stats
        try:
            store = self._engine.get_store()
            meta = store.meta
            parts.append(
                f"## Index\n"
                f"  {meta.total_files} files  •  {meta.total_functions} functions  "
                f"•  {meta.total_edges} edges  •  {', '.join(meta.languages) or 'unknown'}"
            )
        except RuntimeError as e:
            parts.append(f"## Index\n  {e}")
            store = None

        # Zone 1: constraints
        if store:
            cs = store.constraints
            constraint_text = cs.get_all_for_overview() if hasattr(cs, "get_all_for_overview") else cs.get_all_constraints()
            if constraint_text:
                parts.append(f"## Constraints (Zone 1)\n{constraint_text}")

        # Project skeleton: top-N by PageRank
        if store:
            scores = store.pagerank_scores or {}
            top_fqns = sorted(scores, key=lambda f: -scores[f])[:top_n]
            if not top_fqns:
                # Fallback: just list first top_n functions
                top_fqns = list(store.skeleton_table.keys())[:top_n]

            lines = []
            for fqn in top_fqns:
                sk = store.skeleton_table.get(fqn)
                if not sk:
                    continue
                summary = store.summaries.get(fqn) or "" or (sk.docstring or "").splitlines()[0] if sk.docstring else ""
                entry = f"  {sk.signature}"
                if summary:
                    entry += f"  # {summary[:80]}"
                lines.append(entry)

            if lines:
                parts.append(f"## Top functions (PageRank)\n" + "\n".join(lines))

        # Session digest
        if include_session:
            entries = read_log(self._sg_dir, last_n=5)
            digest = format_log_digest(entries, max_turns=5)
            if digest:
                parts.append(digest)

        return "\n\n".join(parts)

    # ── Tool: sg_search ──────────────────────────────────────────────────

    def _tool_search(self, args: Dict) -> str:
        query = str(args.get("query", "")).strip()
        top_n = int(args.get("top_n", 15))
        file_filter = str(args.get("file_filter", "")).strip()

        if not query:
            return "Error: query is required"

        try:
            result = self._engine.heuristic_query(query, top_n=top_n, file_filter=file_filter or None)
        except AttributeError:
            # Fallback if heuristic_query not yet present
            result = self._engine.query(query, delivery="ide")

        if hasattr(result, "candidates"):
            candidates = result.candidates
        elif hasattr(result, "context_text"):
            return result.context_text
        else:
            return str(result)

        if not candidates:
            return f"No results for: {query!r}"

        lines = [f"Results for: {query!r}  ({len(candidates)} matches)\n"]
        for i, c in enumerate(candidates[:top_n], 1):
            sk = c.skeleton
            if file_filter and file_filter not in sk.file_path:
                continue
            summary = (sk.docstring or "").splitlines()[0] if sk.docstring else ""
            lines.append(f"{i}. {sk.fqn}")
            lines.append(f"   {sk.signature}")
            if summary:
                lines.append(f"   # {summary[:100]}")
            lines.append("")

        return "\n".join(lines)

    # ── Tool: sg_get ─────────────────────────────────────────────────────

    def _tool_get(self, args: Dict) -> str:
        fqn = str(args.get("fqn", "")).strip()
        include_callers = bool(args.get("include_callers", True))

        if not fqn:
            return "Error: fqn is required"

        try:
            store = self._engine.get_store()
        except RuntimeError as e:
            return str(e)

        # Fuzzy resolve
        sk = store.skeleton_table.get(fqn)
        resolved_fqn = fqn
        if not sk:
            for k, v in store.skeleton_table.items():
                if k.endswith(fqn) or fqn in k:
                    sk, resolved_fqn = v, k
                    break
        if not sk:
            # Search by short name
            target_name = fqn.split("::")[-1]
            candidates = [
                (k, v) for k, v in store.skeleton_table.items()
                if v.fqn.endswith(f"::{target_name}")
            ]
            if candidates:
                resolved_fqn, sk = candidates[0]

        if not sk:
            return f"Function not found: {fqn}\nTip: use sg_search to find the exact FQN."

        parts = [f"## {resolved_fqn}"]
        parts.append(f"File: {sk.file_path}:{sk.line_start}")
        parts.append(f"\n{sk.signature}")

        if sk.docstring:
            parts.append(f"\nDocstring:\n{sk.docstring.strip()}")

        summary = store.summaries.get(resolved_fqn) or ""
        if summary:
            parts.append(f"\nSummary: {summary}")

        if include_callers:
            callers = []
            for other_fqn, other_sk in store.skeleton_table.items():
                if resolved_fqn in getattr(other_sk, "callees", []):
                    callers.append(other_fqn)
            if callers:
                parts.append(f"\nCallers ({len(callers)}):")
                for caller in callers[:8]:
                    caller_sk = store.skeleton_table.get(caller)
                    if caller_sk:
                        parts.append(f"  {caller_sk.signature}  # {caller}")

        callees = getattr(sk, "callees", [])
        if callees:
            parts.append(f"\nCalls ({len(callees)}):")
            for callee in list(callees)[:8]:
                callee_sk = store.skeleton_table.get(callee)
                if callee_sk:
                    parts.append(f"  {callee_sk.signature}  # {callee}")
                else:
                    parts.append(f"  {callee}")

        parts.append(f"\nUse sg_expand('{resolved_fqn}') to see the full body.")
        return "\n".join(parts)

    # ── Tool: sg_expand ──────────────────────────────────────────────────

    def _tool_expand(self, args: Dict) -> str:
        target = str(args.get("target", "")).strip()
        max_tokens = int(args.get("max_tokens", 4000))

        if not target:
            return "Error: target is required"

        # Parse range syntax: file.py:42-80
        start_line = end_line = None
        if ":" in target and not "::" in target:
            parts_colon = target.rsplit(":", 1)
            range_part = parts_colon[-1]
            if "-" in range_part and all(p.strip().isdigit() for p in range_part.split("-", 1)):
                start_line, end_line = (int(p.strip()) for p in range_part.split("-", 1))
                target = parts_colon[0]

        return self._engine.expand(
            target=target,
            start_line=start_line,
            end_line=end_line,
            max_tokens=max_tokens,
        )

    # ── Tool: sg_constraint ──────────────────────────────────────────────

    def _tool_constraint(self, args: Dict) -> str:
        action = str(args.get("action", "view")).strip().lower()

        try:
            store = self._engine.get_store()
            cs = store.constraints
        except RuntimeError as e:
            return str(e)

        if action == "view":
            items = cs.list_constraints(include_proposed=True)
            if not items:
                raw = cs.get_all_constraints()
                if raw:
                    return f"## Constraints\n{raw}"
                return "No constraints defined yet.\nUse sg_constraint(action='propose', text='...') to add one."

            lines = ["## Constraints\n"]
            for c in items:
                status = "confirmed" if c.confirmed else "proposed"
                lines.append(f"[{c.id}] ({status}) {c.provenance}")
                lines.append(f"  {c.text.strip()[:200]}")
                lines.append("")
            lines.append("Use `sg constraint confirm <id>` to promote a proposal.")
            return "\n".join(lines)

        elif action == "propose":
            text = str(args.get("text", "")).strip()
            if not text:
                return "Error: text is required when action=propose"
            provenance = str(args.get("provenance", "model-proposed"))
            c = cs.propose_constraint(text, source=provenance)
            cs.save_global(self._root)
            return (
                f"Proposal added (id={c.id}).\n"
                f"Run `sg constraint confirm {c.id}` to promote it to confirmed."
            )

        else:
            return f"Unknown action: {action!r}. Use 'view' or 'propose'."

    # ── Tool: sg_log ─────────────────────────────────────────────────────

    def _tool_log(self, args: Dict) -> str:
        last_n = int(args.get("last_n", 10))
        session_id = str(args.get("session_id", "")).strip() or None

        entries = read_log(self._sg_dir, session_id=session_id, last_n=last_n)
        if not entries:
            return "No session log entries found."

        lines = [f"## Session log (last {len(entries)} entries)\n"]
        for e in entries:
            lines.append(f"Turn {e.turn_index}: {e.user_prompt[:100]!r}")
            if e.files_touched:
                lines.append(f"  Files: {', '.join(e.files_touched[:5])}")
            if e.summary:
                lines.append(f"  {e.summary[:120]}")
            lines.append("")
        return "\n".join(lines)


# ── stdio serve loop ─────────────────────────────────────────────────────


def serve(project_root: Path, config: Optional[SGConfig] = None) -> None:
    """Run the MCP server over stdio until EOF."""
    server = MCPServer(project_root, config)
    logger.info("SG MCP server started (project=%s)", project_root)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(error_response), flush=True)
            continue

        response = server.handle(request)
        if response:  # notifications return {}
            print(json.dumps(response), flush=True)
