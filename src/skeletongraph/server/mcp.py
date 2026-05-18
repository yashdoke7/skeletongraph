"""
MCP server: 7 canonical tools for IDE agent integration.

Tools (in order of typical use):
  sg_overview   — project DNA + architecture + constraints + digest (call first)
  sg_search     — hybrid search: BM25 + graph centrality (PREFERRED over native grep)
  sg_get        — get a function (or several) by FQN (signature + summary)
  sg_expand     — full function body / file view on demand
  sg_constraint — view or propose project constraints
  sg_log        — read project memory: recent turns, or recorded decisions
  sg_decision   — record a design/implementation decision for later recall

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
            "PRIMARY retrieval tool — use this AS THE FINAL STEP for most retrievals.\n\n"
            "Returns expanded context for the query:\n"
            "  • Top 3 matches: signature + summary + body excerpt + 1-hop callers\n"
            "  • Top 4..N: signature + summary (no body)\n"
            "  • Retrieval confidence (HIGH/MEDIUM/LOW) so you know if results are trustworthy\n\n"
            "PREFERRED over native grep/glob. Hybrid BM25 + graph centrality.\n"
            "No need to chain sg_get/sg_expand for typical retrieval flows — this tool "
            "already returns what you need. Only call sg_expand if you need MORE body lines "
            "than the excerpt, or sg_get for a different FQN."
        ),
        {
            "query": {
                "type": "string",
                "description": "Natural language or keyword query",
            },
            "top_n": {
                "type": "integer",
                "description": "Total candidates returned (default: 10)",
                "default": 10,
            },
            "expand_top": {
                "type": "integer",
                "description": "How many top results get full body excerpt + callers (default: 3)",
                "default": 3,
            },
            "file_filter": {
                "type": "string",
                "description": "Optional: restrict to files matching this substring",
                "default": "",
            },
            "intent": {
                "type": "string",
                "description": "Optional task intent to shape retrieval — e.g. "
                               "debug_targeted, explain, refactor, review, architecture. "
                               "Omit to let SG infer it.",
                "default": "",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for the whole response (default: 4000)",
                "default": 4000,
            },
        },
        ["query"],
    ),
    _make_tool(
        "sg_get",
        (
            "Get one or more functions/classes by fully-qualified name (FQN). "
            "Returns signature, docstring, summary, callers, and callees.\n\n"
            "FQN format: 'path/to/file.py::ClassName.method_name'\n"
            "Batch: pass several FQNs comma-separated to fetch them in one call.\n"
            "Use sg_search first if you don't know the exact FQN."
        ),
        {
            "fqn": {
                "type": "string",
                "description": "FQN of the function/class. Pass several "
                               "comma-separated for a batch fetch in one call.",
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
            "Read or append project memory without re-reading full context.\n"
            "  action=read (default):\n"
            "    kind=turns    — recent turn notes (what was done)\n"
            "    kind=decision — recorded design decisions, optionally by topic\n"
            "  action=append — record a one-line note of what you did this turn.\n"
            "    Call once at the END of a turn; it is cheap and keeps memory\n"
            "    current for later turns/sessions."
        ),
        {
            "action": {
                "type": "string",
                "enum": ["read", "append"],
                "description": "read: return memory. append: record a turn note.",
                "default": "read",
            },
            "note": {
                "type": "string",
                "description": "action=append only — a one-line conceptual summary "
                               "of what you changed or decided this turn.",
                "default": "",
            },
            "kind": {
                "type": "string",
                "enum": ["turns", "decision"],
                "description": "action=read: turns = recent turn notes; "
                               "decision = recorded decisions.",
                "default": "turns",
            },
            "topic": {
                "type": "string",
                "description": "When kind=decision, filter decisions by topic/keyword.",
                "default": "",
            },
            "last_n": {
                "type": "integer",
                "description": "Number of entries to return (default: 10)",
                "default": 10,
            },
            "session_id": {
                "type": "string",
                "description": "kind=turns only: specific session ID (default: most recent)",
                "default": "",
            },
        },
        [],
    ),
    _make_tool(
        "sg_decision",
        (
            "Record a notable design or implementation decision so it survives "
            "once it scrolls out of recent context — an approach picked or "
            "rejected, and WHY. Recall later with sg_log(kind='decision')."
        ),
        {
            "summary": {
                "type": "string",
                "description": "One line: what was decided.",
            },
            "rationale": {
                "type": "string",
                "description": "Why — the reasoning that should outlive this turn.",
                "default": "",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files this decision concerns.",
                "default": [],
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short tags for later topic-filtered retrieval.",
                "default": [],
            },
        },
        ["summary"],
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
            "sg_decision": self._tool_decision,
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

        # IDE parity: drain the summary queue after a tool call (background).
        self._maybe_drain()

        return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def _read_sg_file(self, name: str, max_chars: int) -> str:
        """Read a capped `.skeletongraph/<name>` file. '' if missing/empty."""
        try:
            p = self._sg_dir / name
            if not p.exists():
                return ""
            return p.read_text(encoding="utf-8", errors="replace").strip()[:max_chars]
        except Exception:
            return ""

    def _maybe_drain(self) -> None:
        """Drain the summary queue in the background — IDE parity. The engine's
        query path drains via the Claude Code hook; the MCP/pull path needs its
        own trigger. Best-effort; lock-guarded inside drain_queue_background."""
        if not getattr(self._config, "summary_queue_enabled", False):
            return
        try:
            from ..summary.queue import drain_queue_background
            drain_queue_background(self._root, self._config)
        except Exception:
            pass

    # ── Tool: sg_overview ────────────────────────────────────────────────

    def _tool_overview(self, args: Dict) -> str:
        top_n = int(args.get("top_n", 20))
        include_session = bool(args.get("include_session", True))

        parts = [_USE_SG_REMINDER, ""]

        # Project DNA — the glimpse: what this project is, so the model frames
        # its retrieval well before searching.
        proj = self._read_sg_file("project.md", 1000)
        if proj:
            parts.append(f"## Project\n{proj}")

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

        # Architecture outline (module map) — reference for navigation.
        arch = self._read_sg_file("architecture.md", 1600)
        if arch:
            parts.append(f"## Architecture\n{arch}")

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
                # Prefer SummaryStore (Tier-0/0.5/1), fall back to first docstring line
                summary = store.summaries.get(fqn) or ""
                if not summary and sk.docstring:
                    summary = sk.docstring.splitlines()[0].strip()
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

        # Prior decisions — pointer only; pulled on demand to stay cheap.
        try:
            from ..session.decision_log import decision_count
            n_dec = decision_count(self._sg_dir)
            if n_dec:
                parts.append(
                    f"## Memory\n  {n_dec} prior decision(s) on record — call "
                    f"sg_log(kind='decision', topic='...') to recall relevant ones."
                )
        except Exception:
            pass

        return "\n\n".join(parts)

    # ── Tool: sg_search ──────────────────────────────────────────────────

    def _tool_search(self, args: Dict) -> str:
        """Primary retrieval — returns expanded context (bodies for top-N, sigs for rest)."""
        query = str(args.get("query", "")).strip()
        top_n = int(args.get("top_n", 10))
        expand_top = int(args.get("expand_top", 3))
        file_filter = str(args.get("file_filter", "")).strip()
        max_tokens = int(args.get("max_tokens", 4000))
        intent_arg = str(args.get("intent", "")).strip() or None

        if not query:
            return "Error: query is required"

        try:
            store = self._engine.get_store()
            result = self._engine.heuristic_query(
                query, top_n=top_n, file_filter=file_filter or None,
                mode_hint=intent_arg,
            )
        except RuntimeError as e:
            return str(e)

        candidates = result.candidates
        confidence = getattr(result, "confidence", "MEDIUM")

        if not candidates:
            return (
                f"No results for: {query!r}\n"
                f"Confidence: MISS. Try different keywords, or use sg_overview "
                f"to see what's indexed."
            )

        # Soft token budget — stop including bodies once we hit ~70% of the budget
        char_budget = max_tokens * 4
        used_chars = 0

        lines = [
            f"# Search: {query!r}",
            f"Confidence: {confidence}  |  Matches: {len(candidates)}  |  Showing top {min(top_n, len(candidates))}",
            "",
        ]

        # ── Top expand_top: full expansion ───────────────────────────────
        bodies_included = 0
        for i, c in enumerate(candidates[:expand_top], 1):
            sk = c.skeleton

            # Prefer SummaryStore (Tier-0/0.5/1)
            summary = store.summaries.get(sk.fqn) or ""
            if not summary and sk.docstring:
                summary = sk.docstring.splitlines()[0].strip()

            lines.append(f"## {i}. {sk.fqn}")
            lines.append(f"   File: {sk.file_path}:{sk.line_start}")
            lines.append(f"   {sk.signature}")
            if summary:
                lines.append(f"   Summary: {summary[:160]}")

            # Body excerpt (capped to ~40 lines)
            if used_chars < char_budget * 0.7:
                body = self._read_body_excerpt(sk, max_lines=40)
                if body:
                    lines.append("")
                    lines.append("```")
                    lines.append(body)
                    lines.append("```")
                    used_chars += len(body)
                    bodies_included += 1

            # 1-hop callers (brief)
            callers = []
            for other_fqn, other_sk in store.skeleton_table.items():
                if sk.fqn in getattr(other_sk, "callees", []):
                    callers.append(other_fqn)
                    if len(callers) >= 3:
                        break
            if callers:
                lines.append(f"   Called by: {', '.join(callers)}")

            lines.append("")
            used_chars = sum(len(l) for l in lines)
            if used_chars >= char_budget:
                lines.append("   _[token budget reached]_")
                break

        # ── Top expand_top..N: signature + summary only ──────────────────
        remaining = candidates[expand_top:top_n]
        if remaining and used_chars < char_budget:
            lines.append("## Other matches (signatures only)")
            lines.append("")
            for c in remaining:
                sk = c.skeleton
                summary = store.summaries.get(sk.fqn) or ""
                if not summary and sk.docstring:
                    summary = sk.docstring.splitlines()[0].strip()
                entry = f"- `{sk.fqn}`  →  {sk.signature[:80]}"
                if summary:
                    entry += f"  # {summary[:60]}"
                lines.append(entry)

        # Footer
        lines.append("")
        if bodies_included < expand_top:
            lines.append(
                f"_Bodies shown for top {bodies_included}; call sg_expand('<fqn>') "
                f"for more._"
            )
        if confidence == "LOW":
            lines.append(
                "_⚠ LOW confidence — these matches may not be the right targets. "
                "Consider refining your query._"
            )

        return "\n".join(lines)

    def _read_body_excerpt(self, sk, max_lines: int = 40) -> str:
        """Read a function body from disk, capped to max_lines."""
        try:
            file_path = self._root / sk.file_path
            if not file_path.exists():
                return ""
            all_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, sk.line_start - 1)
            end = min(len(all_lines), sk.line_end)
            body_lines = all_lines[start:end]
            if len(body_lines) > max_lines:
                head = body_lines[:max_lines]
                trimmed = len(body_lines) - max_lines
                return "\n".join(head) + f"\n    # ... ({trimmed} more lines, use sg_expand)"
            return "\n".join(body_lines)
        except Exception:
            return ""

    # ── Tool: sg_get ─────────────────────────────────────────────────────

    def _tool_get(self, args: Dict) -> str:
        raw = args.get("fqn", "")
        if isinstance(raw, list):
            fqns = [str(x).strip() for x in raw if str(x).strip()]
        else:
            fqns = [s.strip() for s in str(raw).replace("\n", ",").split(",")
                    if s.strip()]
        include_callers = bool(args.get("include_callers", True))

        if not fqns:
            return "Error: fqn is required"

        try:
            store = self._engine.get_store()
        except RuntimeError as e:
            return str(e)

        results = [self._get_one(store, f, include_callers) for f in fqns]
        return "\n\n---\n\n".join(results)

    def _get_one(self, store, fqn: str, include_callers: bool) -> str:
        """Render one function/class by FQN (fuzzy-resolved)."""
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

        # Prefer SummaryStore (Tier-0/0.5/1) — it includes our generated summaries
        summary = store.summaries.get(resolved_fqn) or ""
        if summary:
            parts.append(f"\nSummary: {summary}")
        elif sk.docstring:
            parts.append(f"\nDocstring:\n{sk.docstring.strip()}")

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
        action = str(args.get("action", "read")).strip().lower()

        # action=append — the model records its own conceptual turn note.
        # The model authors the text, so there is no "unknown output" problem;
        # it is one cheap tool call. Reuses the existing session-log writer.
        if action == "append":
            note = str(args.get("note", "")).strip()
            if not note:
                return "Error: note is required when action=append"
            from ..session.log import append_log
            self._turn_index += 1
            append_log(
                self._sg_dir, self._session_id,
                user_prompt="", summary=note,
                agent_action="model-note", turn_index=self._turn_index,
            )
            return "Turn note recorded."

        kind = str(args.get("kind", "turns")).strip().lower()
        last_n = int(args.get("last_n", 10))

        if kind == "decision":
            from ..session.decision_log import query_decisions, format_decisions
            topic = str(args.get("topic", "")).strip() or None
            decisions = query_decisions(self._sg_dir, topic=topic, limit=last_n)
            if not decisions:
                suffix = f" for topic {topic!r}" if topic else ""
                return f"No recorded decisions{suffix}."
            return format_decisions(decisions)

        # kind=turns (default): recent tool-use turns
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

    # ── Tool: sg_decision ────────────────────────────────────────────────

    def _tool_decision(self, args: Dict) -> str:
        from ..session.decision_log import record_decision
        summary = str(args.get("summary", "")).strip()
        if not summary:
            return "Error: summary is required"
        files = args.get("files", []) or []
        topics = args.get("topics", []) or []
        ok = record_decision(
            self._sg_dir,
            summary,
            rationale=str(args.get("rationale", "")),
            files=list(files) if isinstance(files, list) else [str(files)],
            topics=list(topics) if isinstance(topics, list) else [str(topics)],
        )
        return "Decision recorded." if ok else "Error: could not record decision."


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
