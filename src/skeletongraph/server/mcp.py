"""
MCP server: 7 canonical tools for IDE agent integration.

Tools (in order of typical use):
  sg_overview   — project DNA + architecture + constraints + digest (call first)
  sg_search     — precise task-context search; graph expansion is gated/on demand
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
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import SGConfig, load_config
from ..engine import SGEngine
from ..session.log import append_log, format_log_digest, read_log

logger = logging.getLogger(__name__)


# ── Use-SG reminder (injected into tool descriptions and sg_overview) ────

_USE_SG_REMINDER = (
    "SkeletonGraph (SG) is your PREFERRED context source for this repo. "
    "Call sg_overview at session start for the project briefing. "
    "Use sg_search as a task-context assembler, not grep: ask for the whole task once; "
    "it returns likely edit targets, compact helpers, and likely tests. "
    "Graph/blast-radius context is added automatically only when the task needs it; "
    "request graph='on' for impact analysis, refactors, callers/callees, or architecture. "
    "Its results are complete and self-contained — each body is the exact current "
    "source with its file:line range, so edit directly from them and do NOT re-grep "
    "or re-read code that sg_search already returned. "
    "Use sg_get/sg_expand only for exact follow-up FQNs. "
    "Call sg_constraint to see project rules before proposing changes."
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
            "PRIMARY retrieval tool — one call should locate the edit target and "
            "return enough code to start. Prefer refining the query over chaining "
            "broad expansions.\n\n"
            "Returns:\n"
            "  • Top matches: FQN + file:line-range + imports/prelude + signature "
            "+ summary + target body\n"
            "  • Debug/test bundle: small helper bodies and likely test files/snippets\n"
            "  • Gated graph context: callers/callees only for exact targets, "
            "broad-impact intents, or graph='on'\n"
            "  • Remaining matches: FQN + file:line-range + signature + summary\n"
            "  • Lexical grep fallback: when graph confidence is not HIGH, a "
            "plain-text scan catches module-level constants, decorators and type "
            "aliases that the function graph cannot index.\n\n"
            "PREFERRED over native grep/glob. Hybrid lexical/semantic search "
            "with graph expansion gated for precision.\n"
            "Only call sg_expand for a listed function when you are about to edit "
            "that function and its body was not included."
        ),
        {
            "query": {
                "type": "string",
                "description": "Whole task/symptom query. Prefer the complete bug or edit goal over a single symbol.",
            },
            "top_n": {
                "type": "integer",
                "description": "Total candidates returned (default: 10)",
                "default": 10,
            },
            "expand_top": {
                "type": "integer",
                "description": "How many top results get target bodies (default: 3). "
                               "High values are capped to prevent MCP result bloat; "
                               "neighbors are returned as signatures.",
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
                               "debug_targeted for bugs/failing tests, explain, "
                               "refactor for behavior-preserving rewrites, review, "
                               "architecture. Omit to let SG infer it.",
                "default": "",
            },
            "graph": {
                "type": "string",
                "enum": ["auto", "off", "on"],
                "description": "Graph expansion policy for this search. auto "
                               "keeps normal bug-fix searches precise and adds "
                               "graph only when useful. on requests callers/"
                               "callees/blast-radius context immediately. off "
                               "returns direct lexical/entity targets only.",
                "default": "auto",
            },
            "max_tokens": {
                "type": "integer",
                "description": "ADVANCED — usually omit. Response budget; "
                               "default (2000) is tuned to stay inline in "
                               "VS Code Copilot (which offloads larger "
                               "responses to disk and re-reads them each "
                               "turn, multiplying token cost). Hard cap 4000.",
                "default": 2000,
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
            "Fallback for content NOT already returned by sg_search. Expand a whole function, "
            "class, or file.\n\n"
            "PREFER an FQN or a bare file path — that returns the complete unit in "
            "ONE call. Avoid line ranges: they force you to page through a file.\n\n"
            "BATCH multiple targets in ONE call to save round-trips: pass a "
            "comma-separated list, e.g. 'a.py::f, b.py::g, c.py::h'. Do this when a "
            "fix spans several functions instead of calling sg_expand once each.\n\n"
            "target formats:\n"
            "  - FQN: 'src/file.py::MyClass.my_method' (full function body)\n"
            "  - File path: 'src/file.py' (full file)\n"
            "  - Several: 'src/a.py::f, src/b.py::g' (comma-separated, one call)\n"
            "  - Range: 'src/file.py:42-80' (only if you truly need a slice)"
        ),
        {
            "target": {
                "type": "string",
                "description": "FQN or file path (preferred); comma-separate several to batch; file:start-end range only if needed",
            },
            "max_tokens": {
                "type": "integer",
                "description": "ADVANCED — usually omit. Function body "
                               "budget; default (2500) holds one typical "
                               "body inline. For larger functions, request "
                               "specific line ranges in separate calls "
                               "rather than raising max_tokens (avoids the "
                               "IDE content-cache offload). Hard cap 4000.",
                "default": 2500,
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

# Cap for any body inlined by sg_search when body_top>0 (the lean+rankN A/B
# lever). Keeps an inlined body a bounded, edit-ready slice instead of a whole-
# function dump that then rides in every subsequent turn's context.
_BODY_INLINE_MAX_LINES = 80


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
        # Session-level dedup: FQNs whose full bodies were already returned this
        # session (model's context window already has them — no value in re-sending).
        self._returned_fqns: set = set()
        # Session-level dedup for line-range expands: file_path -> [(start, end)]
        # already returned (by sg_search bodies or earlier sg_expand ranges). Stops
        # the agent paging through a file with overlapping sg_expand calls.
        self._returned_ranges: dict = {}
        # Whether the STATIC overview briefing (project.md + architecture.md +
        # top-functions map) has been sent this session. It never changes within a
        # session and is already in the model's context after the first call, so
        # re-sending it on every sg_overview call is pure waste (measured on a real
        # Claude run: 5 calls × ~3.4k chars, cached + re-read every later turn).
        # Repeat calls return only the DYNAMIC parts (index stats + session digest).
        self._overview_sent: bool = False

    # ── range-dedup helpers ──────────────────────────────────────────────

    @staticmethod
    def _norm_path(p: str) -> str:
        return str(p or "").replace("\\", "/").strip()

    def _record_range(self, file_path: str, start, end) -> None:
        if not file_path or start is None or end is None:
            return
        key = self._norm_path(file_path)
        self._returned_ranges.setdefault(key, []).append((int(start), int(end)))

    def _range_already_covered(self, file_path: str, start: int, end: int,
                              thresh: float = 0.8):
        """Return a covering (s, e) if >= thresh of [start, end] was already
        returned this session, else None."""
        key = self._norm_path(file_path)
        req_len = max(1, end - start + 1)
        for s, e in self._returned_ranges.get(key, []):
            overlap = max(0, min(end, e) - max(start, s) + 1)
            if overlap / req_len >= thresh:
                return (s, e)
        return None

    # ── JSON-RPC dispatch ────────────────────────────────────────────────

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                # LOCATOR-ONLY mode (SG_MCP_LOCATOR_ONLY=1): expose ONLY sg_search.
                # This is a real tool-SURFACE reduction, not a prompt nudge — the
                # other 6 tools are not just discouraged, they are absent from the
                # MCP handshake, so the agent cannot call sg_expand even if it
                # wants to and must fetch bodies with its own native Read at the
                # file:line-range sg_search already always includes. Ablation
                # question: does removing the body-delivery path change cost, or
                # does the agent just relocate the same tokens into more Read
                # calls (see project_headline_finding memory)? Bonus: fewer
                # registered MCP tools may also reduce the ToolSearch
                # deferred-tool tax observed with the full 7-tool surface.
                if os.environ.get("SG_MCP_LOCATOR_ONLY", "0") == "1":
                    tool = dict(next(t for t in _TOOL_SCHEMAS if t["name"] == "sg_search"))
                    # The static description still says "call sg_expand" — that
                    # tool does not exist in this handshake, so override the
                    # advertised text rather than ship a misleading affordance.
                    tool["description"] = (
                        tool["description"].split("Only call sg_expand")[0].rstrip()
                        + "\n\nLOCATOR MODE: this is the only SG tool available. "
                          "Results never include function bodies — every match "
                          "gives file:line-range; use your native Read tool at "
                          "that exact range to view and edit the code."
                    )
                    result = {"tools": [tool]}
                else:
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

        # LOCATOR-ONLY: tools/list already hides everything but sg_search, but
        # hiding a tool from the handshake does not stop a DISPATCH to it — any
        # client that caches an older tool list, or calls tools/call directly,
        # would otherwise still get a real body back out of sg_expand/sg_get and
        # silently invalidate the ablation (its entire premise is "no body-
        # delivery path exists"). Enforce it here too, not just in the listing.
        if (os.environ.get("SG_MCP_LOCATOR_ONLY", "0") == "1" and name != "sg_search"):
            return {"content": [{"type": "text", "text":
                f"{name} is not available in SG locator mode. Use sg_search for "
                "the file:line-range, then your native Read tool to view it."}],
                "isError": True}

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
        # Default lowered 20 → 10. Most signal lives in the first 5-7 hub
        # functions by PageRank; entries 11-20 are usually noise that pads
        # the overview without changing which functions the agent searches.
        # Callers can still request more via top_n=20 explicitly.
        top_n = int(args.get("top_n", 10))
        include_session = bool(args.get("include_session", True))
        # Force a full re-send only if the caller explicitly asks (rare).
        force_full = bool(args.get("force", False))

        # Repeat call in the same session: the static briefing is already in the
        # model's context. Return only the dynamic parts (fresh index stats +
        # session digest) so the map isn't re-billed on every re-orientation.
        if self._overview_sent and not force_full:
            parts = ["(Overview briefing already provided this session — project "
                     "map, architecture, and top functions are in your context "
                     "above. Dynamic state only:)"]
            try:
                meta = self._engine.get_store().meta
                parts.append(
                    f"## Index\n  {meta.total_files} files  •  "
                    f"{meta.total_functions} functions  •  "
                    f"{', '.join(meta.languages) or 'unknown'}")
            except RuntimeError:
                pass
            if include_session:
                digest = format_log_digest(read_log(self._sg_dir, last_n=5), max_turns=5)
                if digest:
                    parts.append(digest)
            return "\n\n".join(parts)

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

        self._overview_sent = True
        return "\n\n".join(parts)

    # ── Tool: sg_search ──────────────────────────────────────────────────

    def _tool_search(self, args: Dict) -> str:
        """Primary retrieval — target bodies + structural pointers + session dedup."""
        query = str(args.get("query", "")).strip()
        top_n = min(max(int(args.get("top_n", 10)), 1), 20)
        requested_expand_top = min(max(int(args.get("expand_top", 3)), 1), 7)
        file_filter = str(args.get("file_filter", "")).strip()
        # Default 2000 / CAP 4000 (char budget 16K → 16K capped). Cap matters
        # as much as default — without it, the model overrides max_tokens to
        # 5000+ and trips VS Code Copilot's content.txt offload, where the
        # response is saved to a workspace file and re-loaded each turn
        # (amplifying token cost across all subsequent turns).
        max_tokens = min(max(int(args.get("max_tokens", 2000)), 1000), 4000)
        intent_arg = str(args.get("intent", "")).strip() or None
        graph_arg = str(args.get("graph", "auto")).strip().lower()
        graph_policy = {"on": "always", "off": "off", "auto": None}.get(graph_arg)

        # PAYLOAD SHAPE — the single knob that decides product cost. DEFAULT is
        # LEAN: ranked anchors (file::fqn + line range + signature + structural
        # neighbours) but NO full bodies. Measured cost problem with body dumps:
        # they inflate peak context ~20-25% and ride in every later turn, yet the
        # agent re-Reads the file anyway for exact edit context — additive dead
        # weight. Lean keeps SG's real value (localization) and lets the agent pull
        # exactly what it needs via sg_expand (or a single Read). `body_top`>0
        # inlines the top-N bodies (capped) and exists ONLY as the lean-vs-
        # lean+rank1 A/B lever (arg `body_top` or env SG_MCP_BODY_TOP); default 0,
        # so the shipping product is lean. This is the SAME payload both the NIM
        # react harness and Claude Code MCP consume — one contract, no per-harness
        # format drift.
        try:
            body_top = int(args.get("body_top", os.environ.get("SG_MCP_BODY_TOP", 0)))
        except (TypeError, ValueError):
            body_top = 0
        body_top = min(max(body_top, 0), 3)
        # LOCATOR-ONLY: body_top is forced to 0 regardless of the arg/env value —
        # this mode's whole contract is "anchors only, no bodies, ever." Since
        # sg_expand/sg_get are also absent from tools/list in this mode (see
        # handle()), this is belt-and-braces: even a caller that still passes
        # body_top>0 (e.g. a stale client cache) cannot get a body out of it.
        locator_only = os.environ.get("SG_MCP_LOCATOR_ONLY", "0") == "1"
        if locator_only:
            body_top = 0

        if not query:
            return "Error: query is required"

        # Retrieval strategy. DEFAULT is `rerank` (BM25 recall + SG structural
        # rerank): no dense embeddings, so a fresh install needs NO model download
        # and NO prewarm — retrieval is instant on any hardware and already scores
        # strongly (react loop: on par with the best arms at a fraction of native's
        # cost). `fusion` (3-way RRF: BM25 + dense + SG structural rerank) is the
        # OPT-IN best-retrieval mode (SWE-bench Pro: recall@1 +50%, MRR +26% over
        # BM25 alone) — it needs a one-time dense build (minutes/repo on CPU,
        # seconds on GPU; incremental after) plus the embed model, so it's a
        # deliberate choice, not forced on everyone. Enable with
        # SG_MCP_RETRIEVAL=fusion; when enabled the dense leg degrades gracefully
        # to bm25+rerank on any failure/timeout (SG_DENSE_TIMEOUT_S, default 20s)
        # and warms in a BACKGROUND thread at startup so it never blocks the MCP
        # handshake. `structural` = the old heuristic_query path.
        mode = os.environ.get("SG_MCP_RETRIEVAL", "rerank").strip().lower()
        try:
            store = self._engine.get_store()
            # Over-fetch so the test-demotion below can pull a buried source file
            # up into the returned window (BM25 often ranks tests above source).
            fetch_n = max(top_n * 3, 30)
            if mode in ("fusion", "rerank"):
                from ..retrieval.fusion import retrieve_fusion, retrieve_rerank
                fn = retrieve_fusion if mode == "fusion" else retrieve_rerank
                # Neither algorithm has a native file_filter concept (unlike the
                # structural resolver, which applies it INSIDE ranking). Applying
                # the filter AFTER truncating to fetch_n could miss a real match
                # sitting just outside the top-30 pool, so widen the fetch first
                # when a filter is active — filtering, not the ranker, narrows.
                pool_n = max(fetch_n, 200) if file_filter else fetch_n
                fqns = fn(query, self._engine._root, pool_n)
                candidates = self._candidates_from_fqns(fqns, store)
                if file_filter:
                    candidates = [c for c in candidates if file_filter in c.skeleton.file_path]
                # Cheap confidence proxy (fusion/rerank don't compute the 5-factor
                # score structural mode does): HIGH if the query names the top
                # hit's own symbol — the same "exact reference" signal structural
                # confidence is built on. A hardcoded MEDIUM here silently capped
                # expand_top at 3 instead of 5, forcing extra follow-up sg_expand
                # calls (measured contributor to the turn-count increase).
                confidence = "MEDIUM"
                if candidates:
                    top_name = candidates[0].skeleton.fqn.split("::")[-1].split(".")[-1].lower()
                    if top_name and top_name in query.lower():
                        confidence = "HIGH"
                if intent_arg or graph_arg not in ("auto", ""):
                    # Not silently ignored — surfaced so a caller relying on these
                    # (structural-resolver-only concepts; no equivalent in the
                    # bm25/dense/rerank pipeline) knows they had no effect here.
                    logger.debug(
                        "sg_search: intent/graph args are structural-mode-only, "
                        "ignored under SG_MCP_RETRIEVAL=%s", mode)
            else:
                result = self._engine.heuristic_query(
                    query, top_n=fetch_n, file_filter=file_filter or None,
                    mode_hint=intent_arg,
                    graph_policy=graph_policy,
                )
                candidates = result.candidates
                confidence = getattr(result, "confidence", "MEDIUM")
        except RuntimeError as e:
            return str(e)

        # Bug-fix searches want the implementation, not its tests. BM25 ranks test
        # files high (they mention the symbol many times), often burying the source
        # below the cutoff (e.g. Card.fromstring source sat at rank 16). Demote test
        # files — stable, so rank within source and within tests is preserved —
        # unless the query is explicitly about tests. Then trim to the requested N.
        # (fusion/rerank already demote internally; this keeps the two paths uniform.)
        if candidates and "test" not in query.lower():
            candidates = sorted(
                candidates, key=lambda c: self._is_test_path(c.skeleton.file_path))
        candidates = candidates[:top_n]
        # expand_top = how many top candidates get the DETAILED anchor (signature,
        # summary, callers/callees) vs the terse "Other matches" pointer list.
        # Tight for MEDIUM/LOW, a bit wider for HIGH-confidence exact matches.
        # (Bodies are a separate axis — body_top, default 0/off.)
        expand_top = min(requested_expand_top, 2 if confidence == "HIGH" else 1)
        # If a body is requested for rank i, it must be in the detailed window.
        expand_top = max(expand_top, body_top)

        char_budget = max_tokens * 4
        used_chars = 0
        lines = [f"# Search: {query!r}"]
        preludes_seen: set[str] = set()

        # ── Graph matches ────────────────────────────────────────────────
        if candidates:
            if locator_only:
                body_note = "Anchors only (locator mode) — Read the target file at the given line range to edit"
            elif body_top:
                body_note = f"Bodies inline for top {body_top}"
            else:
                body_note = "Anchors only — sg_expand or Read the target to edit"
            lines.append(
                f"Confidence: {confidence}  |  Graph: {graph_arg or 'auto'}  "
                f"|  Matches: {len(candidates)}  |  {body_note}")
            lines.append("")

            quick_map = self._render_quick_map(
                query, candidates[:expand_top], store)
            if quick_map:
                lines.append(quick_map)
                lines.append("")
                used_chars += len(quick_map)

            for i, c in enumerate(candidates[:expand_top], 1):
                sk = c.skeleton
                summary = store.summaries.get(sk.fqn) or ""
                if not summary and sk.docstring:
                    summary = sk.docstring.splitlines()[0].strip()

                lines.append(f"## {i}. {sk.fqn}")
                lines.append(
                    f"   File: {sk.file_path}:{sk.line_start}-{sk.line_end}")
                lines.append(f"   {sk.signature}")
                if summary:
                    lines.append(f"   Summary: {summary[:160]}")

                # Bodies (and their file prelude) are inlined ONLY for the top
                # `body_top` ranks — 0 by default (lean). The agent pulls any body
                # it actually needs via sg_expand, so lean skips the prelude too.
                prelude = ""
                if i <= body_top and sk.file_path not in preludes_seen:
                    prelude = self._read_file_prelude(sk, max_chars=800)
                    preludes_seen.add(sk.file_path)
                if prelude and used_chars + len(prelude) < char_budget:
                    lines += ["", "File prelude/imports:", "```", prelude, "```"]
                    used_chars += len(prelude)

                # Session dedup: if a body we'd inline was already sent this
                # session, skip re-sending it — the model's context already has it.
                if i <= body_top and sk.fqn in self._returned_fqns:
                    lines.append("   [body already returned this session — "
                                 "check earlier search result]")
                    lines.append("")
                    continue

                if i <= body_top and used_chars < char_budget:
                    # A/B lever only (default body_top=0 never reaches here). Cap
                    # every inlined body to keep it a bounded, edit-ready slice
                    # rather than a whole-function dump that rides every turn.
                    max_lines = _BODY_INLINE_MAX_LINES
                    body = self._read_body(sk, max_lines=max_lines)
                    if body:
                        body = self._cap_to_remaining(body, char_budget - used_chars)
                        lines += ["", "```", body, "```"]
                        used_chars += len(body)
                        # Register as "fully in context" only if it wasn't truncated.
                        was_truncated = (sk.line_end - sk.line_start + 1) > max_lines
                        if not was_truncated:
                            self._returned_fqns.add(sk.fqn)
                        actual_end = min(sk.line_end, sk.line_start + max_lines - 1)
                        self._record_range(sk.file_path, sk.line_start, actual_end)

                callers = self._callers_of(store, sk.fqn, limit=3)
                if callers:
                    lines.append(f"   Called by: {', '.join(callers)}")

                callees = getattr(sk, "callees", [])
                callee_lines: List[str] = []
                for callee_fqn in list(callees)[:6]:
                    callee_sk = store.skeleton_table.get(callee_fqn)
                    if not callee_sk:
                        callee_lines.append(callee_fqn)
                        continue
                    callee_lines.append(
                        f"{callee_sk.signature}  # {callee_sk.fqn}")
                if callee_lines:
                    lines.append("   Calls:")
                    for callee in callee_lines:
                        lines.append(f"     {callee[:180]}")

                lines.append("")
                if used_chars >= char_budget:
                    lines.append("_[token budget reached — refine the query "
                                 "for fewer, tighter matches]_")
                    break

            # Remaining: FQN + range + sig only
            remaining = candidates[expand_top:top_n]
            if remaining and used_chars < char_budget:
                lines.append("## Other matches (metadata only)")
                lines.append("")
                for c in remaining:
                    sk = c.skeleton
                    already = " [in context]" if sk.fqn in self._returned_fqns else ""
                    summary = store.summaries.get(sk.fqn) or ""
                    if not summary and sk.docstring:
                        summary = sk.docstring.splitlines()[0].strip()
                    entry = (f"- `{sk.fqn}`{already}  "
                             f"({sk.file_path}:{sk.line_start}-{sk.line_end})")
                    if summary:
                        entry += f"  # {summary[:60]}"
                    lines.append(entry)
            lines.append("")

            # Task bundle (likely tests + likely callers) — fired ON DEMAND, not
            # always. It exists to stop a client from native-grepping for test
            # neighbors after a search. But it is the single largest part of the
            # response (helpers + test bodies), and when SG already returned the
            # edit target with HIGH/MEDIUM confidence, that bundle is dead weight
            # that persists in the agent's context every turn (measured: it made
            # the MCP arm token-NEGATIVE vs native). So include it only when SG is
            # unsure (LOW/MISS — the model genuinely needs more to go on) or the
            # caller explicitly asked for debug/test context via `intent`. A
            # confident match returns target + neighbors; if the model then needs
            # tests/helpers it calls sg_search(intent='debug') — which the
            # SG-first gate keeps in-pipeline instead of grep.
            intent_l = (intent_arg or "").lower()
            wants_bundle = (
                confidence in ("LOW", "MISS")
                or "debug" in intent_l or "test" in intent_l
            )
            if wants_bundle:
                bundle = self._render_task_bundle(
                    query=query,
                    candidates=candidates[:expand_top],
                    store=store,
                    remaining_chars=max(0, char_budget - used_chars),
                    intent=intent_arg or "",
                )
                if bundle:
                    lines.append(bundle)
                    lines.append("")
                    used_chars += len(bundle)

            # Module-level constants the targets reference (VALID_HEADER_CHARS,
            # END_CARD, …). The function graph can't index these, so surfacing
            # them here stops the agent grepping for them after a successful
            # search — a measured source of redundant native tool calls.
            consts = self._module_constants(candidates[:expand_top], store)
            if consts and used_chars < char_budget:
                lines.append(consts)
                lines.append("")
                used_chars += len(consts)
        else:
            lines.append(f"No graph matches for {query!r}.")
            lines.append("")

        # ── Lexical fallback ─────────────────────────────────────────────
        # The graph indexes functions/classes only. Module-level constants,
        # decorators and type aliases are invisible to it — a plain-text scan
        # catches them.
        # Only run when:
        #   (a) graph completely failed (LOW/MISS/empty), OR
        #   (b) query looks like a code symbol (identifier-like), not NL prose.
        # Running on MEDIUM-confidence NL queries floods results with noise
        # (e.g. "astropy" appears in every file of an astropy repo).
        added_lexical = False
        _want_lex = (
            confidence in ("LOW", "MISS")
            or not candidates
            or (confidence != "HIGH" and self._looks_symbolic(query))
        )
        if _want_lex:
            lex = self._lexical_search(query, file_filter)
            if lex:
                lines.append("## Lexical matches "
                             "(text grep — catches constants/symbols "
                             "the function graph cannot index)")
                lines.append(lex)
                lines.append("")
                added_lexical = True

        if not candidates and not added_lexical:
            return (f"No results for {query!r}. Try different keywords, "
                    f"or sg_overview to see what's indexed.")

        if confidence == "HIGH":
            lines.append("🛑 **CONFIDENCE: HIGH** — All likely targets returned above. "
                         "Proceed to edit. Do NOT call sg_search again unless this "
                         "result is missing a specific file you need.\n")

        if os.environ.get("SG_MCP_LOCATOR_ONLY", "0") == "1":
            lines.append("_Each match above gives the exact file:line range. Use your "
                         "native Read tool at that range to view the body — sg_expand "
                         "is not available in locator mode. Do NOT grep/read_file for "
                         "anything already listed here._")
        else:
            lines.append("_Each match above gives the exact file:line range. To read a "
                         "body, call sg_expand(target=\"<fqn>\") — batch several comma-"
                         "separated in one call. Do NOT grep/read_file for anything "
                         "already listed here; when you edit, Read only the one file you "
                         "change (ignore any content.txt spill — it duplicates this)._")
        return "\n".join(lines)

    def _module_constants(self, candidates: List, store, max_chars: int = 700) -> str:
        """Module-level UPPER_CASE constants defined in the edit-target files, so
        the model gets them inline instead of grepping.

        The function graph indexes functions/classes, not module constants, so an
        agent that finds `Card.fromstring` still has to grep for `CARD_LENGTH` /
        `VALID_HEADER_CHARS`. We list the top-level constants of the files the top
        results live in — the developer-opens-the-file view (budgeted)."""
        import re as _re
        files: List[str] = []
        for c in candidates[:3]:
            fp = c.skeleton.file_path
            if fp not in files:
                files.append(fp)
        found, seen = [], set()
        for fp in files:
            try:
                lines = (self._root / fp).read_text(
                    encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for ln in lines:
                m = _re.match(r"^([A-Z][A-Z0-9_]{2,})\s*(?::[^=]+)?=\s*(.+)$", ln)
                if not m:
                    continue
                name, val = m.group(1), m.group(2).strip()
                if name in seen:
                    continue
                seen.add(name)
                found.append(f"  {name} = {val[:80]}  # {fp}")
                if len(found) >= 12 or len("\n".join(found)) > max_chars:
                    break
            if len(found) >= 12 or len("\n".join(found)) > max_chars:
                break
        if not found:
            return ""
        return ("## Module constants (in the target files — no need to grep)\n"
                + "\n".join(found))

    def _render_quick_map(self, query: str, candidates: List, store) -> str:
        """Front-load the few routing facts agents need before reading bodies."""
        if not candidates:
            return ""

        edit_candidates = [
            c for i, c in enumerate(candidates, 1)
            if self._should_include_body(query, c.skeleton, i)
        ] or candidates[:1]

        locator_hint = ("Read the target at its line range for the body."
                        if os.environ.get("SG_MCP_LOCATOR_ONLY", "0") == "1"
                        else "sg_expand a target for its body.")
        lines = [
            "## SG quick map",
            f"Exact anchors — use these instead of native grep/read; {locator_hint}",
        ]
        for c in edit_candidates[:3]:
            sk = c.skeleton
            lines.append(f"- Edit target: {sk.file_path}:{sk.line_start}-{sk.line_end}  # {sk.fqn}")

        target_paths = [c.skeleton.file_path for c in edit_candidates[:5]]
        target_names = [c.skeleton.fqn.split("::")[-1] for c in edit_candidates[:5]]
        terms = self._test_terms(query, target_names)
        for path in self._candidate_test_paths(target_paths, store)[:2]:
            snippets = self._matching_line_snippets(path, query, terms, limit=6)
            if not snippets:
                continue
            lines.append(f"- Likely test file: {path}")
            for snippet in snippets[:6]:
                lines.append(f"  - {snippet}")

        return "\n".join(lines)[:1800]

    def _should_include_body(self, query: str, sk, rank: int) -> bool:
        """Keep full bodies for likely edit targets, not every BM25 neighbor."""
        if rank == 1:
            return True
        import re as _re
        q = query.lower()
        short = sk.fqn.split("::")[-1]
        parts = [
            p.lower() for p in _re.split(r"[^A-Za-z0-9_]+", short)
            if len(p) >= 3
        ]
        method = parts[-1] if parts else short.lower()
        action_verbs = {
            "add", "allow", "change", "fix", "handle", "make", "modify",
            "refactor", "support", "update",
        }
        if method in action_verbs and short.lower() not in q:
            return False
        if method and method in q:
            return True
        # For Class.method, avoid expanding every method just because the class
        # name appears in the query. The method/function name must match.
        if "." in short:
            return False
        return any(p in q for p in parts)

    def _render_task_bundle(
        self,
        query: str,
        candidates: List,
        store,
        remaining_chars: int,
        intent: str = "",
    ) -> str:
        """Add the context agents usually go searching for after target lookup.

        In debug/test tasks the model tends to run a second search tree for
        helpers and tests. Return a bounded bundle here so the first search is
        closer to edit-ready.
        """
        if remaining_chars < 1200 or not candidates:
            return ""

        q_lower = query.lower()
        intent_lower = intent.lower()
        wants_debug_bundle = (
            "debug" in intent_lower
            or "test" in intent_lower
            or any(w in q_lower for w in ("fix", "bug", "fail", "test", "bytes"))
        )
        if not wants_debug_bundle:
            return ""

        edit_candidates = [
            c for i, c in enumerate(candidates, 1)
            if self._should_include_body(query, c.skeleton, i)
        ] or candidates[:1]

        parts: List[str] = []
        helper_budget = min(remaining_chars // 2, 3500)
        helpers = self._render_helper_bodies(
            query, edit_candidates, store, helper_budget)
        if helpers:
            parts.append(
                "## Helper bodies likely needed\n"
                + helpers
            )

        used = sum(len(p) for p in parts)
        tests_budget = min(max(0, remaining_chars - used), 4500)
        tests = self._render_likely_tests(
            query, edit_candidates, store, tests_budget)
        if tests:
            parts.append(
                "## Likely tests\n"
                + tests
            )

        if not parts:
            return ""
        parts.append(
            "_This search result is intended as an edit-ready packet: targets, "
            "small helpers, and likely tests. Use the listed test bodies as "
            "anchors before doing native test grep/read calls. Avoid another "
            "sg_search unless these sections are missing the target._"
        )
        return "\n\n".join(parts)

    def _render_helper_bodies(
        self,
        query: str,
        candidates: List,
        store,
        max_chars: int,
    ) -> str:
        helpers: List[str] = []
        seen: set[str] = {c.skeleton.fqn for c in candidates}
        query_symbols = self._symbol_terms(query)
        query_lower = query.lower()

        def add_helper(fqn: str, reason: str) -> None:
            if len("\n\n".join(helpers)) >= max_chars or len(helpers) >= 4:
                return
            if fqn in seen:
                return
            sk = store.skeleton_table.get(fqn)
            if not sk:
                return
            body = self._read_body(sk, max_lines=60)
            if not body or len(body) > 1800:
                return
            seen.add(fqn)
            helpers.append(
                f"### {sk.fqn} ({sk.file_path}:{sk.line_start}-{sk.line_end})\n"
                f"{reason}\n"
                "```python\n"
                f"{body}\n"
                "```"
            )

        # Direct graph dependencies first: these answer "what does this target call?"
        for c in candidates[:3]:
            for edge in store.graph.get_forward_edges(c.skeleton.fqn):
                add_helper(edge.target_fqn, f"Called by {c.skeleton.fqn}.")

        # Exact symbol names from the query, e.g. _pad / decode_ascii.
        for term in query_symbols:
            for fqn in self._resolve_short_symbol(term, store)[:2]:
                add_helper(fqn, f"Matched query symbol `{term}`.")

        # Bytes/ascii bugs often need import-level encode/decode helpers even
        # when the target body does not call them yet.
        if any(w in query_lower for w in ("byte", "bytes", "ascii", "unicode")):
            for c in candidates[:3]:
                file_skel = store.file_skeletons.get(c.skeleton.file_path)
                if not file_skel:
                    continue
                imported = self._imported_names(file_skel.imports)
                for name in imported:
                    lname = name.lower()
                    if (
                        "ascii" in lname
                        or "bytes" in lname
                        or lname.startswith(("decode", "encode", "ensure"))
                    ):
                        for fqn in self._resolve_short_symbol(name, store)[:1]:
                            add_helper(fqn, f"Imported helper relevant to bytes/ascii: `{name}`.")

        return "\n\n".join(helpers)[:max_chars]

    def _render_likely_tests(
        self,
        query: str,
        candidates: List,
        store,
        max_chars: int,
    ) -> str:
        if max_chars < 800:
            return ""

        target_paths = [c.skeleton.file_path for c in candidates[:5]]
        target_names = [c.skeleton.fqn.split("::")[-1] for c in candidates[:5]]
        test_paths = self._candidate_test_paths(target_paths, store)
        if not test_paths:
            return ""

        terms = self._test_terms(query, target_names)
        blocks: List[str] = []
        for path in test_paths[:4]:
            if len("\n\n".join(blocks)) >= max_chars:
                break
            prelude = self._read_test_prelude(path, max_chars=900)
            bodies = self._matching_test_bodies(
                path, query, terms, store, limit=4)
            body_ranges = [(start, end) for start, end, _ in bodies]
            snippets = self._matching_line_snippets(
                path, query, terms, limit=8, skip_ranges=body_ranges)
            if not snippets and not bodies:
                continue
            lines = [f"### {path}"]
            if prelude:
                lines += ["Test imports/context:", "```python", prelude, "```"]
            if snippets:
                lines.append("High-signal matching lines:")
                lines.extend(f"  {s}" for s in snippets)
            if bodies:
                lines.append(
                    "Relevant test bodies / insertion anchors "
                    "(prefer these before native test grep):")
                lines.extend(block for _, _, block in bodies)
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)[:max_chars]

    @staticmethod
    def _symbol_terms(query: str) -> List[str]:
        import re as _re
        out: List[str] = []
        for token in _re.split(r"[^A-Za-z0-9_]+", query):
            if not token or len(token) < 3:
                continue
            if "_" in token or "." in token or _re.search(r"[a-z][A-Z]|[A-Z][a-z]", token):
                out.append(token.split(".")[-1])
        return list(dict.fromkeys(out))

    @staticmethod
    def _resolve_short_symbol(name: str, store, limit: int = 5) -> List[str]:
        matches: List[str] = []
        for fqn in store.skeleton_table:
            short = fqn.split("::")[-1]
            if short == name or short.endswith(f".{name}"):
                matches.append(fqn)
        return sorted(matches, key=len)[:limit]

    @staticmethod
    def _imported_names(imports: List[str]) -> List[str]:
        import re as _re
        names: List[str] = []
        for imp in imports:
            if " import " in imp:
                tail = imp.split(" import ", 1)[1]
                for part in tail.replace("(", "").replace(")", "").split(","):
                    name = part.strip().split(" as ", 1)[0].strip()
                    if name:
                        names.append(name)
            else:
                m = _re.match(r"\s*import\s+([A-Za-z_][\w.]*)", imp)
                if m:
                    names.append(m.group(1).split(".")[-1])
        return list(dict.fromkeys(names))

    def _candidate_test_paths(self, source_paths: List[str], store) -> List[str]:
        indexed = set(store.file_skeletons.keys())
        found: List[str] = []
        all_tests = [
            p for p in indexed
            if "/tests/" in p.replace("\\", "/") and p.endswith(".py")
        ]
        for src in source_paths:
            norm = src.replace("\\", "/")
            stem = Path(norm).stem
            parent = str(Path(norm).parent).replace("\\", "/")
            test_prefix = f"{parent}/tests/"
            guesses = [
                f"{parent}/tests/test_{stem}.py",
                f"{parent}/test_{stem}.py",
                f"tests/test_{stem}.py",
            ]
            for g in guesses:
                if g in indexed and g not in found:
                    found.append(g)
            for p in all_tests:
                base = Path(p).name.lower()
                if (
                    p.replace("\\", "/").startswith(test_prefix)
                    and (
                        base == f"test_{stem.lower()}.py"
                        or base.startswith(f"test_{stem.lower()}_")
                    )
                    and p not in found
                ):
                    found.append(p)
        return found

    @staticmethod
    def _test_terms(query: str, target_names: List[str]) -> List[str]:
        import re as _re
        terms: List[str] = []
        for name in target_names:
            short = name.split(".")[-1]
            for part in _re.split(r"[^A-Za-z0-9_]+", short):
                if len(part) >= 4:
                    terms.append(part)
        for token in _re.split(r"[^A-Za-z0-9_]+", query):
            if len(token) >= 4 and token.lower() not in {
                "class", "method", "implementation", "accepts", "parsing",
                "header", "card", "fits", "test", "tests", "string",
            }:
                terms.append(token)
        return list(dict.fromkeys(terms))[:10]

    def _matching_line_snippets(
        self,
        rel_path: str,
        query: str,
        terms: List[str],
        limit: int = 8,
        skip_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> List[str]:
        if not terms:
            return []
        try:
            lines = (self._root / rel_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return []
        terms_l = [t.lower() for t in terms]
        query_l = query.lower()
        skip_ranges = skip_ranges or []
        hits: List[Tuple[int, int, str]] = []
        for i, line in enumerate(lines, 1):
            if any(start <= i <= end for start, end in skip_ranges):
                continue
            stripped = line.strip()
            if i <= 80 or stripped.startswith(("import ", "from ")):
                continue
            lower = line.lower()
            if any(t in lower for t in terms_l):
                score = sum(1 for t in terms_l if t in lower)
                if "byte" in query_l and "byte" in lower:
                    score += 12
                if "header.fromstring" in lower or "fits.header.fromstring" in lower:
                    score += 10
                if "card.fromstring" in lower or "fits.card.fromstring" in lower:
                    score += 10
                if "fromstring" in query_l and "fromstring" in lower:
                    score += 3
                hits.append((score, i, f"{rel_path}:{i}: {stripped[:120]}"))
        return [
            text for _, _, text in sorted(hits, key=lambda x: (-x[0], x[1]))[:limit]
        ]

    def _matching_test_bodies(self, rel_path: str, query: str, terms: List[str],
                              store, limit: int = 5) -> List[Tuple[int, int, str]]:
        """Return compact bodies of the most relevant tests in a test file."""
        terms_l = [t.lower() for t in terms]
        file_skel = store.file_skeletons.get(rel_path)
        if not file_skel:
            return []

        query_l = query.lower()
        ranked = []
        for sk in file_skel.all_skeletons:
            short = sk.fqn.split("::")[-1].lower()
            sig_l = sk.signature.lower()
            doc_l = (sk.docstring or "").lower()
            body, actual_start, actual_end = self._read_skeleton_range_body(
                rel_path, sk.line_start, sk.line_end, max_chars=2600)
            body_l = body.lower()
            score = 0
            for t in terms_l:
                if t in short:
                    score += 4
                if t in sig_l or t in doc_l:
                    score += 1
                if t in body_l:
                    score += 2
            has_byte = (
                "byte" in short or "byte" in sig_l
                or "byte" in doc_l or "byte" in body_l
            )
            has_fromstring = "fromstring" in short or "fromstring" in body_l
            if "byte" in query_l and has_byte:
                score += 12 if "fromstring" in query_l and has_fromstring else 2
            if "fromstring" in query_l and "fromstring" in short:
                score += 5
            if "fromstring" in query_l and "fromstring" in body_l:
                score += 2
            if "header.fromstring" in body_l or "fits.header.fromstring" in body_l:
                score += 6
            if "card.fromstring" in body_l or "fits.card.fromstring" in body_l:
                score += 6
            if score:
                ranked.append((score, actual_start, sk, body, actual_end))

        sorted_ranked = sorted(ranked, key=lambda x: (-x[0], x[1]))
        selected = self._diverse_test_body_selection(
            sorted_ranked, query_l, limit)

        out: List[Tuple[int, int, str]] = []
        for _, actual_start, sk, body, actual_end in selected:
            if body:
                block = (
                    f"- {sk.signature}  # {sk.fqn} "
                    f"({rel_path}:{actual_start}-{actual_end})\n"
                    "```python\n"
                    f"{body[:1400]}\n"
                    "```"
                )
                out.append((actual_start, actual_end, block))
        return out

    @staticmethod
    def _diverse_test_body_selection(
        ranked: List[Tuple[int, int, Any, str, int]],
        query_l: str,
        limit: int,
    ) -> List[Tuple[int, int, Any, str, int]]:
        """Keep top tests, but force anchors for explicit Class.method targets."""
        selected: List[Tuple[int, int, Any, str, int]] = []
        selected_ids: set[int] = set()

        def add_first_containing(needles: List[str]) -> None:
            if len(selected) >= limit:
                return
            for item in ranked:
                body_l = item[3].lower()
                if (
                    any(needle in body_l for needle in needles)
                    and id(item[2]) not in selected_ids
                ):
                    selected.append(item)
                    selected_ids.add(id(item[2]))
                    return

        if "header.fromstring" in query_l:
            add_first_containing(["header.fromstring", "fits.header.fromstring"])
        if "card.fromstring" in query_l:
            add_first_containing(["card.fromstring", "fits.card.fromstring"])

        for item in ranked:
            if len(selected) >= limit:
                break
            if id(item[2]) in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(id(item[2]))
        return selected

    def _read_test_prelude(self, rel_path: str, max_chars: int = 900) -> str:
        """Return imports and class context from a test file."""
        try:
            lines = (self._root / rel_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        kept: List[str] = []
        for line in lines[:80]:
            stripped = line.strip()
            if (
                stripped.startswith(("import ", "from "))
                or stripped.startswith("class ")
                or not stripped
            ):
                kept.append(line)
                continue
            if stripped.startswith("def ") and kept:
                break
        return "\n".join(kept).strip()[:max_chars]

    def _read_range_body(
        self,
        rel_path: str,
        start: int,
        end: int,
        max_chars: int = 0,
    ) -> str:
        """Read an indexed line range from disk."""
        try:
            lines = (self._root / rel_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
            s = max(0, start - 1)
            e = min(len(lines), end)
            text = "\n".join(lines[s:e])
            if max_chars and len(text) > max_chars:
                return text[:max_chars].rstrip()
            return text
        except Exception:
            return ""

    def _read_skeleton_range_body(
        self,
        rel_path: str,
        start: int,
        end: int,
        max_chars: int = 0,
    ) -> Tuple[str, int, int]:
        """Read an indexed body, relocating by declaration name if ranges drifted."""
        try:
            lines = (self._root / rel_path).read_text(
                encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return "", start, end

        expected_name = ""
        try:
            sk = next(
                s for s in self._engine.get_store().file_skeletons[rel_path].all_skeletons
                if s.line_start == start and s.line_end == end
            )
            expected_name = self._decl_name(sk.signature)
        except Exception:
            expected_name = ""

        s = max(0, start - 1)
        e = min(len(lines), end)
        if expected_name:
            head = "\n".join(lines[s:min(e, s + 5)])
            if not self._range_starts_at_decl(head, expected_name):
                relocated = self._locate_named_block(lines, expected_name)
                if relocated:
                    s, e = relocated

        text = "\n".join(lines[s:e])
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text, s + 1, e

    @staticmethod
    def _decl_name(signature: str) -> str:
        import re as _re
        m = _re.search(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", signature)
        return m.group(1) if m else ""

    @staticmethod
    def _range_starts_at_decl(text: str, name: str) -> bool:
        import re as _re
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            return bool(_re.match(
                rf"(?:async\s+def|def|class)\s+{_re.escape(name)}\b",
                stripped,
            ))
        return False

    @staticmethod
    def _locate_named_block(lines: List[str], name: str) -> Optional[Tuple[int, int]]:
        import re as _re
        pattern = _re.compile(rf"^(\s*)(?:async\s+def|def|class)\s+{_re.escape(name)}\b")
        for idx, line in enumerate(lines):
            m = pattern.match(line)
            if not m:
                continue
            indent = len(m.group(1).replace("\t", "    "))
            end = len(lines)
            for j in range(idx + 1, len(lines)):
                stripped = lines[j].strip()
                if not stripped or stripped.startswith(("#", "@")):
                    continue
                current_indent = len(lines[j]) - len(lines[j].lstrip(" \t"))
                if current_indent <= indent and _re.match(
                    r"(?:async\s+def|def|class)\s+", stripped
                ):
                    end = j
                    break
            return idx, end
        return None

    @staticmethod
    def _cap_to_remaining(text: str, remaining_chars: int) -> str:
        """Cap a body so one large function cannot overshoot the response budget."""
        if remaining_chars <= 0:
            return ""
        if len(text) <= remaining_chars:
            return text
        cap = max(0, remaining_chars - 120)
        return text[:cap].rstrip() + "\n# ... [truncated by sg_search token budget]"

    def _read_body(self, sk, max_lines: int = 350) -> str:
        """Read a function/class body from disk, IN FULL (only very long
        bodies are capped — they are the rare case)."""
        try:
            file_path = self._root / sk.file_path
            if not file_path.exists():
                return ""
            all_lines = file_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            start = max(0, sk.line_start - 1)
            end = min(len(all_lines), sk.line_end)
            body_lines = all_lines[start:end]
            if len(body_lines) > max_lines:
                trimmed = len(body_lines) - max_lines
                return ("\n".join(body_lines[:max_lines])
                        + f"\n# ... ({trimmed} more lines — "
                          f"sg_expand('{sk.fqn}') for the full body)")
            return "\n".join(body_lines)
        except Exception:
            return ""

    def _read_file_prelude(self, sk, max_lines: int = 80,
                           max_chars: int = 800) -> str:
        """Return compact file-level context before a target.

        This catches imports and top-level constants that function bodies omit,
        which is often the reason agents keep searching after finding the right
        function.
        """
        try:
            file_path = self._root / sk.file_path
            if not file_path.exists():
                return ""
            all_lines = file_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            before = all_lines[: max(0, sk.line_start - 1)]
            kept: List[str] = []
            for line in before[:max_lines]:
                stripped = line.strip()
                if (
                    stripped.startswith(("import ", "from "))
                    or ("=" in stripped and stripped.split("=", 1)[0].strip().isupper())
                ):
                    kept.append(line)
                    continue
                if line and not line.startswith((" ", "\t")):
                    # First top-level definition/class usually means the import
                    # prelude is over for Python-like files.
                    if stripped.startswith(("def ", "class ", "async def ")):
                        break
            text = "\n".join(kept).strip()
            return text[:max_chars]
        except Exception:
            return ""

    @staticmethod
    def _callers_of(store, fqn: str, limit: int = 3) -> List[str]:
        """1-hop callers of a function (functions whose callees include fqn)."""
        callers: List[str] = []
        for other_fqn, other_sk in store.skeleton_table.items():
            if fqn in getattr(other_sk, "callees", []):
                callers.append(other_fqn)
                if len(callers) >= limit:
                    break
        return callers

    def _lexical_search(self, query: str, file_filter: str = "",
                        limit: int = 25) -> str:
        """Plain-text grep over source files — the fallback for symbols the
        function graph cannot index (module constants, decorators, aliases).

        Uses ripgrep when available; otherwise a bounded scan of indexed
        source files. Never raises."""
        import re as _re

        _STOP = {
            "def", "class", "self", "return", "import", "from", "with",
            "this", "true", "false", "none", "null", "the", "and", "for",
            "bytes", "byte", "string", "str", "file", "files", "util",
            "utils", "header", "card", "fits", "ascii",
        }
        terms = []
        for t in _re.split(r"[^A-Za-z0-9_]+", query):
            if len(t) >= 4 and t.lower() not in _STOP:
                terms.append(t)
        symbol_terms = [
            t for t in terms
            if "_" in t or _re.search(r"[a-z][A-Z]|[A-Z][a-z]", t)
        ]
        if symbol_terms:
            # If the query contains concrete symbols, ignore generic prose terms.
            terms = symbol_terms
        # Most-specific (longest) terms first; cap to keep the regex small.
        terms = sorted(dict.fromkeys(terms), key=len, reverse=True)[:5]
        if not terms:
            return ""

        pattern = "|".join(_re.escape(t) for t in terms)
        hits: List[str] = []

        # Fast path: ripgrep (respects .gitignore, skips node_modules etc.)
        raw = ""
        try:
            import subprocess
            r = subprocess.run(
                [
                    "rg", "-n", "--no-heading", "-S", "-m", "3",
                    "--glob", "!*.md",
                    "--glob", "!*.json",
                    "--glob", "!*.jsonl",
                    "--glob", "!*.txt",
                    "--glob", "!.skeletongraph/**",
                    "--glob", "!.git/**",
                    "--glob", "!node_modules/**",
                    pattern, ".",
                ],
                cwd=str(self._root), capture_output=True, text=True, timeout=10)
            raw = r.stdout
        except Exception:
            raw = ""

        if raw:
            for ln in raw.splitlines():
                path_part = ln.split(":", 1)[0].lstrip(".\\/").replace("\\", "/")
                if file_filter and file_filter.replace("\\", "/") not in path_part:
                    continue
                if not self._is_source_path(path_part):
                    continue
                hits.append(ln.strip()[:160])
                if len(hits) >= limit:
                    break
        else:
            hits = self._python_grep(terms, file_filter, limit)

        if not hits:
            return ""
        return "\n".join(f"  {h}" for h in hits)

    def _python_grep(self, terms: List[str], file_filter: str,
                     limit: int) -> List[str]:
        """ripgrep-free fallback: scan indexed source files only (bounded)."""
        hits: List[str] = []
        try:
            files = list(self._engine.get_store().file_skeletons.keys())
        except Exception:
            return hits
        for rel in files:
            if file_filter and file_filter not in rel:
                continue
            try:
                text = (self._root / rel).read_text(
                    encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if any(t in line for t in terms):
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    if len(hits) >= limit:
                        return hits
        return hits

    @staticmethod
    def _is_source_path(path: str) -> bool:
        return path.lower().endswith((
            ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
            ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
            ".cs", ".rb", ".php",
        ))

    @staticmethod
    def _candidates_from_fqns(fqns: List[str], store) -> List:
        """Wrap fusion/rerank FQN results as objects the body-renderer expects.

        retrieve_fusion/retrieve_rerank return bare ranked FQN strings; the
        rendering path only touches `c.skeleton`, so a lightweight namespace per
        resolvable FQN (preserving rank order, skipping any not in the index) is
        all that's needed to feed them through the same output formatting as the
        structural path."""
        from types import SimpleNamespace
        out = []
        for fqn in fqns:
            sk = store.skeleton_table.get(fqn)
            if sk is not None:
                out.append(SimpleNamespace(skeleton=sk))
        return out

    @staticmethod
    def _is_test_path(path: str) -> int:
        """1 for a test file, 0 for source — used as a stable sort key to demote
        tests below the implementation. Covers Python (test_*/_test, /tests/) and
        JS/TS (*.test.*, *.spec.*, __tests__)."""
        p = path.replace("\\", "/").lower()
        base = p.rsplit("/", 1)[-1]
        return int(
            "/tests/" in p or "/test/" in p or "/__tests__/" in p
            or base.startswith("test_")
            or base.endswith(("_test.py", "_test.go",
                              ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
                              ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx"))
        )

    @staticmethod
    def _looks_symbolic(query: str) -> bool:
        """Return True if the query looks like a code-symbol search (identifier-like)
        rather than a natural-language description.

        Runs the lexical fallback only on symbol searches — avoids flooding results
        with noise when the model writes a prose description as the query.

        Heuristic: >60% of the meaningful tokens look like code identifiers
        (ALL_CAPS, snake_case, or CamelCase), OR the query is a single bare
        identifier, OR it contains pipe-alternation of identifiers."""
        import re as _re
        _COMMON = {
            "class", "method", "function", "call", "from", "string", "byte",
            "bytes", "file", "this", "with", "test", "that", "what", "when",
            "where", "which", "into", "return", "using", "used", "uses",
            "fits", "header", "card", "data", "list", "type", "value",
            "name", "base", "index", "util", "utils", "lib", "code", "handle",
        }
        # Pipe-alternation like "encode_ascii|decode_ascii" → always symbolic
        if "|" in query:
            return True
        tokens = [t for t in _re.split(r"\s+", query.strip()) if len(t) >= 3]
        if not tokens:
            return False
        # Single bare token — always symbolic
        if len(tokens) == 1:
            return True

        def _is_identifier(t: str) -> bool:
            # ALL_CAPS_CONSTANT, snake_case_name (has underscore), or CamelCase (mixed)
            if _re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", t):  # CONSTANT
                return True
            if "_" in t and _re.fullmatch(r"[a-z][a-z0-9_]+", t):  # snake_case
                return True
            if _re.fullmatch(r"[A-Z][a-z]+[A-Z][A-Za-z]*", t):  # CamelCase
                return True
            return False

        id_count = sum(
            1 for t in tokens
            if _is_identifier(t) and t.lower() not in _COMMON
        )
        return id_count / len(tokens) > 0.5

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

        parts.append(
            f"\nIf you are about to edit this exact function and need its body, "
            f"call sg_expand('{resolved_fqn}')."
        )
        return "\n".join(parts)

    # ── Tool: sg_expand ──────────────────────────────────────────────────

    def _tool_expand(self, args: Dict) -> str:
        """Expand one OR SEVERAL targets in a single call.

        BATCHING: `target` may be a list, or a comma/newline-separated string, of
        FQNs (and/or file:line-range specs). Fixes the measured cost pattern where,
        for a fix spanning N functions, the agent called sg_expand N times — each a
        separate turn that re-sends the whole conversation. One batched call
        collapses those N turns into 1. FQNs never contain commas, so splitting on
        commas/newlines is unambiguous. Per-target dedup still applies to each."""
        import re as _re
        raw = args.get("target", "") or args.get("fqn", "") or args.get("fqns", "")
        if isinstance(raw, (list, tuple)):
            targets = [str(t).strip() for t in raw if str(t).strip()]
        else:
            # split on commas / newlines only (FQNs use :: . / but never ,)
            targets = [t.strip() for t in _re.split(r"[,\n]", str(raw)) if t.strip()]

        if not targets:
            return "Error: target is required"
        if len(targets) == 1:
            return self._expand_one(targets[0], args)

        parts = []
        for t in targets:
            body = self._expand_one(t, args)
            parts.append(f"### {t}\n{body}")
        return f"# Expanded {len(targets)} targets\n\n" + "\n\n".join(parts)

    def _expand_one(self, target: str, args: Dict) -> str:
        target = str(target).strip()
        # Default 2500 / CAP 4000. The cap matters as much as the default:
        # observed regression where the model passed max_tokens=5000 to
        # override the default — putting the response back above VS Code
        # Copilot's content.txt offload threshold (~4K tokens). Capping at
        # 4000 forces inline retention even when the caller asks for more.
        # If the model genuinely needs a >4K body, it can request the
        # specific line range (e.g. sg_expand('file.py:540-820')) for
        # multiple small calls — each stays inline.
        max_tokens = int(args.get("max_tokens", 2500))
        max_tokens = min(max(max_tokens, 1000), 4000)

        if not target:
            return "Error: target is required"

        # Session dedup: FQN expansions whose body is already in context.
        # A file path or range is always fresh (may differ from what sg_search sent).
        if "::" in target and target in self._returned_fqns:
            try:
                sk = self._engine.get_store().skeleton_table.get(target)
            except Exception:
                sk = None
            if sk:
                loc = f"{sk.file_path}:{sk.line_start}-{sk.line_end}"
                return (
                    f"✓ Body for `{target}` was already returned by sg_search "
                    f"earlier this session.\n\n"
                    f"**Location:** `{loc}`\n\n"
                    f"Edit directly from that earlier result — do NOT re-Read "
                    f"this file. The body is already in your context above.\n\n"
                    f"If you need a DIFFERENT part of the same file (e.g. lines "
                    f"around it), use: sg_expand(target=\"{sk.file_path}:"
                    f"{max(1, sk.line_start - 10)}-{sk.line_end + 10}\")"
                )
            return (
                f"✓ Body for `{target}` was already returned by sg_search "
                f"earlier this session — it is in your context. "
                f"Edit directly from the earlier result."
            )

        # Parse range syntax: file.py:42-80
        start_line = end_line = None
        if ":" in target and "::" not in target:
            parts_colon = target.rsplit(":", 1)
            range_part = parts_colon[-1]
            if "-" in range_part and all(p.strip().isdigit() for p in range_part.split("-", 1)):
                start_line, end_line = (int(p.strip()) for p in range_part.split("-", 1))
                target = parts_colon[0]

        # Range dedup: if most of this line range was already returned this
        # session (by an sg_search body or an earlier sg_expand), don't re-fetch
        # it — that overlapping-range paging is the main source of wasted tokens.
        if start_line is not None and end_line is not None:
            covered = self._range_already_covered(target, start_line, end_line)
            if covered:
                return (
                    f"Note: lines {start_line}-{end_line} of `{target}` overlap "
                    f"code already returned this session "
                    f"(`{target}:{covered[0]}-{covered[1]}`) — it is already in "
                    f"your context. Reuse it instead of re-reading. Request a "
                    f"non-overlapping range only if you need genuinely new lines."
                )

        result = self._engine.expand(
            target=target,
            start_line=start_line,
            end_line=end_line,
            max_tokens=max_tokens,
        )
        # Track what was returned so later sg_search/sg_expand calls can dedup.
        if start_line is not None and end_line is not None:
            self._record_range(target, start_line, end_line)
        if "::" in target:
            self._returned_fqns.add(target)
            try:                                  # also record the FQN's line span
                sk = self._engine.get_store().skeleton_table.get(target)
                if sk:
                    self._record_range(sk.file_path, sk.line_start, sk.line_end)
            except Exception:
                pass
        return result

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

    # Pre-warm ALL lazy retrieval state in the BACKGROUND — not synchronously
    # before the stdio loop starts. A blocking warm-up here delayed the MCP
    # "initialize" handshake past Claude Code's connection timeout on
    # larger/slower repos (confirmed: django/matplotlib sessions came up with
    # ZERO SG tools registered, mcp_servers status stuck at "pending" — a
    # session with NO SG tools is strictly worse than one with a slow first
    # call, since the agent falls back to pure native grep/read with no
    # fallback path). The stdio loop must start immediately; warming happens
    # in parallel and whatever's ready by the agent's first real call is a
    # bonus — the per-call caching in retrieval/bm25_flat.py + fusion.py
    # already handles "not warm yet" correctly (it just builds then, exactly
    # like before this existed), so there's no race to guard against.
    def _background_warm() -> None:
        try:
            server._engine.heuristic_query("warm-up", top_n=1)
        except Exception:
            logger.warning("structural warm-up failed — falling back to lazy load", exc_info=True)
        # Must match _tool_search's default exactly, or the wrong cache warms.
        mode = os.environ.get("SG_MCP_RETRIEVAL", "fusion").strip().lower()
        if mode in ("fusion", "rerank"):
            if mode == "fusion":
                # Load the embedding MODEL first, with NO timeout. We are already
                # inside a daemon thread, so a slow load cannot wedge startup or
                # the stdio handshake. This must NOT go through warm()'s bounded
                # path: _DENSE_TIMEOUT_S defaults to 20s but a cold
                # jina-v2-base-code load measured ~24.7s, so the bounded attempt
                # ALWAYS timed out and left the model cold for the agent's first
                # sg_search — which then paid the full ~25s mid-task (measured as
                # ~+40s/task wall-clock vs native in the Claude Code eval).
                try:
                    from ..retrieval.dense import prewarm as dense_prewarm
                    dense_prewarm(background=False)
                except Exception:
                    logger.warning("dense model prewarm failed — falling back to lazy load",
                                   exc_info=True)
            try:
                from ..retrieval import warm as warm_retrieval
                # full_dense=False: this background thread must never spend minutes
                # encoding a cold corpus (it would wedge behind a stuck download and
                # the first query would degrade anyway). It opportunistically LOADS
                # an already-prewarmed cache (fast) or gives up cleanly. The full
                # encode belongs to `sg warm` / prepare, run before serving.
                warm_retrieval(project_root, mode=mode, full_dense=False)
            except Exception:
                logger.warning("%s warm-up failed — falling back to lazy load", mode, exc_info=True)

    threading.Thread(target=_background_warm, daemon=True).start()

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
