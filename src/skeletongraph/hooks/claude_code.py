"""
Claude Code hook handlers — P3.

Four hooks used:
  SessionStart       → start new session JSONL, inject "use SG" system message
  UserPromptSubmit   → heuristic sg_overview injected as additionalContext (no LLM call)
  PostToolUse        → append to session log JSONL
  FileChanged        → background incremental re-index

All hooks:
  - Read event data from stdin as JSON
  - Write response to stdout as JSON (where applicable)
  - Exit 0 on any internal error (never block the agent)
  - Write errors to .skeletongraph/last_hook.log

Entry point: `sg hook <event_name> --path <project_root>`
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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_USE_SG_SYSTEM_MSG = (
    "SkeletonGraph (SG) is active for this project. "
    "ALWAYS call sg_overview at session start for the project briefing. "
    "Use sg_search as a task-context assembler, not grep: ask for the whole task once; "
    "it returns likely edit targets, helpers, graph neighbors, and likely tests. "
    "Its results are complete and self-contained — each body is the exact current "
    "source with its file:line range, so edit directly from them and do NOT re-grep "
    "or re-read code sg_search already returned. "
    "Use sg_get/sg_expand only for exact follow-up FQNs. "
    "sg_constraint to view project rules before proposing changes. "
    "MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log."
)


# ── Public hook handlers ─────────────────────────────────────────────────


def hook_session_start(project_root: Path, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """SessionStart: create a fresh session JSONL, inject 'use SG' system message.

    Returns {"systemMessage": "..."} for Claude Code injection.
    """
    sg_dir = project_root / ".skeletongraph"
    try:
        sg_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(uuid.uuid4())[:8]
        # Persist current session id so PostToolUse can find it
        (sg_dir / "current_session.txt").write_text(session_id, encoding="utf-8")
        _write_hook_log(sg_dir, "session_start", f"session_id={session_id}")
    except Exception as e:
        _write_hook_log(sg_dir, "session_start", f"ERROR: {e}")

    return {"systemMessage": _USE_SG_SYSTEM_MSG}


def hook_user_prompt_submit(project_root: Path, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """UserPromptSubmit: inject sg_overview as additionalContext.

    Runs heuristic_query (zero LLM cost) for the prompt, assembles a quick
    overview (constraints + top functions + session digest), returns it as
    additionalContext so it lands in Zone 1 of the main LLM context.

    Returns {"additionalContext": "<overview text>"}.
    """
    sg_dir = project_root / ".skeletongraph"
    prompt = event_data.get("prompt", "")

    try:
        from ..engine import SGEngine
        from ..config import load_config

        cfg = load_config(project_root)
        engine = SGEngine(project_root, cfg)

        # Build overview text: constraints + top PageRank + query-specific hits
        parts: list[str] = [_USE_SG_SYSTEM_MSG, ""]

        # Constraints (Zone 1) — always included, compact
        cs_file = sg_dir / "constraints.md"
        if cs_file.exists():
            cs_text = cs_file.read_text(encoding="utf-8", errors="replace").strip()
            if cs_text:
                # Cap constraints to ~600 tokens to leave room for other context
                if len(cs_text) > 2400:
                    cs_text = cs_text[:2400].rstrip() + "\n... (truncated)"
                parts.append(f"## Constraints\n{cs_text}")

        # Smart-routed MD sections (architecture / project / decisions)
        # — only included when prompt classification suggests they help
        try:
            from ..assembly.context_routing import route_context_sections, format_routed_sections
            routed = route_context_sections(prompt, sg_dir)
            routed_text = format_routed_sections(routed)
            if routed_text:
                parts.append(routed_text)
        except Exception:
            pass

        # Session digest — always included, short
        from ..session.log import read_log, format_log_digest
        entries = read_log(sg_dir, last_n=5)
        digest = format_log_digest(entries, max_turns=5)
        if digest:
            parts.append(digest)

        # Query-relevant functions with summaries (PUSH retrieval). Skipped by
        # default: when the MCP server is installed the agent calls sg_search
        # itself, so running heuristic_query here too would double the retrieval
        # (and the tokens). Enable cfg.hook_push_retrieval only for hook-only
        # installs with no MCP path. See docs/RESEARCH.md §5d.
        if prompt.strip() and getattr(cfg, "hook_push_retrieval", False):
            try:
                result = engine.heuristic_query(prompt.strip(), top_n=8)
                candidates = result.candidates if hasattr(result, "candidates") else []
                if candidates:
                    store = engine.get_store()
                    lines = [f"## Relevant functions (confidence={getattr(result, 'confidence', 'MEDIUM')})"]
                    for c in candidates[:8]:
                        sk = c.skeleton
                        summary = store.summaries.get(sk.fqn) or ""
                        if not summary and sk.docstring:
                            summary = sk.docstring.splitlines()[0].strip()
                        entry = f"  {sk.signature}  # {sk.fqn}"
                        if summary:
                            entry += f"  — {summary[:80]}"
                        lines.append(entry)
                    parts.append("\n".join(lines))
            except Exception:
                pass

        context_text = "\n\n".join(p for p in parts if p)
        _write_hook_log(sg_dir, "user_prompt_submit", f"injected {len(context_text)} chars")
        return {"additionalContext": context_text}

    except Exception as e:
        _write_hook_log(sg_dir, "user_prompt_submit", f"ERROR: {e}")
        # Return minimal fallback — still useful even if engine failed
        return {"additionalContext": _USE_SG_SYSTEM_MSG}


def hook_post_tool_use(project_root: Path, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """PostToolUse: append a session log entry.

    Reads tool_name / tool_input / tool_response from event_data.
    Detects modified files from Edit/Write/Bash tool inputs.
    Returns {} (no injection needed).
    """
    sg_dir = project_root / ".skeletongraph"
    try:
        session_id = _read_current_session(sg_dir)
        if not session_id:
            return {}

        tool_name = event_data.get("tool_name", "")
        tool_input = event_data.get("tool_input", {})
        tool_response = event_data.get("tool_response", {})

        # Detect modified files
        files_touched = _extract_modified_files(tool_name, tool_input, tool_response)

        # Build a short summary
        summary = f"{tool_name}"
        if files_touched:
            summary += f" → {', '.join(files_touched[:3])}"

        from ..session.log import append_log
        # Read turn index from last entry
        from ..session.log import read_log
        existing = read_log(sg_dir, session_id=session_id, last_n=1)
        turn_index = (existing[0].turn_index + 1) if existing else 1

        append_log(
            sg_dir=sg_dir,
            session_id=session_id,
            user_prompt="",          # no user prompt available here
            files_touched=files_touched,
            summary=summary,
            agent_action=tool_name,
            turn_index=turn_index,
        )
        _write_hook_log(sg_dir, "post_tool_use", f"logged turn {turn_index}: {summary}")
    except Exception as e:
        _write_hook_log(sg_dir, "post_tool_use", f"ERROR: {e}")

    # ── Background summary queue drain (best-effort) ─────────────────────
    try:
        from ..config import load_config
        from ..summary.queue import drain_queue_background, queue_size
        cfg = load_config(project_root)
        if getattr(cfg, "summary_queue_enabled", True):
            pending = queue_size(sg_dir)
            if pending > 0:
                drain_queue_background(project_root, cfg)
                _write_hook_log(sg_dir, "post_tool_use", f"draining {pending} queued summaries")
    except Exception:
        pass

    return {}


def hook_file_changed(project_root: Path, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """FileChanged: background incremental re-index for the changed file.

    Returns {} immediately; re-index runs in a daemon thread.
    """
    sg_dir = project_root / ".skeletongraph"
    file_path = event_data.get("file", "")

    def _run_incremental():
        try:
            from ..build import update_index
            update_index(project_root)
            _write_hook_log(sg_dir, "file_changed", f"re-indexed after change to {file_path}")
        except Exception as e:
            _write_hook_log(sg_dir, "file_changed", f"ERROR re-indexing: {e}")

    if file_path and sg_dir.exists():
        t = threading.Thread(target=_run_incremental, daemon=True)
        t.start()

    return {}


# ── Internal helpers ──────────────────────────────────────────────────────


def _read_current_session(sg_dir: Path) -> Optional[str]:
    """Read the active session ID from disk."""
    sid_file = sg_dir / "current_session.txt"
    if sid_file.exists():
        return sid_file.read_text(encoding="utf-8").strip() or None
    return None


def _write_hook_log(sg_dir: Path, event: str, message: str) -> None:
    """Append one line to last_hook.log (best-effort, never raises)."""
    try:
        sg_dir.mkdir(parents=True, exist_ok=True)
        log_path = sg_dir / "last_hook.log"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {event}: {message}\n")
    except Exception:
        pass


def _extract_modified_files(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_response: Dict[str, Any],
) -> list[str]:
    """Heuristically extract file paths from tool call data."""
    files: list[str] = []

    tool_lower = tool_name.lower()

    # Edit / Write tools — file_path is an explicit field
    if any(k in tool_lower for k in ("edit", "write", "create")):
        for key in ("file_path", "path", "filename"):
            val = tool_input.get(key, "")
            if val and isinstance(val, str):
                files.append(val)
                break

    # Bash tool — scan command for file-like arguments
    if "bash" in tool_lower or "shell" in tool_lower or "run" in tool_lower:
        import re
        cmd = str(tool_input.get("command", "") or tool_input.get("cmd", ""))
        # Match common source file patterns
        found = re.findall(
            r"[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cs|rb|php|md|json|yaml|toml)",
            cmd,
        )
        files.extend(found[:4])  # cap at 4

    return list(dict.fromkeys(files))  # deduplicate preserving order
