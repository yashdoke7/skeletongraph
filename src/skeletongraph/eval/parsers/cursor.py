"""
Cursor IDE session parser.

Cursor stores chat history in SQLite databases at:
  %APPDATA%/Cursor/User/workspaceStorage/<hash>/state.vscdb

Also supports manual text exports if the user copies from the UI.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import measure_file_tokens, measure_text_tokens


def discover_cursor_sessions() -> list[Path]:
    """Find Cursor workspace SQLite databases, newest first."""
    base = Path.home() / "AppData" / "Roaming" / "Cursor" / "User" / "workspaceStorage"
    if not base.exists():
        return []

    dbs = []
    for ws_dir in base.iterdir():
        if ws_dir.is_dir():
            db = ws_dir / "state.vscdb"
            if db.exists():
                dbs.append(db)

    dbs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dbs


def parse_cursor_session(
    session_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse a Cursor session.

    Supports:
    - SQLite database (state.vscdb)
    - Text export (manual copy from UI)
    """
    if session_path.suffix in (".vscdb", ".db", ".sqlite"):
        return _parse_sqlite(session_path, project_root, task_prompt, mode, project_name)
    else:
        return _parse_text(session_path, project_root, task_prompt, mode, project_name)


def _parse_sqlite(
    db_path: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse Cursor's SQLite database for chat history AND tool calls.

    Cursor stores data in state.vscdb under these key patterns:
      - 'workbench.panel.chat*'  -> message text
      - 'aidevent*'              -> tool calls (file reads, searches)
      - 'interactive.sessions'   -> session list

    Run this to discover keys in your own Cursor DB:
      sqlite3 state.vscdb ".tables"
      sqlite3 state.vscdb "SELECT DISTINCT key FROM ItemTable LIMIT 50"
    """
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor_db = conn.cursor()

        # ── Step 1: Get chat messages ──────────────────────────────
        cursor_db.execute(
            "SELECT key, value FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%composer%'"
        )

        turn_id = 0
        for key, value in cursor_db.fetchall():
            if not value:
                continue

            try:
                data = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue

            messages = _extract_messages(data)
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", msg.get("text", ""))

                if not content or not isinstance(content, str):
                    continue

                if role in ("assistant", "model"):
                    turn_id += 1
                    agent_responses.append(AgentResponse(
                        turn_id, content, measure_text_tokens(content)
                    ))
                elif role in ("user", "human") and not task_prompt:
                    task_prompt = content.strip()[:200]

        # ── Step 2: Get tool call events ───────────────────────────
        # Cursor logs tool invocations separately from chat messages.
        # The key name varies by Cursor version — try multiple patterns.
        TOOL_KEY_PATTERNS = [
            "aidevent%",
            "%toolcall%",
            "%tool_result%",
            "%composer.tool%",
        ]
        for pattern in TOOL_KEY_PATTERNS:
            cursor_db.execute(
                f"SELECT key, value FROM ItemTable WHERE key LIKE '{pattern}'"
            )
            rows = cursor_db.fetchall()
            if not rows:
                continue

            for key, value in rows:
                if not value:
                    continue
                try:
                    events = json.loads(value)
                    if not isinstance(events, list):
                        events = [events]
                except (json.JSONDecodeError, TypeError):
                    continue

                for event in events:
                    tool_name = event.get("toolName", event.get("name", ""))
                    tool_input = event.get("input", event.get("args", {}))
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except Exception:
                            tool_input = {}

                    target = tool_input.get("path", tool_input.get("file_path",
                             tool_input.get("query", tool_input.get("pattern", ""))))

                    if any(kw in tool_name.lower() for kw in ("read", "view", "file", "open")):
                        local = project_root / target if target else Path("")
                        tokens = measure_file_tokens(local, cap_lines=800)
                        tool_calls.append(ToolCall("view_file", target or "", tokens))
                    elif any(kw in tool_name.lower() for kw in ("search", "grep", "find", "codebase")):
                        tool_calls.append(ToolCall("grep_search", target or "", 100))
                    elif any(kw in tool_name.lower() for kw in ("terminal", "run", "exec", "bash")):
                        tool_calls.append(ToolCall("run_command", target or "", 50))
                    elif tool_name:
                        tool_calls.append(ToolCall(tool_name, target or "", 50))
            break  # stop at first pattern that returned data

        conn.close()

    except (sqlite3.Error, OSError):
        pass

    return AgentTrace(
        agent="cursor", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )


def _parse_text(
    text_path: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse a manually copied text export from Cursor."""
    content = text_path.read_text(encoding="utf-8", errors="replace")
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    # Look for file references
    for match in re.finditer(r'(?:Read|Viewing|Opened?)\s+([^\s\n]+\.(?:py|ts|js))', content):
        local = project_root / match.group(1)
        tool_calls.append(ToolCall(
            "view_file", match.group(1),
            measure_file_tokens(local, cap_lines=800)
        ))

    # Look for search actions
    for match in re.finditer(r'(?:Search|Grep|Find)(?:ed|ing)?\s', content):
        tool_calls.append(ToolCall("grep_search", "", 100))

    return AgentTrace(
        agent="cursor", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )


def _extract_messages(data) -> list[dict]:
    """Recursively extract messages from Cursor's nested JSON structure."""
    messages = []

    if isinstance(data, dict):
        if "role" in data and ("content" in data or "text" in data):
            messages.append(data)
        for v in data.values():
            messages.extend(_extract_messages(v))
    elif isinstance(data, list):
        for item in data:
            messages.extend(_extract_messages(item))

    return messages
