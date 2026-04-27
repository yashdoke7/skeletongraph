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
    """Parse Cursor's SQLite database for chat history."""
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Cursor stores chat data in the ItemTable with keys like
        # 'workbench.panel.chat*' or 'interactive.sessions'
        cursor.execute(
            "SELECT key, value FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%composer%'"
        )

        turn_id = 0
        for key, value in cursor.fetchall():
            if not value:
                continue

            try:
                data = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue

            # Navigate the nested structure for chat messages
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

                    # Extract file mentions from response
                    for file_match in re.finditer(r'`([^\s`]+\.(?:py|ts|js|tsx|jsx))`', content):
                        local = project_root / file_match.group(1)
                        if local.exists():
                            tool_calls.append(ToolCall(
                                "view_file", file_match.group(1),
                                measure_file_tokens(local, cap_lines=800)
                            ))

                elif role in ("user", "human") and not task_prompt:
                    task_prompt = content.strip()[:200]

        conn.close()

    except (sqlite3.Error, OSError) as e:
        pass  # Gracefully handle DB access errors

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
