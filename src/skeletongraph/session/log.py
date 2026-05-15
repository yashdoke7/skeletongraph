"""
Session log accessor — thin wrapper over conversation_logger.

Used by MCP sg_log tool and CLI `sg log` command to expose
recent session turns without exposing raw JSONL internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LogEntry:
    """One turn from the session log."""
    turn_index: int
    user_prompt: str
    files_touched: List[str]
    summary: str
    agent_action: str


def read_log(sg_dir: Path, session_id: Optional[str] = None, last_n: int = 10) -> List[LogEntry]:
    """Read recent session log entries.

    Args:
        sg_dir: .skeletongraph directory path.
        session_id: Specific session ID to read. If None, reads the latest.
        last_n: Maximum entries to return (most recent first).

    Returns:
        List of LogEntry, most-recent first.
    """
    sessions_dir = sg_dir / "sessions"
    if not sessions_dir.exists():
        return []

    if session_id:
        jsonl_path = sessions_dir / f"{session_id}.jsonl"
    else:
        # Find most-recently modified session file
        files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return []
        jsonl_path = files[0]

    if not jsonl_path.exists():
        return []

    lines = jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries = []
    for i, line in enumerate(reversed(lines)):
        if i >= last_n:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries.append(LogEntry(
            turn_index=data.get("turn_index", 0),
            user_prompt=data.get("user_prompt", ""),
            files_touched=data.get("files_modified", []),
            summary=data.get("summary", ""),
            agent_action=data.get("agent_action", ""),
        ))
    return entries


def append_log(
    sg_dir: Path,
    session_id: str,
    user_prompt: str,
    files_touched: Optional[List[str]] = None,
    summary: str = "",
    agent_action: str = "",
    turn_index: int = 0,
) -> None:
    """Append one turn entry to the session log JSONL file."""
    sessions_dir = sg_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = sessions_dir / f"{session_id}.jsonl"

    entry = {
        "turn_index": turn_index,
        "user_prompt": user_prompt,
        "files_modified": files_touched or [],
        "summary": summary,
        "agent_action": agent_action,
    }
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def format_log_digest(entries: List[LogEntry], max_turns: int = 5) -> str:
    """Compact multi-turn digest for sg_overview Zone 1 injection."""
    if not entries:
        return ""
    lines = ["## Recent turns"]
    for e in entries[:max_turns]:
        prompt_short = e.user_prompt[:80].replace("\n", " ")
        files_str = ", ".join(e.files_touched[:4]) if e.files_touched else "—"
        lines.append(f"- Turn {e.turn_index}: {prompt_short!r}  →  {files_str}")
        if e.summary:
            lines.append(f"  {e.summary[:120]}")
    return "\n".join(lines)
