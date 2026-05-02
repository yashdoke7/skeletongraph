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
from typing import Any
from urllib.parse import unquote

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import (
    measure_file_tokens, measure_text_tokens,
    measure_grep_output_tokens, measure_directory_listing_tokens,
)


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
    selected_composer_ids: list[str] = []

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

            if isinstance(data, dict):
                selected_composer_ids.extend(
                    cid for cid in data.get("selectedComposerIds", [])
                    if isinstance(cid, str)
                )

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
                        # Measure actual result content if available
                        result_content = event.get("result", event.get("output", ""))
                        if result_content and isinstance(result_content, str) and len(result_content) > 10:
                            out_tokens = measure_text_tokens(result_content)
                        else:
                            out_tokens = measure_grep_output_tokens()
                        tool_calls.append(ToolCall("grep_search", target or "", out_tokens))
                    elif any(kw in tool_name.lower() for kw in ("terminal", "run", "exec", "bash")):
                        result_content = event.get("result", event.get("output", ""))
                        if result_content and isinstance(result_content, str) and len(result_content) > 10:
                            out_tokens = measure_text_tokens(result_content)
                        else:
                            out_tokens = 50
                        tool_calls.append(ToolCall("run_command", target or "", out_tokens))
                    elif any(kw in tool_name.lower() for kw in ("write", "edit", "patch", "insert")):
                        tool_calls.append(ToolCall("edit_file", target or "", 0))
                    elif any(kw in tool_name.lower() for kw in ("list", "dir", "ls")):
                        tool_calls.append(ToolCall("list_dir", target or "", measure_directory_listing_tokens()))
                    elif tool_name:
                        tool_calls.append(ToolCall(tool_name, target or "", 50))
            break  # stop at first pattern that returned data

        conn.close()

    except (sqlite3.Error, OSError):
        pass

    if not tool_calls and not agent_responses:
        richer_trace = _parse_current_cursor_global_storage(
            db_path, project_root, task_prompt, mode, project_name, selected_composer_ids
        )
        if richer_trace is not None:
            return richer_trace

    return AgentTrace(
        agent="cursor", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )


def _parse_current_cursor_global_storage(
    db_path: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
    composer_ids: list[str],
) -> AgentTrace | None:
    """Parse Cursor's current composer storage schema.

    Recent Cursor builds keep only composer IDs in workspaceStorage and put
    the actual bubbles/tool outputs in User/globalStorage/state.vscdb.
    """
    candidate_dbs: list[Path] = []
    if "globalStorage" in db_path.parts:
        candidate_dbs.append(db_path)
    if "workspaceStorage" in db_path.parts:
        try:
            user_dir = db_path.parents[2]
            candidate_dbs.append(user_dir / "globalStorage" / "state.vscdb")
        except IndexError:
            pass

    for global_db in candidate_dbs:
        if not global_db.exists():
            continue
        trace = _parse_global_composers(
            global_db, project_root, task_prompt, mode, project_name, composer_ids
        )
        if trace is not None and (trace.tool_calls or trace.agent_responses):
            return trace
    return None


def _parse_global_composers(
    global_db: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
    composer_ids: list[str],
) -> AgentTrace | None:
    try:
        conn = sqlite3.connect(str(global_db))
        cursor_db = conn.cursor()
        if not composer_ids:
            rows = cursor_db.execute(
                "SELECT key FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
            ).fetchall()
            composer_ids = [row[0].split(":", 1)[1] for row in rows if ":" in row[0]]

        best_trace: AgentTrace | None = None
        for composer_id in composer_ids:
            trace = _parse_one_composer(cursor_db, composer_id, project_root, task_prompt, mode, project_name)
            if trace is None:
                continue
            if best_trace is None or trace.tool_call_count + len(trace.agent_responses) > (
                best_trace.tool_call_count + len(best_trace.agent_responses)
            ):
                best_trace = trace
        conn.close()
        return best_trace
    except (sqlite3.Error, OSError):
        return None


def _parse_one_composer(
    cursor_db: sqlite3.Cursor,
    composer_id: str,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace | None:
    row = cursor_db.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{composer_id}",),
    ).fetchone()
    if not row:
        return None

    composer = _json_loads_maybe(row[0])
    if not isinstance(composer, dict):
        return None

    headers = composer.get("fullConversationHeadersOnly", [])
    if not isinstance(headers, list):
        return None

    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []
    files_modified: list[str] = []
    turn_id = 0

    for header in headers:
        if not isinstance(header, dict):
            continue
        bubble_id = header.get("bubbleId")
        if not isinstance(bubble_id, str):
            continue

        bubble_row = cursor_db.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"bubbleId:{composer_id}:{bubble_id}",),
        ).fetchone()
        if not bubble_row:
            continue

        bubble = _json_loads_maybe(bubble_row[0])
        if not isinstance(bubble, dict):
            continue

        text = bubble.get("text", "")
        if isinstance(text, str) and text.strip() and bubble.get("type") == 2:
            turn_id += 1
            agent_responses.append(AgentResponse(turn_id, text, measure_text_tokens(text)))

        if bubble.get("type") == 1 and not task_prompt:
            user_text = _extract_rich_text(bubble.get("richText", ""))
            if user_text:
                task_prompt = user_text.strip()[:200]

        tool_data = bubble.get("toolFormerData")
        if isinstance(tool_data, dict):
            call = _tool_call_from_cursor_tool(tool_data, project_root)
            if call is not None:
                tool_calls.append(call)
                if call.tool_type == "edit_file" and call.target:
                    files_modified.append(Path(call.target).name)

    return AgentTrace(
        agent="cursor",
        mode=mode,
        task_prompt=task_prompt or "Unknown",
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
        files_modified=sorted(set(files_modified)),
    )


def _tool_call_from_cursor_tool(tool_data: dict[str, Any], project_root: Path) -> ToolCall | None:
    name = str(tool_data.get("name", "unknown"))
    name_lower = name.lower()
    args = _json_loads_maybe(tool_data.get("params")) or _json_loads_maybe(tool_data.get("rawArgs")) or {}
    result = _json_loads_maybe(tool_data.get("result"))

    target = _cursor_tool_target(name, args)
    result_text = _cursor_result_text(result, tool_data.get("result", ""))

    if any(kw in name_lower for kw in ("read_file", "open_file", "view_file")):
        tokens = measure_text_tokens(result_text) if result_text else _measure_target_file(target, project_root)
        return ToolCall("view_file", _display_target(target), tokens)

    if any(kw in name_lower for kw in ("grep", "search", "glob")):
        tokens = measure_text_tokens(result_text) if result_text else measure_grep_output_tokens()
        return ToolCall("grep_search", target, tokens)

    if any(kw in name_lower for kw in ("edit", "apply", "patch", "write")):
        return ToolCall("edit_file", _display_target(target), 0)

    if any(kw in name_lower for kw in ("terminal", "command", "run")):
        tokens = measure_text_tokens(result_text) if result_text else 50
        return ToolCall("run_command", target or name, tokens)

    if "lint" in name_lower:
        tokens = measure_text_tokens(result_text) if result_text else 0
        return ToolCall("read_lints", target, tokens)

    if name:
        tokens = measure_text_tokens(result_text) if result_text else 50
        return ToolCall(name, target, tokens)
    return None


def _cursor_tool_target(name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("grep", "search", "glob")):
        for key in ("pattern", "globPattern", "query"):
            val = args.get(key)
            if isinstance(val, str) and val:
                return val
    if any(kw in name_lower for kw in ("edit", "apply", "patch", "write")):
        for key in ("relativeWorkspacePath", "targetFile", "path", "filePath", "effectiveUri"):
            val = args.get(key)
            if isinstance(val, str) and val:
                return val
    for key in ("targetFile", "path", "filePath", "effectiveUri", "targetDirectory", "cwd"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    if isinstance(args.get("paths"), list) and args["paths"]:
        return str(args["paths"][0])
    for key in ("pattern", "globPattern", "query", "command"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    return name


def _cursor_result_text(result: Any, raw_result: Any) -> str:
    if isinstance(result, dict):
        for key in ("contents", "output", "text", "result"):
            val = result.get(key)
            if isinstance(val, str):
                return val
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        return json.dumps(result, ensure_ascii=False)
    if isinstance(raw_result, str):
        return raw_result
    return ""


def _measure_target_file(target: str, project_root: Path) -> int:
    if not target:
        return 0
    local = _resolve_cursor_path(target, project_root)
    return measure_file_tokens(local, cap_lines=800)


def _display_target(target: str) -> str:
    if not target:
        return ""
    return Path(_clean_cursor_path(target)).name or target[:80]


def _resolve_cursor_path(target: str, project_root: Path) -> Path:
    cleaned = _clean_cursor_path(target)
    direct = Path(cleaned)
    if direct.exists():
        return direct
    candidate = project_root / cleaned
    if candidate.exists():
        return candidate
    return project_root / Path(cleaned).name


def _clean_cursor_path(target: str) -> str:
    cleaned = target.replace("file:///", "")
    cleaned = unquote(cleaned).replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", cleaned):
        return cleaned
    if re.match(r"^/[a-zA-Z]:/", cleaned):
        return cleaned[1:]
    return cleaned


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _extract_rich_text(rich_text: str) -> str:
    parsed = _json_loads_maybe(rich_text)
    if not isinstance(parsed, dict):
        return ""

    texts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            val = node.get("text")
            if isinstance(val, str):
                texts.append(val)
            for child in node.get("children", []):
                walk(child)
            for key, child in node.items():
                if key not in ("text", "children"):
                    walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(parsed)
    return "\n".join(t for t in texts if t.strip())


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

    # Split by any horizontal rule variant
    blocks = re.split(r"\n(?:---|\*\*\*|___)\n", content)
    
    turn_id = 0
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        if block.startswith("**Cursor**") or block.startswith("**Assistant**"):
            text = re.sub(r"^\*\*(?:Cursor|Assistant)\*\*\s*", "", block).strip()
            if text:
                turn_id += 1
                agent_responses.append(AgentResponse(turn_id, text, measure_text_tokens(text)))
        elif (block.startswith("**User**") or block.startswith("**Human**")) and not task_prompt:
            task_prompt = re.sub(r"^\*\*(?:User|Human)\*\*\s*", "", block).strip()[:200]

    # --- Fuzzy Tool Call Extraction ---
    # 1. Look for explicit mentions of files being "read", "found", or "checked"
    file_patterns = [
        r'(?:Read|Viewing|Opened?|Found|Check(?:ed|ing)|In)\s+(?:the\s+)?([^\s\n]+\.(?:py|ts|js|tsx|jsx|json|md))',
        r'`([^\s\n]+\.(?:py|ts|js|tsx|jsx|json|md))`',
        r'([^\s\n]+\.(?:py|ts|js|tsx|jsx|json|md))'
    ]
    
    seen_files = set()
    for pattern in file_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            target = match.group(1).strip().strip('`').strip('*')
            if target in seen_files or not ('.' in target):
                continue
                
            # Filter out obvious false positives
            if target.lower() in ("requirements.txt", "readme.md", "pyproject.toml"):
                continue

            local = _resolve_cursor_path(target, project_root)
            if local.exists():
                seen_files.add(target)
                tool_calls.append(ToolCall(
                    "view_file", target,
                    measure_file_tokens(local, cap_lines=800)
                ))

    # 2. Infer searches if the agent says "searching" or "looking for"
    if re.search(r'(?:Searching|Looking for|Grep|Find)\s', content, re.IGNORECASE):
        # Add an inferred search call
        tool_calls.append(ToolCall("grep_search", "inferred", measure_grep_output_tokens()))

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
