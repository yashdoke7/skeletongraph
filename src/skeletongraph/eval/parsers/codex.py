"""
OpenAI Codex CLI session parser.

Parses local logs from ~/.codex/ directory or output from
@ccusage/codex (which gives exact token breakdowns).

Codex logs typically contain structured JSONL with:
  - Tool calls (file reads, searches, edits)
  - Token usage (input, output, reasoning, cache)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import (
    measure_file_tokens, measure_text_tokens,
    measure_grep_output_tokens, measure_directory_listing_tokens,
)


def discover_codex_sessions() -> list[Path]:
    """Find Codex session log files, newest first."""
    base = Path.home() / ".codex"
    if not base.exists():
        return []

    logs = []
    for f in base.rglob("*.json"):
        logs.append(f)
    for f in base.rglob("*.jsonl"):
        logs.append(f)

    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


def parse_codex_session(
    log_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse a Codex session log file.

    Supports:
    - JSONL format (structured tool calls)
    - JSON format (session summary)
    - @ccusage output format (token breakdowns)
    """
    content = log_path.read_text(encoding="utf-8", errors="replace")
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []
    reasoning_tokens = 0

    # Try JSONL format first
    if content.strip().startswith("{") and "\n{" in content:
        return _parse_jsonl(content, project_root, task_prompt, mode, project_name)

    # Try JSON format
    try:
        data = json.loads(content)
        return _parse_json(data, project_root, task_prompt, mode, project_name)
    except json.JSONDecodeError:
        pass

    # Fallback: text-based parsing
    return _parse_text(content, project_root, task_prompt, mode, project_name)


def _parse_jsonl(
    content: str,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse Codex JSONL log format."""
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []
    reasoning_tokens = 0
    turn_id = 0

    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type", "")

        # Tool calls
        if entry_type in ("tool_call", "function_call"):
            name = entry.get("name", entry.get("function", {}).get("name", ""))
            args = entry.get("arguments", entry.get("input", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            target = args.get("path", args.get("file", args.get("query", "")))

            if any(kw in name.lower() for kw in ("read", "view", "cat", "file")):
                local = project_root / target if target else Path("")
                tokens = measure_file_tokens(local, cap_lines=800)
                tool_calls.append(ToolCall("view_file", target, tokens))
            elif any(kw in name.lower() for kw in ("search", "grep", "find")):
                # Use actual result content if available
                result_content = entry.get("result", entry.get("output", ""))
                if result_content and isinstance(result_content, str) and len(result_content) > 10:
                    out_tokens = measure_text_tokens(result_content)
                else:
                    out_tokens = measure_grep_output_tokens()
                tool_calls.append(ToolCall("grep_search", target, out_tokens))
            elif any(kw in name.lower() for kw in ("write", "edit", "patch")):
                tool_calls.append(ToolCall("edit_file", target, 0))
            elif any(kw in name.lower() for kw in ("run", "exec", "shell")):
                result_content = entry.get("result", entry.get("output", ""))
                if result_content and isinstance(result_content, str) and len(result_content) > 10:
                    out_tokens = measure_text_tokens(result_content)
                else:
                    out_tokens = 50
                tool_calls.append(ToolCall("run_command", target, out_tokens))
            elif any(kw in name.lower() for kw in ("list", "dir", "ls")):
                tool_calls.append(ToolCall("list_dir", target, measure_directory_listing_tokens()))
            else:
                tool_calls.append(ToolCall(name, target, 50))

        # Responses
        elif entry_type in ("assistant", "response"):
            text = entry.get("content", entry.get("text", ""))
            if text:
                turn_id += 1
                agent_responses.append(AgentResponse(turn_id, text, measure_text_tokens(text)))

        # Token usage
        elif entry_type == "usage":
            reasoning_tokens += entry.get("reasoning_tokens", 0)

        # User messages
        elif entry_type in ("user", "human") and not task_prompt:
            task_prompt = str(entry.get("content", ""))[:200]

    trace = AgentTrace(
        agent="codex", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )
    if reasoning_tokens:
        trace.reasoning_tokens = reasoning_tokens
    return trace


def _parse_json(
    data: dict,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse a Codex JSON session summary."""
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    # @ccusage format
    if "sessions" in data:
        for session in data["sessions"]:
            for msg in session.get("messages", []):
                if msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    if text:
                        agent_responses.append(AgentResponse(
                            len(agent_responses) + 1, text, measure_text_tokens(text)
                        ))

    # Usage summary
    usage = data.get("usage", data.get("total_usage", {}))
    reasoning = usage.get("reasoning_tokens", 0)

    trace = AgentTrace(
        agent="codex", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )
    if reasoning:
        trace.reasoning_tokens = reasoning
    return trace


def _parse_text(
    content: str,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Fallback text parser for Codex output."""
    tool_calls: list[ToolCall] = []

    # Look for file reads
    for match in re.finditer(r'(?:Reading|Read|Viewed?)\s+([^\s\n]+\.(?:py|ts|js|go|rs))', content):
        local = project_root / match.group(1)
        tool_calls.append(ToolCall("view_file", match.group(1),
                                   measure_file_tokens(local, cap_lines=800)))

    return AgentTrace(
        agent="codex", mode=mode, task_prompt=task_prompt or "Unknown",
        project=project_name, tool_calls=tool_calls, task_completed=True,
    )
