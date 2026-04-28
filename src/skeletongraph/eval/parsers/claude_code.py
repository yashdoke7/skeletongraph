"""
Claude Code conversation export parser.

Parses the markdown export from Claude Code's `/export` command,
or the raw JSONL session logs at ~/.claude/projects/.

Export formats:
  - /export → markdown transcript
  - ~/.claude/projects/<hash>/ → JSONL structured logs
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


def discover_claude_code_sessions() -> list[Path]:
    """Find Claude Code session directories, newest first."""
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []

    sessions = []
    for project_dir in base.iterdir():
        if project_dir.is_dir():
            for f in project_dir.glob("*.jsonl"):
                sessions.append(f)

    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


def parse_claude_code_export(
    export_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse a Claude Code /export markdown file."""
    content = export_path.read_text(encoding="utf-8", errors="replace")

    # Detect format: JSONL or markdown
    if export_path.suffix == ".jsonl" or content.strip().startswith("{"):
        return _parse_jsonl(export_path, project_root, task_prompt, mode, project_name)

    return _parse_markdown(content, project_root, task_prompt, mode, project_name)


def _parse_markdown(
    content: str,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse Claude Code's markdown export format.

    Claude Code exports look similar to Antigravity but with different markers:
    - Tool calls shown as: `Read file: path/to/file.py`
    - Search shown as: `Search: query`
    - Responses are the main text blocks
    """
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    # File reads: "Read file: path" or "Read path/to/file.py"
    for match in re.finditer(r'(?:Read (?:file)?:?\s*)([^\s\n]+\.(?:py|ts|js|tsx|jsx|rs|go|java|rb|c|cpp|h))', content):
        filepath = match.group(1)
        local = _resolve(filepath, project_root)
        tool_calls.append(ToolCall(
            tool_type="view_file",
            target=Path(filepath).name,
            output_tokens=measure_file_tokens(local, cap_lines=800),
        ))

    # Searches: "Search: pattern" or "Searched for 'pattern'"
    for match in re.finditer(r'(?:Search(?:ed)?(?:\s+for)?:?\s*)["\']?(.+?)["\']?\s*$', content, re.MULTILINE):
        tool_calls.append(ToolCall(
            tool_type="grep_search",
            target=match.group(1),
            output_tokens=measure_grep_output_tokens(),
        ))

    # List files/directories
    for match in re.finditer(r'(?:Listed? (?:files|directory)|ls\b)', content):
        tool_calls.append(ToolCall(
            tool_type="list_dir",
            target="",
            output_tokens=measure_directory_listing_tokens(),
        ))

    # Edits
    for match in re.finditer(r'(?:Wrote|Edited|Updated|Modified)\s+(?:file:?\s*)?([^\s\n]+)', content):
        tool_calls.append(ToolCall(
            tool_type="edit_file",
            target=match.group(1),
            output_tokens=0,  # Edits don't have meaningful output tokens
        ))

    # Extract response blocks - Claude's responses are typically the main content
    # between user messages (marked by "Human:" or ">" or prompt indicators)
    blocks = re.split(r'(?:^>[^\n]*$|^Human:|^User:)', content, flags=re.MULTILINE)
    turn_id = 0
    for block in blocks:
        clean = block.strip()
        if clean and len(clean) > 50:  # Skip very short blocks
            turn_id += 1
            agent_responses.append(AgentResponse(
                turn_id=turn_id,
                text=clean,
                token_count=measure_text_tokens(clean),
            ))

    if not task_prompt and agent_responses:
        task_prompt = "Extracted from Claude Code export"

    return AgentTrace(
        agent="claude_code",
        mode=mode,
        task_prompt=task_prompt,
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
    )


def _parse_jsonl(
    jsonl_path: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse Claude Code's JSONL session log format.

    Data flow:
      1. `tool_use` entries → record the invocation with tool name + args
      2. `tool_result` entries → contain the ACTUAL tool output text
      3. `usage` entries → API token counts (L4 ground truth)
      4. `assistant` entries → agent response text (L2)

    We match tool_use → tool_result by tool_use_id to get actual
    L1 measurements instead of estimates.
    """
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    # Accumulate API usage across all turns (L4 ground truth)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_write = 0

    # Track pending tool_use calls waiting for their tool_result
    # Maps tool_use_id -> index in tool_calls list
    pending_tool_results: dict[str, int] = {}

    turn_id = 0
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = entry.get("type", entry.get("role", ""))
        content = entry.get("content", entry.get("text", ""))

        # ── Tool invocations ──────────────────────────────────
        if msg_type == "tool_use":
            tool_name = entry.get("name", "unknown")
            tool_input = entry.get("input", {})
            tool_use_id = entry.get("id", entry.get("tool_use_id", ""))
            target = tool_input.get("path", tool_input.get("query", ""))

            if "read" in tool_name.lower() or "view" in tool_name.lower():
                local = _resolve(target, project_root)
                tokens = measure_file_tokens(local, cap_lines=800)
                tool_calls.append(ToolCall(
                    tool_type="view_file", target=target, output_tokens=tokens
                ))
            elif "search" in tool_name.lower() or "grep" in tool_name.lower():
                # Placeholder — will be overwritten if tool_result arrives
                tool_calls.append(ToolCall(
                    tool_type="grep_search", target=target,
                    output_tokens=measure_grep_output_tokens(),
                ))
            elif any(kw in tool_name.lower() for kw in ("write", "edit", "patch", "create")):
                tool_calls.append(ToolCall(
                    tool_type="edit_file", target=target, output_tokens=0,
                ))
            elif any(kw in tool_name.lower() for kw in ("run", "exec", "bash", "shell")):
                tool_calls.append(ToolCall(
                    tool_type="run_command", target=target, output_tokens=50,
                ))
            elif any(kw in tool_name.lower() for kw in ("list", "dir", "ls")):
                tool_calls.append(ToolCall(
                    tool_type="list_dir", target=target,
                    output_tokens=measure_directory_listing_tokens(),
                ))
            else:
                tool_calls.append(ToolCall(
                    tool_type=tool_name, target=target, output_tokens=50
                ))

            # Track for tool_result matching
            if tool_use_id and tool_calls:
                pending_tool_results[tool_use_id] = len(tool_calls) - 1

        # ── Tool results (ACTUAL output content) ──────────────
        elif msg_type == "tool_result":
            tool_use_id = entry.get("tool_use_id", "")
            result_content = entry.get("content", "")

            # Extract text from content (can be string or list of blocks)
            result_text = ""
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                for block in result_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text += block.get("text", "")
                    elif isinstance(block, str):
                        result_text += block

            # Overwrite the placeholder with actual measured tokens
            if result_text and tool_use_id in pending_tool_results:
                idx = pending_tool_results[tool_use_id]
                if idx < len(tool_calls):
                    actual_tokens = measure_text_tokens(result_text)
                    tool_calls[idx].output_tokens = actual_tokens

        # ── Agent responses ────────────────────────────────────
        elif msg_type in ("assistant", "model"):
            if content:
                turn_id += 1
                text = content if isinstance(content, str) else json.dumps(content)
                agent_responses.append(AgentResponse(
                    turn_id=turn_id,
                    text=text,
                    token_count=measure_text_tokens(text),
                ))

        # ── API usage object (L4 ground truth) ─────────────────
        # Claude Code JSONL logs the Anthropic API response including
        # the usage field: {input_tokens, output_tokens,
        # cache_read_input_tokens, cache_creation_input_tokens}
        elif msg_type == "usage" or "usage" in entry:
            usage = entry.get("usage", entry) if msg_type != "usage" else entry
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)
            total_cache_read += usage.get("cache_read_input_tokens", 0)
            total_cache_write += usage.get("cache_creation_input_tokens", 0)

        elif msg_type in ("human", "user") and not task_prompt:
            task_prompt = (content if isinstance(content, str) else str(content))[:200]

    trace = AgentTrace(
        agent="claude_code",
        mode=mode,
        task_prompt=task_prompt or "Unknown",
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
    )

    # If we captured real API usage, store it as L4 ground truth
    if total_output_tokens > 0:
        trace.reasoning_tokens = total_output_tokens

    return trace


def _resolve(filepath: str, project_root: Path) -> Path:
    """Resolve a file path relative to project root."""
    p = Path(filepath)
    if p.is_absolute() and p.exists():
        return p
    candidate = project_root / filepath
    if candidate.exists():
        return candidate
    return p
