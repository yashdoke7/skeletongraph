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
from ..token_counter import measure_file_tokens, measure_text_tokens


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
            output_tokens=100,
        ))

    # List files/directories
    for match in re.finditer(r'(?:Listed? (?:files|directory)|ls\b)', content):
        tool_calls.append(ToolCall(
            tool_type="list_dir",
            target="",
            output_tokens=50,
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
    """Parse Claude Code's JSONL session log format."""
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

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

        if msg_type == "tool_use":
            tool_name = entry.get("name", "unknown")
            tool_input = entry.get("input", {})
            target = tool_input.get("path", tool_input.get("query", ""))

            if "read" in tool_name.lower() or "view" in tool_name.lower():
                local = _resolve(target, project_root)
                tokens = measure_file_tokens(local, cap_lines=800)
                tool_calls.append(ToolCall(
                    tool_type="view_file", target=target, output_tokens=tokens
                ))
            elif "search" in tool_name.lower() or "grep" in tool_name.lower():
                tool_calls.append(ToolCall(
                    tool_type="grep_search", target=target, output_tokens=100
                ))
            else:
                tool_calls.append(ToolCall(
                    tool_type=tool_name, target=target, output_tokens=50
                ))

        elif msg_type in ("assistant", "model"):
            if content:
                turn_id += 1
                text = content if isinstance(content, str) else json.dumps(content)
                agent_responses.append(AgentResponse(
                    turn_id=turn_id,
                    text=text,
                    token_count=measure_text_tokens(text),
                ))

        elif msg_type in ("human", "user") and not task_prompt:
            task_prompt = (content if isinstance(content, str) else str(content))[:200]

    return AgentTrace(
        agent="claude_code",
        mode=mode,
        task_prompt=task_prompt or "Unknown",
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
    )


def _resolve(filepath: str, project_root: Path) -> Path:
    """Resolve a file path relative to project root."""
    p = Path(filepath)
    if p.is_absolute() and p.exists():
        return p
    candidate = project_root / filepath
    if candidate.exists():
        return candidate
    return p
