"""
Copilot (VS Code) chat export parser.

Parses the JSON export from VS Code's "Chat: Export Session..." command.
Also supports the Agent Debug Logs for detailed token tracking.

Export locations:
  - JSON Export: User-selected file path
  - Chat sessions: %APPDATA%/Code/User/workspaceStorage/<hash>/chatSessions/
  - Debug logs: Enabled via github.copilot.chat.agentDebugLog.fileLogging.enabled
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import measure_file_tokens, measure_text_tokens


def discover_copilot_sessions() -> list[Path]:
    """Find Copilot chat session files, newest first."""
    base = Path.home() / "AppData" / "Roaming" / "Code" / "User" / "workspaceStorage"
    if not base.exists():
        return []

    sessions = []
    for workspace_dir in base.iterdir():
        if not workspace_dir.is_dir():
            continue
        chat_dir = workspace_dir / "chatSessions"
        if chat_dir.exists():
            for f in chat_dir.glob("*.json"):
                sessions.append(f)

    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions


def parse_copilot_json_export(
    export_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse a Copilot JSON export file.

    Copilot's JSON export typically contains:
    - requester/responder message pairs
    - references (files used for context)
    - token usage metadata (if debug mode enabled)
    """
    content = export_path.read_text(encoding="utf-8", errors="replace")
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fall back to text parsing for non-JSON exports
        return _parse_copilot_text(export_path, project_root, task_prompt, mode, project_name)

    # Handle array-of-messages format
    messages = data if isinstance(data, list) else data.get("messages", data.get("turns", []))

    turn_id = 0
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type", ""))
            text = msg.get("content", msg.get("text", msg.get("message", "")))

            # Extract references (files injected as context)
            refs = msg.get("references", msg.get("context", []))
            if isinstance(refs, list):
                for ref in refs:
                    ref_path = ref.get("uri", ref.get("path", "")) if isinstance(ref, dict) else str(ref)
                    if ref_path:
                        local = _resolve_path(ref_path, project_root)
                        tokens = measure_file_tokens(local, cap_lines=800)
                        tool_calls.append(ToolCall(
                            tool_type="view_file",
                            target=Path(ref_path).name,
                            output_tokens=tokens,
                        ))

            # Extract agent responses
            if role in ("assistant", "responder", "model") and text:
                turn_id += 1
                agent_responses.append(AgentResponse(
                    turn_id=turn_id,
                    text=text,
                    token_count=measure_text_tokens(text),
                ))

            # Extract task prompt
            if not task_prompt and role in ("user", "requester") and text:
                task_prompt = text.strip()[:200]

            # Extract token usage if available
            usage = msg.get("usage", {})
            if usage and isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                if prompt_tokens:
                    tool_calls.append(ToolCall(
                        tool_type="api_context",
                        target="copilot_context_injection",
                        output_tokens=prompt_tokens,
                    ))

    return AgentTrace(
        agent="copilot",
        mode=mode,
        task_prompt=task_prompt,
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
    )


def _parse_copilot_text(
    export_path: Path,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Fallback text-based parser for Copilot exports."""
    content = export_path.read_text(encoding="utf-8", errors="replace")
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    # Count file references
    file_refs = re.findall(r'Used (\d+) references', content)
    for ref_count in file_refs:
        tool_calls.append(ToolCall(
            tool_type="context_injection",
            target=f"{ref_count} references",
            output_tokens=int(ref_count) * 3000,  # ~3000 tokens per reference file
        ))

    # Extract prompt_tokens if in debug format
    token_matches = re.findall(r'"prompt_tokens":\s*(\d+)', content)
    for t in token_matches:
        tool_calls.append(ToolCall(
            tool_type="api_prompt_tokens",
            target="measured",
            output_tokens=int(t),
        ))

    return AgentTrace(
        agent="copilot",
        mode=mode,
        task_prompt=task_prompt or "Unknown",
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
    )


def _resolve_path(ref_path: str, project_root: Path) -> Path:
    """Resolve a file reference to a local path."""
    cleaned = ref_path.replace("file:///", "").replace("\\", "/")
    direct = Path(cleaned)
    if direct.exists():
        return direct
    candidate = project_root / Path(cleaned).name
    return candidate if candidate.exists() else direct
