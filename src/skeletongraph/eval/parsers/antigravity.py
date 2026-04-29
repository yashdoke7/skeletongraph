"""
Antigravity chat export parser.

Parses the exported .txt file from Google Antigravity IDE agent.
Auto-discovers the latest conversation from the brain directory.

Export format (from the user):
    *Listed directory [name](file:///path)*
    *Grep searched codebase*
    *Viewed [filename](file:///path)*
    *Edited relevant file*
    *User accepted the command `...`*
    ### Planner Response
    [agent response text]
    ### User Input
    [user message]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import (
    measure_file_tokens, measure_text_tokens,
    measure_grep_output_tokens, measure_directory_listing_tokens,
)


# ── Auto-discovery ─────────────────────────────────────────────────────

def discover_antigravity_logs(project_root: Optional[Path] = None) -> list[Path]:
    """Find all Antigravity conversation logs, newest first."""
    brain_dir = Path.home() / ".gemini" / "antigravity" / "brain"
    if not brain_dir.exists():
        return []

    logs = []
    for conv_dir in brain_dir.iterdir():
        if not conv_dir.is_dir():
            continue
        overview = conv_dir / ".system_generated" / "logs" / "overview.txt"
        if overview.exists():
            logs.append(overview)

    # Sort by modification time, newest first
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


def discover_latest_log() -> Optional[Path]:
    """Get the most recent Antigravity conversation log."""
    logs = discover_antigravity_logs()
    return logs[0] if logs else None


# ── Parser ─────────────────────────────────────────────────────────────

# Patterns for parsing the exported chat
VIEW_PATTERN = re.compile(
    r'\*Viewed \[(.+?)\]\(file:///(.+?)\)\s*\*'
)
GREP_PATTERN = re.compile(r'\*Grep searched codebase\*')
LIST_DIR_PATTERN = re.compile(r'\*Listed directory \[(.+?)\]')
EDIT_PATTERN = re.compile(r'\*Edited relevant file\*')
COMMAND_PATTERN = re.compile(r'\*User accepted the command `(.+?)`\*')
USER_INPUT_PATTERN = re.compile(r'^### User Input$', re.MULTILINE)
PLANNER_RESPONSE_PATTERN = re.compile(r'^### Planner Response$', re.MULTILINE)


def parse_antigravity_export(
    export_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse an Antigravity exported chat into an AgentTrace.

    Args:
        export_path: Path to the exported .txt chat file.
        project_root: Root of the project being analyzed (for file measurement).
        task_prompt: The original task prompt (extracted from first user message if empty).
        mode: "native" or "skeletongraph".
        project_name: e.g. "flask", "fastapi".

    Returns:
        AgentTrace with precise token measurements.
    """
    content = export_path.read_text(encoding="utf-8", errors="replace")

    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []
    files_modified: list[str] = []

    # ── Extract tool calls ─────────────────────────────────────────

    for match in VIEW_PATTERN.finditer(content):
        filename = match.group(1)
        file_uri = match.group(2)

        # Convert URI to local path for measurement
        local_path = _uri_to_path(file_uri, project_root)
        tokens = measure_file_tokens(local_path, cap_lines=800)

        tool_calls.append(ToolCall(
            tool_type="view_file",
            target=filename,
            output_tokens=tokens,
        ))

    for match in GREP_PATTERN.finditer(content):
        tool_calls.append(ToolCall(
            tool_type="grep_search",
            target="",
            output_tokens=measure_grep_output_tokens(),
        ))

    for match in LIST_DIR_PATTERN.finditer(content):
        tool_calls.append(ToolCall(
            tool_type="list_dir",
            target=match.group(1),
            output_tokens=measure_directory_listing_tokens(),
        ))

    for match in EDIT_PATTERN.finditer(content):
        files_modified.append("(edited)")

    for match in COMMAND_PATTERN.finditer(content):
        tool_calls.append(ToolCall(
            tool_type="run_command",
            target=match.group(1),
            output_tokens=50,  # Command output is typically small
        ))

    # ── Extract agent responses ────────────────────────────────────

    # Split content into sections by ### headers
    sections = re.split(r'(### (?:User Input|Planner Response))', content)

    turn_id = 0
    for i, section in enumerate(sections):
        if section.strip() == "### Planner Response" and i + 1 < len(sections):
            response_text = sections[i + 1].strip()
            # Clean out the tool action summaries (lines starting with *)
            clean_lines = []
            for line in response_text.splitlines():
                if not line.strip().startswith("*"):
                    clean_lines.append(line)
            clean_text = "\n".join(clean_lines).strip()

            if clean_text:
                turn_id += 1
                agent_responses.append(AgentResponse(
                    turn_id=turn_id,
                    text=clean_text,
                    token_count=measure_text_tokens(clean_text),
                ))

    # Extract task prompt from first user input if not provided
    if not task_prompt:
        for i, section in enumerate(sections):
            if section.strip() == "### User Input" and i + 1 < len(sections):
                first_input = sections[i + 1].strip()
                # Get first non-empty, non-action line
                for line in first_input.splitlines():
                    if line.strip() and not line.strip().startswith("*"):
                        task_prompt = line.strip()
                        break
                break

    return AgentTrace(
        agent="antigravity",
        mode=mode,
        task_prompt=task_prompt,
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        files_modified=files_modified,
        task_completed=True,  # Can be overridden
    )


def parse_antigravity_sg_session(
    project_root: Path,
    project_name: str = "",
    agent: str = "antigravity",
    session_path: Optional[Path] = None,
) -> AgentTrace:
    """Parse the SkeletonGraph session data from current.json.

    Captures ALL layers for the SG side:
      L1: query_context tool output tokens (already working)
      L2: agent response tokens (from response_text if available)
      L3: history compounding (computed from L1+L2 per turn, same as native)
      L5: MCP schema overhead (num_turns x schema cost per turn)

    Layer 4 (reasoning) is not available for Antigravity (Gemini internal).
    """
    from ..token_counter import measure_text_tokens, measure_mcp_schema_overhead

    session_path = session_path or project_root / ".skeletongraph" / "session" / "current.json"
    if not session_path.exists():
        raise FileNotFoundError(f"No SG session found at {session_path}")

    import json
    data = json.loads(session_path.read_text(encoding="utf-8"))

    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []
    task_prompt = ""

    for turn in data.get("turns", []):
        if not task_prompt:
            task_prompt = turn.get("prompt", "")

        # L1: The assembled context payload sent to the agent
        tool_calls.append(ToolCall(
            tool_type="query_context",
            target=turn.get("prompt", ""),
            output_tokens=turn.get("token_count", 0),
        ))

        # Retrieval-quality scoring needs file paths, not just the prompt.
        # These zero-token markers represent files surfaced inside the SG context.
        files_seen = set()
        for fqn in turn.get("fqns_returned", []):
            if "::" not in fqn:
                continue
            file_path = fqn.split("::", 1)[0]
            if "." not in Path(file_path).name or file_path in files_seen:
                continue
            files_seen.add(file_path)
            tool_calls.append(ToolCall(
                tool_type="sg_context_file",
                target=file_path,
                output_tokens=0,
            ))

        # L2: The agent's response to this turn (from CHANGE 8 field)
        response_text = turn.get("response_text", "")
        if response_text:
            agent_responses.append(AgentResponse(
                turn_id=len(agent_responses) + 1,
                text=response_text,
                token_count=measure_text_tokens(response_text),
            ))

    # L5: Schema overhead — every turn loads all tool schemas
    num_turns = len(data.get("turns", []))
    schema_overhead = measure_mcp_schema_overhead(num_turns=num_turns)

    return AgentTrace(
        agent=agent,
        mode="skeletongraph",
        task_prompt=task_prompt,
        project=project_name,
        tool_calls=tool_calls,
        agent_responses=agent_responses,
        task_completed=True,
        mcp_schema_overhead_tokens=schema_overhead,
    )


# ── Helpers ────────────────────────────────────────────────────────────

def _uri_to_path(file_uri: str, project_root: Path) -> Path:
    """Convert a file:/// URI to a local Path, searching in project_root."""
    # Try direct path from URI
    cleaned = file_uri.replace("file:///", "").replace("/", "\\")
    direct = Path(cleaned)
    if direct.exists():
        return direct

    # Try relative to project root
    # Extract just the filename
    filename = Path(cleaned).name
    for candidate in project_root.rglob(filename):
        return candidate

    return direct  # Return best guess even if not found
