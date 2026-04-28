"""
Copilot (VS Code) chat export parser.

Parses the JSON export from VS Code's "Chat: Export Session..." command.
Uses the authoritative data sources in the export:
  - result.metadata.toolCallRounds: tool names, IDs, and arguments
  - result.metadata.toolCallResults: actual tool output (VS Code tree nodes)
  - completionTokens: API-level output token count (L4)
  - response[]: agent text and thinking blocks (L2)

Export locations:
  - JSON Export: User-selected file path (Ctrl+Shift+P -> Chat: Export Session)
  - Chat sessions: %APPDATA%/Code/User/workspaceStorage/<hash>/chatSessions/
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from ..schema import AgentTrace, ToolCall, AgentResponse
from ..token_counter import measure_file_tokens, measure_text_tokens


# ── VS Code tree node text extractor ──────────────────────────────────

def _extract_text_from_nodes(node) -> str:
    """Recursively extract text content from VS Code tree node structures.

    Copilot tool results are stored as serialized VS Code tree nodes
    (type 1 = container, type 2 = text leaf). This walks the tree and
    concatenates all text leaves.
    """
    texts: list[str] = []

    if isinstance(node, dict):
        # Text leaf node (type 2)
        if "text" in node and isinstance(node["text"], str):
            texts.append(node["text"])
        # String value (some results are plain strings)
        if "value" in node and isinstance(node["value"], str):
            texts.append(node["value"])
        # Recurse into children array
        for child in node.get("children", []):
            texts.append(_extract_text_from_nodes(child))
        # Recurse into 'node' wrapper
        if "node" in node and isinstance(node["node"], (dict, list)):
            texts.append(_extract_text_from_nodes(node["node"]))
    elif isinstance(node, list):
        for item in node:
            texts.append(_extract_text_from_nodes(item))

    return "\n".join(t for t in texts if t)


def _measure_tool_result(result: dict) -> int:
    """Measure token count of a toolCallResult by extracting its text content."""
    content_list = result.get("content", [])
    total_text = ""
    for c in content_list:
        if isinstance(c, dict):
            val = c.get("value")
            if isinstance(val, str):
                total_text += val
            elif isinstance(val, (dict, list)):
                total_text += _extract_text_from_nodes(val)
    if not total_text:
        return 0
    return measure_text_tokens(total_text)


# ── Tool call classification ──────────────────────────────────────────

# Tools whose output is code retrieval (counts toward L1)
_RETRIEVAL_TOOLS = {
    "read_file", "readFile", "copilot_readFile",
    "grep_search", "search", "findTextInFiles", "copilot_findTextInFiles",
    "list_files", "list_dir", "listFiles",
}

# Tools that are write operations (0 retrieval tokens)
_WRITE_TOOLS = {
    "apply_patch", "applyPatch", "copilot_applyPatch",
    "insert_edit_into_file", "insertEditIntoFile",
    "write_file", "writeFile",
    "create_file", "createFile",
}

# Tools that are meta/exec (count output but not as retrieval)
_META_TOOLS = {
    "manage_todo_list",
}


def _classify_tool(name: str) -> str:
    """Classify a tool call by name. Returns 'retrieval', 'write', 'exec', or 'meta'."""
    name_lower = name.lower()
    if name in _RETRIEVAL_TOOLS or any(kw in name_lower for kw in ("read", "grep", "search", "find", "list")):
        return "retrieval"
    if name in _WRITE_TOOLS or any(kw in name_lower for kw in ("patch", "edit", "write", "create", "insert")):
        return "write"
    if name in _META_TOOLS or "todo" in name_lower:
        return "meta"
    if any(kw in name_lower for kw in ("run", "exec", "code", "snippet", "python", "terminal")):
        return "exec"
    return "other"


def _infer_tool_type(name: str) -> str:
    """Map a Copilot tool name to our unified tool_type."""
    name_lower = name.lower()
    if "read" in name_lower or name_lower == "read_file":
        return "view_file"
    if any(kw in name_lower for kw in ("grep", "search", "find")):
        return "grep_search"
    if any(kw in name_lower for kw in ("patch", "edit", "write", "insert", "create")):
        return "edit_file"
    if any(kw in name_lower for kw in ("list", "dir")):
        return "list_dir"
    if any(kw in name_lower for kw in ("run", "exec", "code", "snippet", "python", "terminal")):
        return "run_command"
    if "todo" in name_lower:
        return "meta"
    return name


# ── Auto-discovery ────────────────────────────────────────────────────

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


# ── Main parser ───────────────────────────────────────────────────────

def parse_copilot_json_export(
    export_path: Path,
    project_root: Path,
    task_prompt: str = "",
    mode: str = "native",
    project_name: str = "",
) -> AgentTrace:
    """Parse a Copilot JSON export file.

    Uses the authoritative data sources (toolCallRounds + toolCallResults)
    when available, falling back to response-array parsing for older formats.
    """
    content = export_path.read_text(encoding="utf-8", errors="replace")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _parse_copilot_text(export_path, project_root, task_prompt, mode, project_name)

    # ── Format A: VS Code 'Export Session' (requests/response) ──────
    if isinstance(data, dict) and "requests" in data:
        return _parse_vscode_export(data, project_root, task_prompt, mode, project_name)

    # ── Format B: Flat Messages (APIs / Other tools) ───────────────
    return _parse_flat_messages(data, project_root, task_prompt, mode, project_name)


def _parse_vscode_export(
    data: dict,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse VS Code 'Export Session' format using authoritative data sources.

    Data flow:
      1. toolCallRounds gives us tool names + args (what was called)
      2. toolCallResults gives us actual output (what was returned)
      3. completionTokens gives us L4 (API output tokens)
      4. response[] gives us L2 (agent text to user) and thinking blocks
    """
    all_tool_calls: list[ToolCall] = []
    all_responses: list[AgentResponse] = []
    total_reasoning_tokens = 0
    total_api_output_tokens = 0
    files_modified: list[str] = []

    for req_idx, req in enumerate(data.get("requests", [])):
        # ── Extract task prompt from user message ──────────────────
        user_msg = req.get("message", {})
        if isinstance(user_msg, dict):
            p = user_msg.get("text", user_msg.get("message", ""))
            if p and not task_prompt:
                task_prompt = p.strip()[:200]

        # ── L4: API output tokens ─────────────────────────────────
        api_completion = req.get("completionTokens", 0)
        if api_completion:
            total_api_output_tokens += api_completion

        # ── Try authoritative sources first ───────────────────────
        result_meta = {}
        result_obj = req.get("result", {})
        if isinstance(result_obj, dict):
            result_meta = result_obj.get("metadata", {})

        rounds = result_meta.get("toolCallRounds", []) if isinstance(result_meta, dict) else []
        tcr = result_meta.get("toolCallResults", {}) if isinstance(result_meta, dict) else {}

        if rounds and tcr:
            # ── PRIMARY PATH: use toolCallRounds + toolCallResults ──
            tool_calls, modified = _parse_from_rounds(rounds, tcr, project_root)
            all_tool_calls.extend(tool_calls)
            files_modified.extend(modified)
        else:
            # ── FALLBACK: parse from response array ────────────────
            tool_calls, modified = _parse_from_response_array(
                req.get("response", []), project_root
            )
            all_tool_calls.extend(tool_calls)
            files_modified.extend(modified)

        # ── L2: Agent responses from response array ────────────────
        response_parts = req.get("response", [])
        if isinstance(response_parts, list):
            combined_text = _extract_agent_text(response_parts)
            if combined_text:
                all_responses.append(AgentResponse(
                    turn_id=req_idx + 1,
                    text=combined_text,
                    token_count=measure_text_tokens(combined_text),
                ))

        # ── L4: Reasoning tokens from thinking blocks ──────────────
        if isinstance(response_parts, list):
            for part in response_parts:
                if not isinstance(part, dict):
                    continue
                if part.get("kind") == "thinking":
                    val = part.get("value", "")
                    if isinstance(val, str) and val.strip():
                        total_reasoning_tokens += measure_text_tokens(val)

    # Use API completionTokens as L4 if available (more accurate than
    # counting decoded thinking text, since thinking is often encrypted)
    reasoning = total_api_output_tokens if total_api_output_tokens > 0 else (
        total_reasoning_tokens if total_reasoning_tokens > 0 else None
    )

    return AgentTrace(
        agent="copilot",
        mode=mode,
        task_prompt=task_prompt or "Unknown",
        project=project_name,
        tool_calls=all_tool_calls,
        agent_responses=all_responses,
        task_completed=True,
        reasoning_tokens=reasoning,
        files_modified=files_modified,
    )


def _parse_from_rounds(
    rounds: list,
    tcr: dict,
    project_root: Path,
) -> tuple[list[ToolCall], list[str]]:
    """Parse tool calls from toolCallRounds + toolCallResults.

    This is the authoritative path — it has exact tool names, arguments,
    and actual output content.
    """
    tool_calls: list[ToolCall] = []
    files_modified: list[str] = []

    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue

        for tc in rnd.get("toolCalls", []):
            call_id = tc.get("id", "")
            name = tc.get("name", "unknown")
            args_raw = tc.get("arguments", "{}")

            # Parse arguments
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}

            # Get actual tool output tokens from toolCallResults
            result = tcr.get(call_id, {})
            output_tokens = _measure_tool_result(result) if result else 0

            # Classify and extract target
            category = _classify_tool(name)
            tool_type = _infer_tool_type(name)
            target = _extract_target(name, args, project_root)

            # For write operations, track modified files but don't count
            # their result as retrieval tokens
            if category == "write":
                file_path = args.get("filePath", args.get("file", ""))
                if not file_path and name == "apply_patch":
                    # Extract from patch input
                    patch_input = args.get("input", "")
                    if "Update File:" in patch_input:
                        file_path = patch_input.split("Update File:")[1].strip().split("\n")[0].strip()
                
                if file_path:
                    files_modified.append(Path(file_path).name)
                # Write tool results are just confirmations, not retrieval
                output_tokens = 0

            elif category == "meta":
                output_tokens = 0

            tool_calls.append(ToolCall(
                tool_type=tool_type,
                target=target,
                output_tokens=output_tokens,
            ))

    return tool_calls, files_modified


def _extract_target(name: str, args: dict, project_root: Path) -> str:
    """Extract a human-readable target from tool arguments."""
    name_lower = name.lower()

    if "read" in name_lower or "file" in name_lower:
        fp = args.get("filePath", args.get("file", args.get("path", "")))
        start = args.get("startLine", "")
        end = args.get("endLine", "")
        if fp:
            basename = Path(fp).name
            if start and end:
                return f"{basename}#{start}-{end}"
            return basename
    elif any(kw in name_lower for kw in ("grep", "search", "find")):
        query = args.get("query", args.get("pattern", ""))
        return query[:60] if query else ""
    elif any(kw in name_lower for kw in ("patch", "edit", "write")):
        fp = args.get("filePath", args.get("file", ""))
        if not fp and name_lower == "apply_patch":
            patch_input = args.get("input", "")
            if "Update File:" in patch_input:
                fp = patch_input.split("Update File:")[1].strip().split("\n")[0].strip()
        return Path(fp).name if fp else "unknown"
    elif "todo" in name_lower:
        return "todo_list"
    elif any(kw in name_lower for kw in ("code", "snippet", "python", "run")):
        return "code_execution"

    # Fallback: try common arg names
    for key in ("filePath", "file", "path", "query", "pattern"):
        val = args.get(key, "")
        if val:
            return str(val)[:60]
    return ""


def _extract_agent_text(response_parts: list) -> str:
    """Extract agent's text responses from the response array (L2).

    Concatenates all plain text blocks (non-tool, non-thinking, non-reference).
    """
    text_parts: list[str] = []

    for part in response_parts:
        if not isinstance(part, dict):
            continue

        kind = part.get("kind")

        # Skip tool invocations, thinking, references, progress
        if kind in (
            "toolInvocationSerialized",
            "thinking",
            "inlineReference",
            "progressTaskSerialized",
            "mcpServersStarting",
        ):
            continue

        # Plain text value (the agent's actual response text)
        if kind is None and "value" in part:
            val = part["value"]
            if isinstance(val, str) and val.strip():
                text_parts.append(val.strip())

    return "\n".join(text_parts) if text_parts else ""


# ── Fallback: parse from response array ───────────────────────────────

def _parse_from_response_array(
    response_parts: list,
    project_root: Path,
) -> tuple[list[ToolCall], list[str]]:
    """Fallback parser for exports without toolCallRounds metadata.

    Extracts tool calls from toolInvocationSerialized entries in the
    response array. Less accurate than the rounds-based parser.
    """
    tool_calls: list[ToolCall] = []
    files_modified: list[str] = []

    for part in response_parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") != "toolInvocationSerialized":
            continue

        tool_name = part.get("toolId", "unknown")
        category = _classify_tool(tool_name)
        tool_type = _infer_tool_type(tool_name)

        # Try to extract file path from resultDetails or invocationMessage
        target = ""
        details = part.get("resultDetails", [])
        if details and isinstance(details, list):
            uri_data = details[0].get("uri", {}) if isinstance(details[0], dict) else {}
            target = uri_data.get("fsPath", uri_data.get("path", ""))

        if not target:
            inv = part.get("invocationMessage", {})
            msg = inv.get("value", "") if isinstance(inv, dict) else str(inv) if inv else ""
            if isinstance(msg, str):
                # Extract path from file:/// URI
                match = re.search(r'file:///([^\s`)]+)', msg)
                if match:
                    target = match.group(1)
                # Extract line range
                line_match = re.search(r'lines?\s+(\d+)\s+to\s+(\d+)', msg)
                if line_match and target:
                    target = f"{Path(target).name}#{line_match.group(1)}-{line_match.group(2)}"

        if category == "retrieval" and target:
            # Measure actual file content for the line range
            local = _resolve_path(target.split("#")[0], project_root)
            line_range = target.split("#")[1] if "#" in target else ""
            if line_range:
                parts = line_range.split("-")
                try:
                    start, end = int(parts[0]), int(parts[1])
                    tokens = measure_file_tokens(local, start_line=start, end_line=end)
                except (ValueError, IndexError):
                    tokens = measure_file_tokens(local, cap_lines=800)
            else:
                tokens = measure_file_tokens(local, cap_lines=800)
            tool_calls.append(ToolCall(
                tool_type=tool_type,
                target=Path(target.split("#")[0]).name if target else "",
                output_tokens=tokens if tokens > 0 else 50,
            ))
        elif category == "write":
            if target:
                files_modified.append(Path(target).name)
            # Write ops have 0 retrieval tokens
        elif category in ("exec", "other"):
            tool_calls.append(ToolCall(
                tool_type=tool_type,
                target=Path(target).name if target else tool_name,
                output_tokens=50,
            ))

    return tool_calls, files_modified


# ── Format B: Flat messages ───────────────────────────────────────────

def _parse_flat_messages(
    data,
    project_root: Path,
    task_prompt: str,
    mode: str,
    project_name: str,
) -> AgentTrace:
    """Parse flat messages format (APIs / other tools)."""
    tool_calls: list[ToolCall] = []
    agent_responses: list[AgentResponse] = []

    messages = data if isinstance(data, list) else data.get("messages", data.get("turns", []))
    turn_id = 0
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("type", ""))
            text = msg.get("content", msg.get("text", msg.get("message", "")))

            # Extract references
            refs = msg.get("references", msg.get("context", []))
            if isinstance(refs, list):
                for ref in refs:
                    ref_path = ref.get("uri", ref.get("path", "")) if isinstance(ref, dict) else str(ref)
                    if ref_path:
                        local = _resolve_path(ref_path, project_root)
                        tokens = measure_file_tokens(local, cap_lines=800)
                        tool_calls.append(ToolCall(
                            tool_type="view_file", target=Path(ref_path).name, output_tokens=tokens,
                        ))

            if role in ("assistant", "responder", "model") and text:
                turn_id += 1
                agent_responses.append(AgentResponse(
                    turn_id=turn_id, text=text, token_count=measure_text_tokens(text),
                ))

            if not task_prompt and role in ("user", "requester") and text:
                task_prompt = text.strip()[:200]

    return AgentTrace(
        agent="copilot", mode=mode, task_prompt=task_prompt,
        project=project_name, tool_calls=tool_calls,
        agent_responses=agent_responses, task_completed=True,
    )


# ── Fallback text parser ─────────────────────────────────────────────

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
    # Handle URL-encoded paths
    try:
        from urllib.parse import unquote
        cleaned = unquote(cleaned)
    except ImportError:
        pass
    direct = Path(cleaned)
    if direct.exists():
        return direct
    candidate = project_root / Path(cleaned).name
    return candidate if candidate.exists() else direct
