"""
Precise token counting via actual file measurement.

All parsers import from here — fixes in this file apply everywhere.
Uses tiktoken BPE encoding for ~5% accuracy vs true tokenization,
compared to len//4's ~20%+ error on code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import subprocess

# ── Token counting engine ──────────────────────────────────────────────

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    # cl100k_base is the encoding used by GPT-4, Claude uses a similar
    # BPE scheme — this gives ~5% accuracy vs true Claude tokenization,
    # which is far better than len//4's ~20% error on code.

    def measure_text_tokens(text: str) -> int:
        """Count tokens using tiktoken BPE. Accurate for code."""
        if not text:
            return 0
        return len(_enc.encode(text, allowed_special="all"))

    def measure_file_tokens(
        file_path: Path,
        cap_lines: int = 800,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> int:
        """Measure the token cost of viewing a file or portion of it."""
        if not file_path.exists():
            return 0
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return 0

        actual_start = max(0, start_line - 1)
        if end_line is not None:
            actual_end = min(end_line, len(lines))
        else:
            actual_end = min(actual_start + cap_lines, len(lines))

        viewed_text = "\n".join(lines[actual_start:actual_end])
        return len(_enc.encode(viewed_text))

except ImportError:
    # Fallback if tiktoken not installed — better than crashing
    def measure_text_tokens(text: str) -> int:  # type: ignore[misc]
        return max(0, len(text) // 4)

    def measure_file_tokens(  # type: ignore[misc]
        file_path: Path,
        cap_lines: int = 800,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> int:
        if not file_path.exists():
            return 0
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return 0
        actual_start = max(0, start_line - 1)
        if end_line is not None:
            actual_end = min(end_line, len(lines))
        else:
            actual_end = min(actual_start + cap_lines, len(lines))
        viewed_text = "\n".join(lines[actual_start:actual_end])
        return max(0, len(viewed_text) // 4)


# ── MCP schema overhead constants ─────────────────────────────────────
#
# These are the token costs of SG's tool schemas as loaded by the agent
# from the MCP protocol. Measured by encoding the actual JSON schema
# of each tool. The agent receives ALL schemas every single turn,
# so the overhead is: num_turns x sum(all schema tokens).
#
# To re-measure these after schema changes:
#   import json, tiktoken
#   enc = tiktoken.get_encoding("cl100k_base")
#   schema_json = json.dumps(tool["inputSchema"])  # from mcp.py _TOOLS
#   print(len(enc.encode(schema_json)))

SG_TOOL_SCHEMA_TOKENS: dict[str, int] = {
    "query_context":    230,
    "expand_context":   210,
    "expand_function":  162,
    "show_graph":       148,
    "search_index":     141,
    "index_status":     88,
    "review_delta":     195,
    "get_blast_radius": 155,
    "get_dependencies": 152,
    "detect_changes":   170,
    "get_stats":        95,
}

# Total cost per turn to load all SG tools
SG_TOTAL_SCHEMA_TOKENS_PER_TURN = sum(SG_TOOL_SCHEMA_TOKENS.values())  # ~1,524


def measure_mcp_schema_overhead(num_turns: int, active_tools: list[str] = None) -> int:
    """
    Calculate the total MCP schema overhead across all turns.

    The agent loads all tool schemas into context at the start of EVERY turn.
    This is real token spend that doesn't show up in tool output measurements.

    Args:
        num_turns: How many agent turns happened in this session.
        active_tools: List of tool names active (None = all SG tools).

    Returns:
        Total tokens spent loading tool schemas across the full session.
    """
    if active_tools is None:
        per_turn = SG_TOTAL_SCHEMA_TOKENS_PER_TURN
    else:
        per_turn = sum(SG_TOOL_SCHEMA_TOKENS.get(t, 150) for t in active_tools)
    return per_turn * num_turns


def measure_grep_output_tokens(num_matches: int = 0, avg_line_length: int = 80) -> int:
    """Estimate tokens for grep search output."""
    if num_matches > 0:
        return num_matches * (20 + avg_line_length) // 4
    return 100


def measure_directory_listing_tokens(num_entries: int = 0) -> int:
    """Estimate tokens for a directory listing."""
    if num_entries > 0:
        return num_entries * 40 // 4
    return 50


def measure_codebase_tokens(project_root: Path) -> int:
    """Measure the absolute token size of the entire readable codebase.
    
    This acts as the static 'Total Repo Baseline' for evaluation reports.
    Attempts to read tracked files via `git ls-files`, with a fallback 
    to standard recursive iteration for non-git folders.
    """
    if not project_root.exists() or not project_root.is_dir():
        return 0

    files_to_measure = []
    
    # Try git ls-files first to automatically respect .gitignore
    try:
        result = subprocess.run(
            ["git", "ls-files"], 
            cwd=project_root, 
            capture_output=True, 
            text=True, 
            check=True
        )
        for line in result.stdout.splitlines():
            file_path = project_root / line.strip()
            if file_path.is_file():
                files_to_measure.append(file_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: scan directory manually, ignoring hidden folders and common binaries
        ignore_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".next", ".pytest_cache"}
        for fp in project_root.rglob("*"):
            if not fp.is_file():
                continue
            if any(part in ignore_dirs for part in fp.parts):
                continue
            # Skip likely binary/media files
            if fp.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mov", ".db", ".sqlite"}:
                continue
            files_to_measure.append(fp)

    total_tokens = 0
    for fp in files_to_measure:
        # We don't cap lines for the codebase baseline, we want the true size
        if fp.exists():
            try:
                # Use measure_text_tokens to avoid the cap_lines logic in measure_file_tokens
                text = fp.read_text(encoding="utf-8", errors="ignore")
                total_tokens += measure_text_tokens(text)
            except Exception:
                pass

    return total_tokens
