"""
Precise token counting via actual file measurement.

All parsers import from here — fixes in this file apply everywhere.
Uses tiktoken BPE encoding for ~5% accuracy vs true tokenization,
compared to len//4's ~20%+ error on code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
        return len(_enc.encode(text))

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
    "query_context":    218,
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
