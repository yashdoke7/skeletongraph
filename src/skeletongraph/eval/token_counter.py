"""
Precise token counting via actual file measurement.

Unlike the main pipeline's `len(text) // 4` estimation, this module
reads actual files from disk to compute exact token costs for tool
outputs that viewed those files. Used only in the eval module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def measure_file_tokens(
    file_path: Path,
    cap_lines: int = 800,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> int:
    """Measure the token cost of viewing a file (or portion).

    Args:
        file_path: Absolute path to the source file.
        cap_lines: Maximum lines an agent can view per call (800 for Antigravity).
        start_line: 1-indexed start line (for agents that specify line ranges).
        end_line: 1-indexed end line (None = use cap_lines from start).

    Returns:
        Estimated token count for the viewed portion.
    """
    if not file_path.exists():
        return 0

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return 0

    total_lines = len(lines)

    # Apply line range
    actual_start = max(0, start_line - 1)
    if end_line is not None:
        actual_end = min(end_line, total_lines)
    else:
        actual_end = min(actual_start + cap_lines, total_lines)

    viewed_lines = lines[actual_start:actual_end]
    viewed_text = "\n".join(viewed_lines)

    return len(viewed_text) // 4


def measure_text_tokens(text: str) -> int:
    """Estimate tokens from raw text. Standard ~4 chars per token."""
    return max(0, len(text) // 4)


def measure_grep_output_tokens(
    num_matches: int = 0,
    avg_line_length: int = 80,
) -> int:
    """Estimate tokens for grep search output.

    Grep output includes filename + line number + matching line per result.
    If we don't know the exact match count, use conservative defaults.
    """
    if num_matches > 0:
        # Each result: ~20 chars overhead (path + line num) + line content
        return num_matches * (20 + avg_line_length) // 4

    # Conservative default: ~100 tokens per grep call
    return 100


def measure_directory_listing_tokens(num_entries: int = 0) -> int:
    """Estimate tokens for a directory listing.

    Each entry is typically: filename + type + size ≈ 40 chars.
    """
    if num_entries > 0:
        return num_entries * 40 // 4

    # Conservative default: ~50 tokens per listing
    return 50
