"""
Empirical log parser to extract true baseline metrics from unaltered agent sessions.
Currently supports Antigravity overview.txt parsing.
"""

from pathlib import Path
import re
from typing import Dict, List, Any

# Simple regex to catch tool call returns
# Antigravity's overview.txt tracks everything chronologically
FILE_VIEW_PATTERN = r"""call:default_api:view_file[\s\S]*?response:default_api:view_file\{output:[`"']?(.*?)['"`]?\}"""
GREP_PATTERN = r"""call:default_api:grep_search[\s\S]*?response:default_api:grep_search\{output:[`"']?(.*?)['"`]?\}"""

def parse_antigravity_log(log_path: Path) -> Dict[str, Any]:
    """Parse an Antigravity overview.txt log to find file reads and greps."""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    
    view_files = re.findall(FILE_VIEW_PATTERN, content)
    greps = re.findall(GREP_PATTERN, content)
    
    total_view_chars = sum(len(body) for body in view_files)
    total_grep_chars = sum(len(body) for body in greps)
    
    total_chars = total_view_chars + total_grep_chars
    tokens = total_chars // 4  # standard approximation for rough source code char-to-token
    
    return {
        "files_viewed": len(view_files),
        "grep_searches": len(greps),
        "total_native_tokens": tokens,
        "duration_ms": 0, # Cannot reliably parse agent think time from raw txt easily
        "files_involved": [] # Could regex file paths if needed, but keeping it simple for tokens
    }
