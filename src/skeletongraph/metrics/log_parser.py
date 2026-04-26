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
    """Parse an Antigravity log (raw or exported summary)."""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    
    # Pattern for raw logs
    raw_views = re.findall(FILE_VIEW_PATTERN, content)
    raw_greps = re.findall(GREP_PATTERN, content)
    
    # Pattern for exported summaries (Aggressive Keyword Matching)
    # Catches: *Viewed [file](link)* or *Viewed file* or Viewed [file]
    summary_views = re.findall(r'[vV]iewed\s+\[?([\w\.]+)', content)
    summary_greps = re.findall(r'[gG]rep\s+searched', content)
    
    # Check for "Listed directory" which also costs tokens
    summary_lists = re.findall(r'[lL]isted\s+directory', content)
    
    view_count = len(raw_views) + len(summary_views)
    # Also count "Listed directory" as a small view
    view_count += len(summary_lists)
    
    grep_count = len(raw_greps) + len(summary_greps)
    
    # Estimate tokens: 3000 tokens per viewed file (avg) + 200 per grep
    tokens = (view_count * 3000) + (grep_count * 200)
    
    return {
        "files_viewed": view_count,
        "grep_searches": grep_count,
        "total_native_tokens": tokens,
        "duration_ms": 0,
        "files_involved": summary_views
    }

def parse_copilot_log(log_path: Path) -> Dict[str, Any]:
    """Parse a VS Code GitHub Copilot Chat Trace log."""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    
    # 1. Look for the "usage :" JSON blob in the request summary
    usage_match = re.search(r'usage\s*:\s*(\{.*?\})', content)
    if usage_match:
        import json
        try:
            usage_data = json.loads(usage_match.group(1))
            return {
                "files_viewed": 0,
                "grep_searches": 1,
                "total_native_tokens": usage_data.get("prompt_tokens", 0),
                "duration_ms": 0,
                "files_involved": []
            }
        except:
            pass

    # 2. Fallback to older debug log JSON counts
    token_matches = re.findall(r'"prompt_tokens":\s*(\d+)', content)
    total_tokens = sum(int(t) for t in token_matches)
    
    # 3. Look for reference counting (e.g. "Used 12 references")
    ref_matches = re.findall(r'Used (\d+) references', content)
    total_refs = sum(int(r) for r in ref_matches)
    
    if total_tokens == 0 and total_refs > 0:
        total_tokens = total_refs * 3000
        
    return {
        "files_viewed": total_refs,
        "grep_searches": 1,
        "total_native_tokens": total_tokens,
        "duration_ms": 0,
        "files_involved": []
    }
