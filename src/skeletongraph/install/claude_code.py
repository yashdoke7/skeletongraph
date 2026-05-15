"""
Claude Code installer.

Writes:
  .claude/settings.json   — hooks (SessionStart, UserPromptSubmit, PostToolUse, FileChanged)
                            + MCP server registration
  CLAUDE.md               — "use SG" rules block (appended, not overwritten)

Hooks exit 0 on any internal error (never block the agent).
All hook output goes to .skeletongraph/last_hook.log.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


_SG_RULES_BLOCK = """\
## SkeletonGraph (SG) — context assistant

SG is active for this project. Follow these rules every session:

1. **Call `sg_overview` first** — at session start, before reading any files.
   It shows the top functions by PageRank, active constraints, and recent turns.
2. **Use `sg_search` instead of grep/glob** — hybrid BM25 + graph centrality search.
3. **Use `sg_get` / `sg_expand` instead of reading full files** — token-efficient retrieval.
4. **Check `sg_constraint` before proposing changes** — see project rules that must not be violated.
5. **Use `sg_log` to review recent session turns** — avoids re-reading history.

Available MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
"""


def install(project_root: Path, verbose: bool = True) -> List[str]:
    """Write Claude Code hooks + MCP config for project_root.

    Returns list of files written.
    """
    written: List[str] = []

    # ── 1. .claude/settings.json — hooks + MCP ───────────────────────
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"

    settings = _load_json(settings_path)

    # Hooks section
    sg_exe = _sg_exe()
    path_arg = str(project_root)
    hooks = settings.setdefault("hooks", {})

    _set_hook(hooks, "SessionStart", f'{sg_exe} hook session_start --path "{path_arg}"')
    _set_hook(hooks, "UserPromptSubmit", f'{sg_exe} hook user_prompt_submit --path "{path_arg}"')
    _set_hook(hooks, "PostToolUse", f'{sg_exe} hook post_tool_use --path "{path_arg}"', matcher="")
    _set_hook(hooks, "FileChanged", f'{sg_exe} hook file_changed --path "{path_arg}"')

    # MCP server
    mcp_servers = settings.setdefault("mcpServers", {})
    mcp_servers["skeletongraph"] = {
        "command": sg_exe,
        "args": ["serve", "--path", path_arg],
        "type": "stdio",
    }

    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    written.append(str(settings_path.relative_to(project_root)))

    # ── 2. CLAUDE.md — "use SG" rules block ──────────────────────────
    claude_md = project_root / "CLAUDE.md"
    _append_rules_block(claude_md, _SG_RULES_BLOCK)
    written.append("CLAUDE.md")

    return written


# ── Helpers ───────────────────────────────────────────────────────────────


def _sg_exe() -> str:
    """Path to the sg executable (same Python env that's running now)."""
    # Prefer the sg script; fall back to python -m invocation
    import shutil
    sg = shutil.which("sg")
    if sg:
        return sg
    return f"{sys.executable} -m skeletongraph.cli.main"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file or return empty dict if missing/corrupt."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _set_hook(hooks: Dict, event: str, command: str, matcher: str | None = None) -> None:
    """Upsert a hook entry. Skips if command already present."""
    event_list = hooks.setdefault(event, [])

    # Check if already wired
    for entry in event_list:
        for h in entry.get("hooks", []):
            if h.get("command", "") == command:
                return  # already present

    hook_entry: Dict[str, Any] = {"type": "command", "command": command}
    wrapper: Dict[str, Any] = {"hooks": [hook_entry]}
    if matcher is not None:
        wrapper["matcher"] = matcher
    event_list.append(wrapper)


def _append_rules_block(target: Path, block: str) -> None:
    """Append the SG rules block to a markdown file. Skip if already present."""
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        if "SkeletonGraph (SG)" in existing:
            return
        content = existing.rstrip() + "\n\n" + block
    else:
        content = block
    target.write_text(content, encoding="utf-8")
