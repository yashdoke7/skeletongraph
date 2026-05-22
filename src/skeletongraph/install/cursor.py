"""
Cursor installer.

Writes:
  .cursor/mcp.json                    — MCP server registration
  .cursor/rules/skeletongraph.mdc     — "use SG" rules (always-on rule)
  .cursor/settings.json               — hooks (Cursor v1.7+: beforeSubmitPrompt,
                                        afterFileEdit, sessionStart)

Cursor hooks are a v1.7+ feature. We write them but wrap gracefully if the
version doesn't support them yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


_SG_RULES_MDC = """\
---
description: SkeletonGraph context assistant — call these tools every session
alwaysApply: true
---

## SkeletonGraph (SG)

SG is active. Follow these rules:

1. Call `sg_overview` first at session start. It is the project briefing:
   project purpose, important structure, constraints, recent turns, and index status.
2. Use `sg_search` as a task-context assembler, not as grep. Ask for the whole
   task/symptom once. For coding/debug tasks it returns likely edit targets,
   imports/prelude, helper bodies, graph neighbors, and likely tests.
   Do not split one task into many symbol searches unless confidence is LOW/MISS
   or the target is absent.
3. Use `sg_get` / `sg_expand` only for exact follow-up. Expand a specific FQN
   only when you are about to edit it and the body was not already in `sg_search`.
   `sg_search` results are complete and self-contained (each body is exact current
   source with its file:line range) — edit directly from them; do NOT re-grep or
   re-read code `sg_search` already returned, and ignore any `content.txt` spill.
4. Check `sg_constraint` before proposing architectural changes.
5. Use `sg_log` to review recent session history.

MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
"""


def install(project_root: Path, verbose: bool = True) -> List[str]:
    """Write Cursor MCP config + hooks + rules for project_root.

    Returns list of files written.
    """
    written: List[str] = []
    cursor_dir = project_root / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    sg_exe = _sg_exe()
    path_arg = str(project_root)

    # ── 1. .cursor/mcp.json ───────────────────────────────────────────
    mcp_path = cursor_dir / "mcp.json"
    mcp_config = _load_json(mcp_path)
    mcp_config.setdefault("mcpServers", {})["skeletongraph"] = {
        "command": sg_exe,
        "args": ["serve", "--path", path_arg],
        "type": "stdio",
    }
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    written.append(".cursor/mcp.json")

    # ── 2. .cursor/rules/skeletongraph.mdc ───────────────────────────
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    mdc_path = rules_dir / "skeletongraph.mdc"
    if not mdc_path.exists():
        mdc_path.write_text(_SG_RULES_MDC, encoding="utf-8")
        written.append(".cursor/rules/skeletongraph.mdc")

    # ── 3. .cursor/settings.json — Cursor v1.7 hooks ─────────────────
    settings_path = cursor_dir / "settings.json"
    settings = _load_json(settings_path)
    hooks = settings.setdefault("hooks", {})

    _set_hook(hooks, "sessionStart",
              f'{sg_exe} hook session_start --path "{path_arg}"')
    _set_hook(hooks, "beforeSubmitPrompt",
              f'{sg_exe} hook user_prompt_submit --path "{path_arg}"')
    _set_hook(hooks, "afterFileEdit",
              f'{sg_exe} hook file_changed --path "{path_arg}"')

    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    written.append(".cursor/settings.json")

    return written


# ── Helpers ───────────────────────────────────────────────────────────────


def _sg_exe() -> str:
    import shutil
    sg = shutil.which("sg")
    if sg:
        return sg
    return f"{sys.executable} -m skeletongraph.cli.main"


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _set_hook(hooks: Dict, event: str, command: str) -> None:
    """Upsert a Cursor hook entry. Skips if command already present."""
    event_list = hooks.setdefault(event, [])
    for entry in event_list:
        if entry.get("command", "") == command:
            return
    event_list.append({"type": "command", "command": command})
