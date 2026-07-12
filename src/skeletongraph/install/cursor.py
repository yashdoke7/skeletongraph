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

1. `sg_overview` is OPTIONAL — call it only when you need orientation (unfamiliar
   codebase, architecture/cross-cutting work). Skip it for a focused bug fix and
   go straight to `sg_search`.
2. Use `sg_search` as a task-context assembler, not as grep. Ask for the whole
   task/symptom once. For coding/debug tasks it returns the likely edit targets as
   exact anchors (`file::symbol` + line range). Normal bug-fix searches stay
   precise; use `graph="on"` only for impact analysis, callers/callees,
   architecture, migration, review, or refactor work. Do not split one task into
   many symbol searches unless confidence is LOW/MISS or the target is absent.
3. `sg_search` locates; `sg_expand` shows the code — edit from `sg_expand`, don't
   re-read. `sg_search` returns exact anchors (`file::symbol` + line range), NOT
   bodies. To see a body call `sg_expand(target="<fqn>")` — it returns the exact
   current source with file:line; edit straight from that. Do NOT re-Read or
   re-grep a symbol whose body `sg_expand` already returned (repeats work, adds
   turns). `sg_expand` takes SEVERAL FQNs at once (comma-separated) — batch them in
   ONE call when a fix spans multiple functions. Use native Grep/Read only for what
   SG did not give you (e.g. where to insert NEW code); ignore `content.txt` spill.
4. SG indexes code files only (Python, Go, JS/TS). If you need to search or edit
   unstructured files (JSON, Markdown, YAML, configs, templates), use your native
   grep/read tools directly.
5. Check `sg_constraint` before proposing architectural changes.
6. Use `sg_log` to review recent session history.
7. Stay scoped: make the smallest change that satisfies the request and stop when
   it's done. Don't add changelog/release notes, sync `.pyi` stubs, write docs, or
   refactor code the task doesn't require. SG surfaces related code precisely — use
   it to find the RIGHT edit, not to edit everything nearby.
8. Do NOT verify edits by running `inspect.getsource()` on the installed
   package or grepping the site-packages directory. Trust the file content SG
   gave you and the edits you made. Post-edit verification via terminal is wasteful
   — if you need to confirm, re-Read the few edited lines with a narrow range.

MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
"""


def install(project_root: Path, verbose: bool = True) -> List[str]:
    """Write Cursor MCP config + hooks + rules for project_root.

    Returns list of files written. See claude_code.py for the cross-platform
    pitfalls (bash strips backslashes, prefer bare `sg` if on PATH).
    """
    project_root = project_root.resolve()
    written: List[str] = []
    cursor_dir = project_root / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    sg_cmd, on_path = _resolve_sg_command()
    path_arg = _posix_path(project_root)

    # ── 1. .cursor/mcp.json ───────────────────────────────────────────
    mcp_path = cursor_dir / "mcp.json"
    mcp_config = _load_json(mcp_path)
    mcp_config.setdefault("mcpServers", {})["skeletongraph"] = {
        "type": "stdio",
        "command": sg_cmd,
        "args": ["serve", "--path", path_arg],
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
    # Same bash-escape concern as Claude Code: use forward-slash path +
    # single-quote the --path argument.
    settings_path = cursor_dir / "settings.json"
    settings = _load_json(settings_path)
    hooks = settings.setdefault("hooks", {})

    _set_hook(hooks, "sessionStart",
              f"{sg_cmd} hook session_start --path '{path_arg}'")
    _set_hook(hooks, "beforeSubmitPrompt",
              f"{sg_cmd} hook user_prompt_submit --path '{path_arg}'")
    _set_hook(hooks, "afterFileEdit",
              f"{sg_cmd} hook file_changed --path '{path_arg}'")

    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    written.append(".cursor/settings.json")

    if verbose:
        _print_postinstall_report(sg_cmd, on_path, project_root, written)

    return written


# ── Helpers ───────────────────────────────────────────────────────────────


def _posix_path(p: Path) -> str:
    """Forward slashes — bash strips backslashes when running hook commands."""
    return str(p).replace("\\", "/")


def _resolve_sg_command():
    """Bare `sg` if on PATH (cleanest), else absolute python invocation."""
    import shutil
    sg = shutil.which("sg")
    if sg:
        return ("sg", True)
    py = _posix_path(Path(sys.executable))
    return (f"{py} -m skeletongraph.cli.main", False)


def _print_postinstall_report(sg_cmd: str, on_path: bool,
                              project_root: Path, written: List[str]) -> None:
    out = sys.stderr.write
    out("\n  SkeletonGraph install — Cursor\n")
    out(f"  Project: {project_root}\n")
    out(f"  Files written: {', '.join(written)}\n\n")
    if not on_path:
        out(f"  WARNING — `sg` is not on PATH. Using fallback: {sg_cmd}\n\n")
    try:
        import sentence_transformers  # noqa: F401
        out("  Embedder: sentence-transformers detected.\n")
    except ImportError:
        out("  WARNING — sentence-transformers not installed (semantic retrieval off).\n")
        out("  Optional install:  pip install sentence-transformers\n\n")
    if not (project_root / ".skeletongraph").exists():
        out(f"  Index: NOT BUILT. Run:  sg index --path '{_posix_path(project_root)}'\n\n")


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
