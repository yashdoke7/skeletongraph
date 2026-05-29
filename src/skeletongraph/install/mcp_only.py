"""
MCP-only installer for IDEs with no first-class hooks:
  Cline, Roo, Zed, Continue, Copilot (VS Code)

Writes:
  - MCP config in the IDE's expected location
  - "use SG" rules block appended to the IDE's rules file

No hooks wired (these IDEs don't support them). The agent discovers SG
through the rules file + MCP tools list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


_SG_RULES_BLOCK = """\

## SkeletonGraph (SG) — context assistant

SG MCP tools are available. Use them every session:

- `sg_overview`   — project briefing: project purpose, structure, constraints,
                    recent turns/decisions, and index status (call FIRST).
- `sg_search`     — task-context assembler, not grep. Ask for the whole task or
                    symptom once; for coding/debug tasks it returns likely edit
                    targets, imports/prelude, compact helper context, and likely
                    tests. Normal bug-fix searches stay precise; use `graph="on"`
                    only for impact analysis, callers/callees, architecture,
                    migration, review, or refactor work. Do not split one task
                    into many symbol searches unless confidence is LOW/MISS or
                    the target is absent.
- `sg_get`        — exact FQN metadata when you already know the target.
- `sg_expand`     — exact follow-up only. Expand a specific FQN when you are
                    about to edit it and the body was not already returned.
                    sg_search results are complete inline (exact source + file:line
                    range) — edit from them; do NOT re-grep/re-read returned code,
                    and ignore any `content.txt` spill (it duplicates the result).
- `sg_constraint` — view/propose project constraints
- `sg_log`        — recent session log entries
"""


# Per-IDE config: (rules_file, mcp_config_path, mcp_key)
_IDE_CONFIGS: Dict[str, tuple] = {
    "cline":    (".clinerules",        ".vscode/mcp.json",     "servers"),
    "roo":      (".roorules",          ".vscode/mcp.json",     "servers"),
    "continue": (".continue/config.md", ".continue/config.json", "models"),  # special
    "zed":      (None,                 ".zed/settings.json",   "assistant"),  # special
    "copilot":  (".github/copilot-instructions.md", ".vscode/mcp.json", "servers"),
    "windsurf": (".windsurfrules",     ".windsurf/mcp.json",   "mcpServers"),
}


def install(ide: str, project_root: Path, verbose: bool = True) -> List[str]:
    """Write MCP config + rules block for a no-hooks IDE.

    Returns list of files written. Uses forward-slash paths so the config
    survives any host (Windows + bash, Linux, macOS) without escape issues.
    """
    if ide not in _IDE_CONFIGS:
        return []

    project_root = project_root.resolve()
    rules_file, mcp_path_rel, mcp_key = _IDE_CONFIGS[ide]
    written: List[str] = []
    sg_cmd, on_path = _resolve_sg_command()
    path_arg = _posix_path(project_root)

    server_entry = {
        "type": "stdio",
        "command": sg_cmd,
        "args": ["serve", "--path", path_arg],
    }

    # ── MCP config ────────────────────────────────────────────────────
    if mcp_path_rel and mcp_key not in ("models", "assistant"):
        mcp_path = project_root / mcp_path_rel
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_config = _load_json(mcp_path)
        mcp_config.setdefault(mcp_key, {})["skeletongraph"] = server_entry
        mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
        written.append(mcp_path_rel)

    # Zed: uses assistant.default_model section — just add mcp_servers at top level
    if ide == "zed":
        mcp_path = project_root / ".zed/settings.json"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        zed_config = _load_json(mcp_path)
        zed_config.setdefault("context_servers", {})["skeletongraph"] = {
            "command": {"path": sg_cmd, "args": ["serve", "--path", path_arg]},
        }
        mcp_path.write_text(json.dumps(zed_config, indent=2), encoding="utf-8")
        written.append(".zed/settings.json")

    # ── Rules block ───────────────────────────────────────────────────
    if rules_file:
        target = project_root / rules_file
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
            if "SkeletonGraph" in existing:
                pass  # already installed
            else:
                target.write_text(existing.rstrip() + _SG_RULES_BLOCK, encoding="utf-8")
                written.append(rules_file)
        else:
            target.write_text(_SG_RULES_BLOCK.lstrip(), encoding="utf-8")
            written.append(rules_file)

    if verbose:
        _print_postinstall_report(ide, sg_cmd, on_path, project_root, written)

    return written


# ── Helpers ───────────────────────────────────────────────────────────────


def _posix_path(p: Path) -> str:
    """Forward slashes — survives bash escape stripping on Windows hosts."""
    return str(p).replace("\\", "/")


def _resolve_sg_command():
    """Bare `sg` if on PATH (cleanest), else absolute python invocation."""
    import shutil
    sg = shutil.which("sg")
    if sg:
        return ("sg", True)
    py = _posix_path(Path(sys.executable))
    return (f"{py} -m skeletongraph.cli.main", False)


def _print_postinstall_report(ide: str, sg_cmd: str, on_path: bool,
                              project_root: Path, written: List[str]) -> None:
    out = sys.stderr.write
    out(f"\n  SkeletonGraph install — {ide}\n")
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
