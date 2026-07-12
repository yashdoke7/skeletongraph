"""
Claude Code installer.

Writes:
  .claude/settings.json   — hooks (SessionStart, UserPromptSubmit, PostToolUse, FileChanged)
  .mcp.json               — MCP server registration (Claude Code reads MCP config from
                            project-root .mcp.json, NOT from .claude/settings.json)
  CLAUDE.md               — "use SG" rules block (appended, not overwritten)

Hooks exit 0 on any internal error (never block the agent).
All hook output goes to .skeletongraph/last_hook.log.

Cross-platform pitfalls handled here (and learned the hard way from
Claude Code v2.x on Windows):

  1. MCP server discovery: Claude Code only sees project-level MCP
     servers from .mcp.json at the project root. Putting them in
     .claude/settings.json under mcpServers is silently ignored.

  2. Windows backslashes in hook commands: Claude Code runs hooks
     through bash (Git Bash bundled). Bash interprets backslashes
     as escape characters and strips them, breaking Windows paths.
     We use forward slashes everywhere — Windows accepts them.

  3. Embedder dependency check: sentence-transformers is an OPTIONAL
     dependency that enables semantic retrieval. If missing at index
     time, SG silently falls back to BM25+graph only. We surface a
     warning so the user can install it before first index.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


_SG_RULES_BLOCK = """\
## SkeletonGraph (SG) — context assistant

SG is active for this project. Follow these rules every session:

1. **`sg_overview` is OPTIONAL — call it only when you need orientation.**
   Use it for an unfamiliar codebase, architecture/cross-cutting work, or when
   you need project purpose/constraints/recent decisions. For a focused bug fix,
   SKIP it and go straight to `sg_search` — the briefing is a cost you only pay
   when it earns its keep.
2. **Use `sg_search` as a task-context assembler, not as grep.**
   Ask for the whole task/symptom once. For coding/debug tasks it returns the
   likely edit targets as exact anchors (`file::symbol` + line range). Normal
   bug-fix searches stay precise; use `graph="on"` only for impact analysis,
   callers/callees, architecture, migration, review, or refactor work. Do not
   split one task into many symbol searches unless confidence is LOW/MISS or the
   target is absent.
3. **`sg_search` locates; `sg_expand` shows the code — edit from `sg_expand`, don't re-read.**
   `sg_search` returns exact anchors (`file::symbol` + line range), NOT bodies —
   it tells you WHERE. To see a body, call `sg_expand(target="<fqn>")`; it returns
   the exact current source with file:line, so edit straight from that. `sg_expand`
   accepts SEVERAL FQNs at once (comma-separated) — batch them in ONE call when a
   fix spans multiple functions, instead of one call each. Do NOT re-Read or
   re-grep a symbol whose body `sg_expand` already returned; that repeats work and
   adds turns for nothing. Reach for native Grep/Read only for what SG did not give
   you (e.g. finding where to insert NEW code). Ignore any `content.txt` spill (it
   just duplicates the result).
4. **SG indexes code files only.** It indexes Python, Go, and JS/TS. If you need
   to search or edit unstructured files (JSON, Markdown, YAML, configs, templates),
   use your native grep/read tools directly.
5. **Check `sg_constraint` before proposing changes** — see project rules that must not be violated.
6. **Use `sg_log` to review recent session turns** — avoids re-reading history.
7. **Stay scoped — stop when the task is done.** Make the smallest change that
   correctly satisfies the request. Once the code you have is enough to complete
   it, stop searching/reading "to be thorough". Do NOT add changelog/release
   notes, sync type-stub (`.pyi`) files, write docs, or refactor code the task
   does not require. SG surfaces related code precisely — treat that as a map to
   the RIGHT edit, not an invitation to edit everything nearby. Broaden scope
   only if the request explicitly asks for it.
8. **Do NOT verify edits by running `inspect.getsource()` on the installed
   package or grepping the site-packages directory.** Trust the file content SG
   gave you and the edits you made. Post-edit verification via Bash is wasteful
   — if you need to confirm, re-Read the few edited lines with a narrow range.

Available MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log
"""


# ── Public install entry point ───────────────────────────────────────────────


def install(project_root: Path, verbose: bool = True) -> List[str]:
    """Write Claude Code hooks + MCP config for project_root.

    Returns list of files written. Emits actionable warnings on stderr for
    optional-but-recommended setup issues (e.g. embedder missing).
    """
    project_root = project_root.resolve()
    written: List[str] = []
    path_arg = _posix_path(project_root)

    # Pre-flight: where is sg.EXE? Use the bare command if it's on PATH
    # (avoids backslash-escape issues with bash hook execution on Windows),
    # else fall back to the absolute path with forward slashes.
    sg_cmd, on_path = _resolve_sg_command()

    # ── 1. .claude/settings.json — HOOKS ONLY (no mcpServers — see docstring) ──
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"

    settings = _load_json(settings_path)
    hooks = settings.setdefault("hooks", {})

    _set_hook(hooks, "SessionStart",
              _build_hook_cmd(sg_cmd, "session_start", path_arg))
    _set_hook(hooks, "UserPromptSubmit",
              _build_hook_cmd(sg_cmd, "user_prompt_submit", path_arg))
    # SG-first gate (Grep/Glob: deny until sg_search used) + Read-dedup nudge
    # (Read: block the first re-read of a file SG already returned, point to
    # sg_expand). Matcher scopes the hook to exactly these tools.
    _set_hook(hooks, "PreToolUse",
              _build_hook_cmd(sg_cmd, "pre_tool_use", path_arg),
              matcher="Grep|Glob|Read")
    _set_hook(hooks, "PostToolUse",
              _build_hook_cmd(sg_cmd, "post_tool_use", path_arg),
              matcher="")
    _set_hook(hooks, "FileChanged",
              _build_hook_cmd(sg_cmd, "file_changed", path_arg))

    # If a prior installer wrote mcpServers HERE (old bug), strip it — that
    # entry is silently ignored by Claude Code and only confuses diagnosis.
    if "mcpServers" in settings:
        # Preserve any non-SG entries the user may have added manually
        other = {k: v for k, v in settings["mcpServers"].items()
                 if k != "skeletongraph"}
        if other:
            settings["mcpServers"] = other
        else:
            del settings["mcpServers"]

    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    written.append(str(settings_path.relative_to(project_root)))

    # ── 2. .mcp.json at PROJECT ROOT — MCP server (Claude Code reads this) ───
    mcp_path = project_root / ".mcp.json"
    mcp_config = _load_json(mcp_path)
    mcp_servers = mcp_config.setdefault("mcpServers", {})
    mcp_servers["skeletongraph"] = {
        "type": "stdio",
        "command": sg_cmd,
        "args": ["serve", "--path", path_arg],
    }
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    written.append(".mcp.json")

    # ── 3. CLAUDE.md — "use SG" rules block ──────────────────────────────────
    claude_md = project_root / "CLAUDE.md"
    _append_rules_block(claude_md, _SG_RULES_BLOCK)
    written.append("CLAUDE.md")

    # ── 4. Post-install verification + actionable warnings ───────────────────
    if verbose:
        _print_postinstall_report(sg_cmd, on_path, project_root, written)

    return written


# ── Path / command helpers ───────────────────────────────────────────────────


def _posix_path(p: Path) -> str:
    """Return path with forward slashes — survives bash hook execution on Windows.

    Windows accepts forward slashes everywhere (Python, sg.EXE, git, etc.).
    Backslashes get stripped by bash, breaking `C:\\Users\\...` into
    `C:Users...` which then fails as "command not found".
    """
    return str(p).replace("\\", "/")


def _resolve_sg_command() -> Tuple[str, bool]:
    """Decide which sg invocation to use in hooks + MCP.

    Returns (command_string, is_on_path).

    Preference order:
      1. bare `sg` if it's on PATH — cleanest, no escape issues
      2. absolute path with forward slashes — works if PATH not set up
      3. `python -m skeletongraph.cli.main` — last resort, works anywhere
    """
    sg = shutil.which("sg")
    if sg:
        # Bare command works in bash hooks AND Claude Code's MCP launcher,
        # because the OS resolves it via PATH (no escape interpretation).
        return ("sg", True)

    # No `sg` on PATH — fall back to forward-slash absolute python invocation
    py = _posix_path(Path(sys.executable))
    return (f"{py} -m skeletongraph.cli.main", False)


def _build_hook_cmd(sg_cmd: str, event: str, path_arg: str) -> str:
    """Build a hook command string that survives Windows bash execution.

    Uses single quotes around the path (bash doesn't strip backslashes inside
    single quotes either, but forward slashes are safer regardless).
    """
    return f"{sg_cmd} hook {event} --path '{path_arg}'"


# ── JSON helpers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file or return empty dict if missing/corrupt."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _set_hook(hooks: Dict, event: str, command: str,
              matcher: str | None = None) -> None:
    """Upsert a hook entry. Replace any prior SG-prefixed entry for this event
    (so re-running the installer updates stale commands instead of stacking)."""
    event_list = hooks.setdefault(event, [])

    # Drop ANY existing SG hook entries for this event (they're stale if the
    # command changed — e.g. path was relocated, sg now on PATH, etc.)
    sg_markers = ("sg.EXE", "sg hook", "skeletongraph.cli.main",
                  "sg.exe", "skeletongraph hook")
    pruned = []
    for entry in event_list:
        keep = True
        for h in entry.get("hooks", []):
            cmd = str(h.get("command", ""))
            if any(m in cmd for m in sg_markers):
                keep = False
                break
        if keep:
            pruned.append(entry)
    event_list[:] = pruned

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


# ── Post-install verification (no behavior change, just user-facing report) ──


def _print_postinstall_report(sg_cmd: str, on_path: bool,
                              project_root: Path, written: List[str]) -> None:
    """Print a one-screen summary of what installed + any actionable warnings."""
    out = sys.stderr.write

    out("\n  SkeletonGraph install — Claude Code\n")
    out(f"  Project: {project_root}\n")
    out(f"  Files written: {', '.join(written)}\n\n")

    if not on_path:
        out("  WARNING — `sg` is not on PATH. Using a fallback invocation:\n")
        out(f"    {sg_cmd}\n")
        out("  This works but is fragile. Add Python Scripts to PATH for cleaner hooks:\n")
        out("    Windows: System Properties → Environment Variables → add\n")
        out("    %LOCALAPPDATA%\\Programs\\Python\\Python311\\Scripts\n\n")

    # Embedder check — only needed for OPT-IN fusion mode. The default (rerank:
    # BM25 + structural) runs without it, so a missing embedder is informational.
    try:
        import sentence_transformers  # noqa: F401
        out("  Embedder: sentence-transformers detected — fusion mode available.\n")
        out("  (enable per-session with SG_MCP_RETRIEVAL=fusion; needs a one-time\n")
        out("   dense build, minutes/repo on CPU. Default rerank needs nothing.)\n\n")
    except ImportError:
        out("  Embedder: sentence-transformers not installed — that's fine.\n")
        out("  Default retrieval (rerank) is ready now: no model download, no prewarm.\n")
        out("  Install it only if you want fusion mode (best retrieval, one-time build):\n")
        out("    pip install sentence-transformers\n")
        out(f"    SG_MCP_RETRIEVAL=fusion sg index --path '{_posix_path(project_root)}' --force\n\n")

    # Index status — if index doesn't exist yet, user must run sg index
    if not (project_root / ".skeletongraph").exists():
        out("  Index: NOT BUILT YET. Run before launching Claude Code:\n")
        out(f"    sg index --path '{_posix_path(project_root)}'\n\n")
    else:
        out("  Index: present at .skeletongraph/. Run `sg index --force` to rebuild.\n\n")

    out("  Next steps:\n")
    out("    1. (If listed above) install sentence-transformers and/or sg index\n")
    out("    2. Launch:  claude\n")
    out("    3. In session: /trust  (if prompted)\n")
    out("    4. In session: /mcp    (should list `skeletongraph` connected)\n")
    out("    5. Test:    'Use sg_overview to show the project skeleton'\n\n")
