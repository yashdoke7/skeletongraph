"""
CLI commands for agent integration and hook management.

Commands:
  sg init --agent <name>    Bootstrap agent integration files
  sg hooks install          Install Claude Code hooks
  sg hooks test             Test hook pipeline with sample prompts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from ..config import SGConfig, AGENT_PRESETS, load_config, save_config
from ..integrations.templates import get_template, get_hook_script


def cmd_init_agent(
    project_root: Path,
    agent: str,
    force: bool = False,
) -> None:
    """Initialize SkeletonGraph integration for a specific agent.
    
    Creates the appropriate config files and integration templates.
    """
    if agent not in AGENT_PRESETS:
        print(f"Unknown agent: {agent}")
        print(f"Available: {', '.join(AGENT_PRESETS.keys())}")
        sys.exit(1)
    
    sg_dir = project_root / ".skeletongraph"
    sg_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create agent-specific config
    config = SGConfig.from_agent_preset(agent)
    save_config(config, project_root)
    print(f"✓ Config saved with {agent} model presets")
    print(f"  SLM: {config.slm_model}")
    print(f"  MLM: {config.mlm_model}")
    print(f"  LLM: {config.llm_model}")
    
    # 2. Write integration template
    _write_agent_template(project_root, agent, force)
    
    # 3. Agent-specific post-setup
    preset = AGENT_PRESETS[agent]
    integration = preset.get("integration", "mcp")
    
    print(f"\n--- Next Steps ---")
    if "mcp" in integration:
        _print_mcp_setup(project_root, agent)
    if "hooks" in integration:
        _print_hooks_setup(project_root)
    if "agents_md" in integration:
        print(f"  AGENTS.md written to project root")
    if "extension" in integration:
        print(f"  Extension snippet written. See docs for VS Code integration.")


def _write_agent_template(project_root: Path, agent: str, force: bool) -> None:
    """Write the agent-specific integration template."""
    template = get_template(agent, str(project_root))
    
    targets = {
        "claude_code": ("CLAUDE.md", project_root / "CLAUDE.md"),
        "cursor": (".cursorrules", project_root / ".cursorrules"),
        "codex": ("AGENTS.md", project_root / "AGENTS.md"),
        "copilot": ("copilot_extension.js", project_root / ".skeletongraph" / "copilot_extension.js"),
        "antigravity": ("mcp_config.json", project_root / ".skeletongraph" / "mcp_config.json"),
    }
    
    name, path = targets.get(agent, ("template.txt", project_root / ".skeletongraph" / "template.txt"))
    
    if path.exists() and not force:
        print(f"  ⚠ {name} already exists (use --force to overwrite)")
        return
    
    path.write_text(template, encoding="utf-8")
    print(f"✓ {name} written to {path.relative_to(project_root)}")


def _print_mcp_setup(project_root: Path, agent: str) -> None:
    """Print MCP setup instructions."""
    mcp_cmd = f"python -m skeletongraph.server.mcp --path {project_root}"
    
    if agent == "claude_code":
        print(f"\n  Add to ~/.claude/mcp_settings.json:")
        print(f'    "skeletongraph": {{')
        print(f'      "command": "python",')
        print(f'      "args": ["-m", "skeletongraph.server.mcp", "--path", "{project_root}"]')
        print(f'    }}')
    elif agent == "cursor":
        print(f"\n  Add MCP server in Cursor Settings → MCP Servers:")
        print(f"    Command: {mcp_cmd}")
    elif agent == "antigravity":
        print(f"\n  MCP config written to .skeletongraph/mcp_config.json")
        print(f"  Add this server in Antigravity's MCP configuration.")
    else:
        print(f"\n  MCP command: {mcp_cmd}")


def _print_hooks_setup(project_root: Path) -> None:
    """Print hook setup instructions for Claude Code."""
    print(f"\n  Claude Code hooks:")
    print(f"    1. Run: sg hooks install")
    print(f"    2. Enable in Claude Code settings: Hooks → SkeletonGraph")


def cmd_hooks_install(project_root: Path) -> None:
    """Install Claude Code hooks."""
    hooks_dir = Path.home() / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    hook_script = get_hook_script(str(project_root))
    hook_file = hooks_dir / "skeletongraph.sh"
    hook_file.write_text(hook_script, encoding="utf-8")
    
    # Make executable on Unix
    try:
        hook_file.chmod(0o755)
    except Exception:
        pass
    
    print(f"✓ Hook script installed: {hook_file}")
    print(f"  Enable in Claude Code: Settings → Hooks → Add SkeletonGraph")


def cmd_hooks_test(project_root: Path, prompt: Optional[str] = None) -> None:
    """Test the hook pipeline with a sample prompt."""
    from ..hooks.claude_code import hook_session_start, hook_pre_prompt
    
    test_prompt = prompt or "fix the authentication bug in the login handler"
    
    print("--- Testing Hook Pipeline ---\n")
    
    # 1. Session start
    print("1. hook_session_start():")
    result = hook_session_start(project_root)
    print(f"   {result[:200]}")
    
    # 2. Pre-prompt
    print(f"\n2. hook_pre_prompt(\"{test_prompt}\"):")
    result = hook_pre_prompt(project_root, test_prompt)
    print(f"   {result}")
    
    # 3. Check context.md
    context_path = project_root / ".skeletongraph" / "context.md"
    if context_path.exists():
        content = context_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        print(f"\n3. context.md: {len(lines)} lines, ~{len(content) // 4} tokens")
        print(f"   Preview: {lines[0][:100]}..." if lines else "   (empty)")
    else:
        print("\n3. context.md: not created")
    
    print("\n--- Test Complete ---")
