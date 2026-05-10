"""
CLI prepare command: runs the full v3 pipeline and writes context files.

Usage:
  sg prepare "fix the content-length bug in GET requests"

Flow:
  1. Load index
  2. Classify query → QueryType + ContextMode
  3. Resolve targets
  4. Score confidence
  5. Assemble (attention-optimal, mode-based layers)
  6. Write .skeletongraph/context.md
  7. Optionally write shadow files
"""

from __future__ import annotations

import json
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command("prepare")
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--mode-override", "-m", default=None, help="Override mode: fast|standard|deep|planning|review")
@click.option("--out", "-o", default=None, help="Write context to this file instead of .skeletongraph/context.md")
@click.option("--copy", "copy_to_clipboard", is_flag=True, help="Copy the prepared context to the clipboard")
@click.option("--for", "for_agent", default=None, help="Target workflow hint: aider|copilot|web|claude|codex")
@click.option("--shadows/--no-shadows", default=True, help="Write shadow files for Claude Code")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def prepare(
    prompt: str,
    path: str,
    mode_override: Optional[str],
    out: Optional[str],
    copy_to_clipboard: bool,
    for_agent: Optional[str],
    shadows: bool,
    quiet: bool,
):
    """Run the full v3 pipeline and write context files."""
    project_root = Path(path).resolve()
    sg_dir = project_root / ".skeletongraph"

    if not sg_dir.exists():
        console.print("[red]Error:[/red] No .skeletongraph index found. Run `sg build` first.")
        sys.exit(1)

    t0 = time.perf_counter()

    from ..engine import SGEngine
    from ..retrieval.resolver import Tier

    engine = SGEngine(project_root=project_root)
    result = engine.query(prompt, delivery="cli")
    config = engine.get_config()
    if not result.success:
        console.print(f"[red]Error:[/red] {result.error}")
        sys.exit(1)

    targets = [
        c.skeleton.fqn for c in result.candidates
        if c.tier == Tier.TIER1
    ]

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # 6. Write context.md
    context_path = Path(out).resolve() if out else sg_dir / "context.md"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(result.context_text, encoding="utf-8")

    copied = False
    if copy_to_clipboard:
        copied = _copy_text_to_clipboard(result.context_text)

    # 7. Write shadow files (for Claude Code)
    if shadows and targets:
        shadow_dir = sg_dir / "shadows"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        _write_shadow_files(result.candidates, project_root, shadow_dir)

    # 8. Log to hit_log.jsonl
    eval_dir = sg_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _log_hit(eval_dir / "hit_log.jsonl", prompt, result, targets, duration_ms)

    # Output
    if quiet:
        console.print(
            f"{result.query_mode.value} {result.context_tokens}tok "
            f"{result.model_tier.value} {duration_ms}ms"
        )
    else:
        table = Table(show_header=False, border_style="dim")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mode", result.query_mode.value.upper())
        table.add_row("Model Tier", result.model_tier.value)
        table.add_row("CLI Provider", config.cli_provider)
        table.add_row("Recommended Model", result.recommended_model or "default")
        table.add_row("Confidence", result.confidence)
        table.add_row("Complexity", f"{result.complexity_score:.2f}")
        table.add_row("Routing", result.routing_reason or "static")
        table.add_row("Tokens", str(result.context_tokens))
        table.add_row("Targets", ", ".join(targets[:3]) or "none")
        table.add_row("Pipeline", result.pipeline_path)
        if for_agent:
            table.add_row("Prepared For", for_agent)
        if result.slm_used:
            table.add_row("SLM Entities", str(result.slm_entities_found))
        table.add_row("Time", f"{duration_ms}ms")
        console.print(Panel(table, title="sg prepare", border_style="blue"))
        console.print(f"  -> {context_path}")
        if copy_to_clipboard:
            status = "copied" if copied else "clipboard unavailable"
            console.print(f"  -> {status}")


def _write_shadow_files(candidates, project_root: Path, shadow_dir: Path):
    """Write pre-assembled file views as shadow files."""
    # Collect unique file paths from targets
    files_seen = set()
    for candidate in candidates:
        sk = candidate.skeleton
        fp = sk.file_path
        if fp in files_seen:
            continue
        files_seen.add(fp)

        source = project_root / fp
        if not source.exists():
            continue

        # Mirror directory structure
        shadow_path = shadow_dir / fp
        shadow_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = source.read_text(encoding="utf-8", errors="replace")
            shadow_path.write_text(content, encoding="utf-8")
        except Exception:
            pass


def _log_hit(log_path: Path, prompt: str, result, targets: list[str], duration_ms: int):
    """Append a hit log entry for passive evaluation."""
    entry = {
        "timestamp": time.time(),
        "prompt": prompt[:200],
        "mode": result.query_mode.value,
        "model_tier": result.model_tier.value,
        "base_model_tier": result.base_model_tier.value,
        "complexity_score": result.complexity_score,
        "routing_reason": result.routing_reason,
        "confidence": result.confidence,
        "tokens": result.context_tokens,
        "targets": targets[:5],
        "pipeline_path": result.pipeline_path,
        "slm_used": result.slm_used,
        "slm_entities_found": result.slm_entities_found,
        "duration_ms": duration_ms,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _copy_text_to_clipboard(text: str) -> bool:
    """Copy text using common platform clipboard commands."""
    commands = [
        ("clip", []),
        ("pbcopy", []),
        ("xclip", ["-selection", "clipboard"]),
        ("xsel", ["--clipboard", "--input"]),
    ]
    for executable, args in commands:
        if not shutil.which(executable):
            continue
        try:
            proc = subprocess.run(
                [executable, *args],
                input=text,
                text=True,
                encoding="utf-8",
                check=False,
                capture_output=True,
            )
            if proc.returncode == 0:
                return True
        except Exception:
            continue
    return False
