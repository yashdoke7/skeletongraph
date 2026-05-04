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
@click.option("--shadows/--no-shadows", default=True, help="Write shadow files for Claude Code")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def prepare(prompt: str, path: str, mode_override: Optional[str], shadows: bool, quiet: bool):
    """Run the full v3 pipeline and write context files."""
    project_root = Path(path).resolve()
    sg_dir = project_root / ".skeletongraph"

    if not sg_dir.exists():
        console.print("[red]Error:[/red] No .skeletongraph index found. Run `sg build` first.")
        sys.exit(1)

    t0 = time.perf_counter()

    # Late imports to avoid slow startup
    from ..storage.local import load_index
    from ..retrieval.resolver import resolve_context
    from ..retrieval.classifier import classify_query, ContextMode
    from ..assembly.prompt_builder import assemble

    # 1. Load index
    store = load_index(project_root)
    if not store:
        console.print("[red]Error:[/red] Failed to load index.")
        sys.exit(1)

    # 2. Resolve targets
    result = resolve_context(prompt, store)

    # 3. Classify query
    n_files = len({c.skeleton.file_path for c in result.candidates})
    target_fqns = {c.skeleton.fqn for c in result.candidates}
    classification = classify_query(
        intent=result.intent,
        confidence=result.confidence_score,
        target_fqns=target_fqns,
        n_files_involved=n_files,
    )

    # 4. Apply mode override if specified
    if mode_override:
        try:
            classification.mode = ContextMode(mode_override.lower())
        except ValueError:
            console.print(f"[yellow]Warning:[/yellow] Unknown mode '{mode_override}', using auto-selected.")

    # 5. Assemble
    assembled = assemble(
        classification=classification,
        resolver_result=result,
        store=store,
        project_root=project_root,
    )

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # 6. Write context.md
    context_path = sg_dir / "context.md"
    context_path.write_text(assembled.text, encoding="utf-8")

    # 7. Write shadow files (for Claude Code)
    if shadows and assembled.mode not in (ContextMode.PLANNING, ContextMode.REVIEW, ContextMode.PASS_THROUGH):
        shadow_dir = sg_dir / "shadows"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        _write_shadow_files(assembled, store, project_root, shadow_dir)

    # 8. Log to hit_log.jsonl
    eval_dir = sg_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _log_hit(eval_dir / "hit_log.jsonl", prompt, assembled, duration_ms)

    # Output
    if quiet:
        console.print(f"{assembled.mode.value} {assembled.token_count}tok {duration_ms}ms")
    else:
        table = Table(show_header=False, border_style="dim")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Mode", assembled.mode.value.upper())
        table.add_row("Query Type", assembled.query_type.value)
        table.add_row("Confidence", assembled.confidence_level)
        table.add_row("Tokens", str(assembled.token_count))
        table.add_row("Targets", ", ".join(assembled.targets[:3]) or "none")
        table.add_row("Layers", ", ".join(assembled.layers_loaded))
        table.add_row("Modifiers", ", ".join(assembled.modifiers) or "none")
        if assembled.extended_thinking:
            table.add_row("Extended Thinking", "✓ recommended")
        if assembled.reduction_ratio > 0:
            table.add_row("Reduction", f"{assembled.reduction_ratio}x")
        table.add_row("Time", f"{duration_ms}ms")
        if assembled.warning:
            table.add_row("Warning", assembled.warning)
        console.print(Panel(table, title="sg prepare", border_style="blue"))
        console.print(f"  → {context_path}")


def _write_shadow_files(assembled, store, project_root: Path, shadow_dir: Path):
    """Write pre-assembled file views as shadow files."""
    # Collect unique file paths from targets
    files_seen = set()
    for fqn in assembled.targets:
        sk = store.skeleton_table.get(fqn)
        if not sk:
            continue
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


def _log_hit(log_path: Path, prompt: str, assembled, duration_ms: int):
    """Append a hit log entry for passive evaluation."""
    entry = {
        "timestamp": time.time(),
        "prompt": prompt[:200],
        "mode": assembled.mode.value,
        "query_type": assembled.query_type.value,
        "confidence": assembled.confidence_level,
        "tokens": assembled.token_count,
        "targets": assembled.targets[:5],
        "layers": assembled.layers_loaded,
        "modifiers": assembled.modifiers,
        "extended_thinking": assembled.extended_thinking,
        "reduction_ratio": assembled.reduction_ratio,
        "duration_ms": duration_ms,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
