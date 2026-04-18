"""
CLI entry point: `skeletongraph` command.

Commands:
  build    - Index the current project
  update   - Incremental update (only changed files)
  status   - Show index status
  query    - Query the index with a prompt
  serve    - Start MCP server (future)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="skeletongraph")
def app():
    """SkeletonGraph - Token-minimal context assembly for AI coding agents."""
    pass


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def build(path: str):
    """Build the full index for a project."""
    project_root = Path(path).resolve()

    if not project_root.is_dir():
        console.print(f"[red]Error:[/red] {project_root} is not a directory")
        sys.exit(1)

    console.print(f"[bold]> Building index for[/bold] {project_root.name}")

    from ..build import build_index, discover_files

    files = discover_files(project_root)
    console.print(f"  Found [cyan]{len(files)}[/cyan] source files")

    if not files:
        console.print("[yellow]No supported files found. Nothing to index.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing...", total=len(files))

        def on_progress(file_path: str, current: int, total: int):
            progress.update(task, completed=current, description=f"Parsing {file_path}")

        store = build_index(project_root, on_progress=on_progress)

    # Print summary
    console.print()
    table = Table(title="Build Complete", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files indexed", str(store.meta.total_files))
    table.add_row("Functions found", str(store.meta.total_functions))
    table.add_row("Dependency edges", str(store.meta.total_edges))
    table.add_row("Languages", ", ".join(store.meta.languages))
    table.add_row("Build time", f"{store.meta.build_duration_seconds:.2f}s")
    table.add_row("Index location", str(project_root / ".skeletongraph"))
    console.print(table)


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def update(path: str):
    """Incrementally update the index (only changed files)."""
    project_root = Path(path).resolve()

    from ..build import update_index

    start = time.time()
    store = update_index(project_root)
    elapsed = time.time() - start

    console.print(
        f"[green]✓[/green] Updated in {elapsed:.2f}s — "
        f"{store.meta.total_functions} functions, "
        f"{store.meta.total_edges} edges"
    )


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def status(path: str):
    """Show the current index status."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    console.print(f"[bold]> Index Status[/bold]")
    console.print(f"  {store.status_summary()}")


@app.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--budget", "-b", default=128000, help="Model context limit")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def query(prompt: str, path: str, budget: int, verbose: bool):
    """Query the index with a natural language prompt."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index
    from ..retrieval.resolver import resolve_context
    from ..assembly.zone_assembler import assemble_context

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    # Resolve and assemble
    result = resolve_context(prompt, store)
    assembled = assemble_context(result, store, project_root, model_context_limit=budget)

    # Output
    if verbose:
        console.print(f"\n[bold]Intent:[/bold] {result.intent.task_type.value}")
        console.print(f"[bold]Confidence:[/bold] {assembled.confidence}")
        console.print(f"[bold]Reason:[/bold] {assembled.confidence_reason}")
        console.print(f"[bold]Entities:[/bold] {assembled.entities_matched}")
        console.print(f"[bold]Candidates:[/bold] {len(result.candidates)}")

        table = Table(title="Token Budget")
        table.add_column("Zone", style="cyan")
        table.add_column("Tokens", style="green")
        for zone, tokens in assembled.zone_breakdown.items():
            table.add_row(zone, str(tokens))
        table.add_row("Total", str(assembled.token_count), style="bold")
        if assembled.reduction_ratio > 0:
            table.add_row("Reduction", f"{assembled.reduction_ratio}×")
        console.print(table)

        if assembled.warning:
            console.print(f"\n[yellow]⚠ {assembled.warning}[/yellow]")

    console.print(f"\n[dim]--- Assembled Context ({assembled.token_count} tokens) ---[/dim]\n")
    console.print(assembled.text)


if __name__ == "__main__":
    app()
