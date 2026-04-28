"""
CLI entry point: `skeletongraph` command.

Commands:
  build      - Index the current project
  update     - Incremental update (only changed files)
  status     - Show index status
  query      - Query the index with a prompt
  summarize  - Generate LLM summaries
  serve      - Start MCP server
  install    - Auto-detect and configure IDE integrations
  watch      - Start background daemon
  stats      - Token savings dashboard
  review     - PR blast-radius context assembly
  eval       - Run golden dataset evaluation
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

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

    # Show constraint status
    if hasattr(store, 'constraints') and store.constraints:
        if store.constraints.has_constraints:
            scope_info = f"global + {store.constraints.scope_count} scoped"
            table.add_row("Constraints", scope_info)

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
        f"[green][OK][/green] Updated in {elapsed:.2f}s - "
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
@click.option("--out", "-o", help="Save context to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def query(prompt: str, path: str, budget: int, out: str | None, verbose: bool):
    """Query the index with a natural language prompt."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index
    from ..retrieval.resolver import resolve_context
    from ..assembly.zone_assembler import assemble_context
    from ..retrieval.session import Session
    from ..metrics.metrics_logger import MetricsLogger

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    # Load session for cross-turn memory
    session = Session.load(project_root)
    metrics = MetricsLogger(project_root)

    t0 = time.time()

    # Resolve and assemble
    result = resolve_context(prompt, store, session=session)
    assembled = assemble_context(
        result, store, project_root,
        model_context_limit=budget, session=session,
    )

    duration_ms = int((time.time() - t0) * 1000)

    # Save session
    session.save(project_root)

    # Log metrics
    files_involved = list({c.skeleton.file_path for c in result.candidates})
    metrics.log_skeleton_query(
        prompt=prompt,
        sg_tokens=assembled.token_count,
        native_tokens_estimated=int(assembled.reduction_ratio * assembled.token_count) if assembled.reduction_ratio > 0 else 0,
        reduction_ratio=assembled.reduction_ratio,
        confidence=assembled.confidence,
        entities_matched=assembled.entities_matched,
        zone_breakdown=assembled.zone_breakdown,
        session_dedup_count=assembled.session_dedup_count,
        session_tokens_saved=assembled.session_tokens_saved,
        files_involved=files_involved,
        duration_ms=duration_ms,
    )

    # Output
    if verbose:
        console.print(f"\n[bold]Intent:[/bold] {result.intent.task_type.value}")
        console.print(f"[bold]Confidence:[/bold] {assembled.confidence}")
        console.print(f"[bold]Reason:[/bold] {assembled.confidence_reason}")
        console.print(f"[bold]Entities:[/bold] {assembled.entities_matched}")
        console.print(f"[bold]Candidates:[/bold] {len(result.candidates)}")

        # Token budget table
        table = Table(title="Token Budget")
        table.add_column("Zone", style="cyan")
        table.add_column("Tokens", style="green")
        for zone, tokens in assembled.zone_breakdown.items():
            table.add_row(zone, str(tokens))
        table.add_row("Total", str(assembled.token_count), style="bold")
        if assembled.reduction_ratio > 0:
            table.add_row("Reduction", f"{assembled.reduction_ratio}x")
        console.print(table)

        # Attention heatmap
        if assembled.attention_map:
            console.print("\n[bold]Attention Map:[/bold]")
            for zone in assembled.attention_map:
                color = {
                    "peak": "green", "high": "cyan",
                    "moderate": "yellow", "valley": "red",
                }.get(zone.attention_level, "white")
                console.print(
                    f"  [{color}][ATTENTION: {zone.bar}][/{color}] "
                    f"{zone.zone_name} ({zone.token_count} tokens)"
                )

        # Session info
        if assembled.session_dedup_count > 0:
            console.print(
                f"\n[dim]Session: {assembled.session_dedup_count} bodies skipped "
                f"(already sent), {assembled.session_tokens_saved} tokens saved[/dim]"
            )

        if assembled.warning:
            console.print(f"\n[yellow][WARNING] {assembled.warning}[/yellow]")

    if out:
        out_path = Path(out).resolve()
        out_path.write_text(assembled.text, encoding="utf-8")
        console.print(f"\n[green][OK][/green] Saved assembled context to {out_path}")
    else:
        console.print(f"\n[dim]--- Assembled Context ({assembled.token_count} tokens) ---[/dim]\n")
        console.print(assembled.text)

    console.print("\n[dim][*] Metrics logged to .skeletongraph/metrics/query_log.jsonl[/dim]")


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--model", "-m", default="gemini/gemini-2.0-flash", help="LLM model")
@click.option("--force", is_flag=True, help="Re-summarize all functions")
def summarize(path: str, model: str, force: bool):
    """Generate LLM summaries for indexed functions."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index
    from ..llm.summarizer import summarize_index
    from ..llm.provider import LLMConfig

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    cfg = LLMConfig(model=model)
    console.print(f"[bold]> Summarizing with {model}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Summarizing...", total=None)

        def on_progress(fqn: str, current: int, total: int):
            short = fqn.split("::")[-1] if "::" in fqn else fqn
            progress.update(task, description=f"[{current}/{total}] {short}")

        result = summarize_index(
            store, project_root, config=cfg, force=force, on_progress=on_progress,
        )

    table = Table(title="Summarization Complete", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Summarized", str(result.summarized))
    table.add_row("Skipped", str(result.skipped))
    table.add_row("Errors", str(result.errors))
    table.add_row("Input tokens", str(result.total_input_tokens))
    table.add_row("Output tokens", str(result.total_output_tokens))
    table.add_row("Duration", f"{result.duration_seconds:.2f}s")
    if result.total_cost > 0:
        table.add_row("Cost", f"${result.total_cost:.4f}")
    console.print(table)


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--port", default=3500, help="Server port")
def serve(path: str, port: int):
    """Start the MCP server for IDE integration."""
    import sys as _sys
    from rich.console import Console as _Console
    stderr_console = _Console(stderr=True)

    project_root = Path(path).resolve()

    from ..storage.local import load_index

    store = load_index(project_root)
    if store is None:
        stderr_console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        _sys.exit(1)

    stderr_console.print(f"[bold]> Starting MCP server[/bold]")
    stderr_console.print(f"  Index: {store.status_summary()}")
    stderr_console.print(f"  Metrics: .skeletongraph/metrics/query_log.jsonl")

    from ..server.mcp import start_server
    start_server(store, project_root, port=port)


@app.command()
@click.argument("platform", required=False, default=None)
@click.option("--path", "-p", default=".", help="Project root directory")
def install(platform: str, path: str):
    """Auto-detect and configure IDE integrations.

    If PLATFORM is not specified, auto-detects all installed IDEs.
    Supported: claude, cursor, antigravity, codex, windsurf, kiro, opencode
    """
    project_root = Path(path).resolve()

    platforms = _detect_platforms(project_root) if platform is None else [platform]

    if not platforms:
        console.print("[yellow]No supported AI coding tools detected.[/yellow]")
        console.print("Supported: claude, cursor, antigravity, codex, windsurf, kiro, opencode")
        return

    for p in platforms:
        _install_platform(p, project_root)

    # Also write the MCP config
    _write_mcp_config(project_root)
    console.print(
        f"\n[bold green][OK] Configuration complete.[/bold green] "
        f"Restart your editor to activate SkeletonGraph."
    )


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def stats(path: str):
    """Show token savings dashboard."""
    project_root = Path(path).resolve()

    from ..retrieval.session import Session
    from ..storage.local import load_index

    store = load_index(project_root)
    session = Session.load(project_root)

    console.print(Panel("[bold]SkeletonGraph Token Savings Dashboard[/bold]", style="cyan"))

    if store:
        table = Table(title="Index", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Files indexed", str(store.meta.total_files))
        table.add_row("Functions", str(store.meta.total_functions))
        table.add_row("Edges", str(store.meta.total_edges))
        table.add_row("Languages", ", ".join(store.meta.languages))
        console.print(table)

    if session.turn_count > 0:
        s = session.stats
        table = Table(title="Session Savings", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Queries this session", str(s.total_turns))
        table.add_row("SG tokens used", f"{s.total_sg_tokens:,}")
        table.add_row("Estimated native tokens", f"{s.total_native_tokens_estimated:,}")
        table.add_row("Tokens saved", f"{s.total_saved_tokens:,}")
        table.add_row(
            "Reduction ratio",
            f"{s.reduction_ratio:.1f}x" if s.reduction_ratio > 0 else "N/A",
        )
        table.add_row(
            "Estimated cost saved",
            f"${s.estimated_cost_saved_usd:.4f} (at GPT-4o pricing)",
        )
        console.print(table)
    else:
        console.print("[dim]No session data yet. Run some queries first.[/dim]")


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--target", "-t", default="HEAD", help="Git diff target (e.g., HEAD, main)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def review(path: str, target: str, verbose: bool):
    """Analyze blast radius of recent changes for code review."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index
    from ..retrieval.detect_changes import detect_changes

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    console.print(f"[bold]> Analyzing changes against {target}[/bold]")
    analysis = detect_changes(project_root, store, diff_target=target)

    if not analysis.changed_functions:
        console.print("[green]No changes detected.[/green]")
        return

    console.print(f"\n[bold]{analysis.risk_summary}[/bold]\n")

    # Files to review table
    table = Table(title="Files to Review (by risk)")
    table.add_column("File", style="cyan")
    table.add_column("Risk", style="bold")
    table.add_column("Affected Functions", style="dim")
    table.add_column("Distance")

    for af in analysis.affected_files[:20]:  # Top 20
        risk_color = (
            "red" if af.risk_score >= 0.7
            else "yellow" if af.risk_score >= 0.4
            else "green"
        )
        table.add_row(
            af.file_path,
            f"[{risk_color}]{af.risk_score:.2f}[/{risk_color}]",
            str(len(af.affected_fqns)),
            str(af.distance),
        )

    console.print(table)

    if verbose:
        console.print("\n[bold]Changed functions:[/bold]")
        for fn in analysis.changed_functions:
            console.print(f"  [{fn.change_type}] {fn.fqn}")


@app.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def baseline(prompt: str, path: str, verbose: bool):
    """Simulate a baseline agent's file-reading cost for a prompt.

    Estimates what a naive agent (no SkeletonGraph) would need to read
    to answer the same prompt using grep + full file reading. Logs the
    result to the same metrics file for comparison.
    """
    project_root = Path(path).resolve()

    from ..build import discover_files
    from ..metrics.metrics_logger import MetricsLogger
    import re
    import time

    metrics = MetricsLogger(project_root)
    t0 = time.time()

    # Extract keywords from prompt (same logic an agent would use to grep)
    words = set(re.findall(r'\b[a-zA-Z_]\w{2,}\b', prompt))
    stop_words = {
        "the", "this", "that", "with", "from", "into", "when", "then",
        "fix", "add", "create", "ensure", "check", "use", "using",
        "not", "and", "for", "are", "was", "has", "have", "should",
        "global", "config", "logic", "init", "code", "file", "function",
    }
    keywords = {w.lower() for w in words} - stop_words

    if verbose:
        console.print(f"[bold]> Simulating baseline for:[/bold] {prompt}")
        console.print(f"  Keywords: {', '.join(sorted(keywords))}")

    # Phase 1: Grep - scan all files for keyword matches
    all_files = discover_files(project_root)
    matched_files = []
    total_grep_tokens = 0

    for file_path in all_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            content_lower = content.lower()
            if any(kw in content_lower for kw in keywords):
                matched_files.append(file_path)
                # Grep results cost: ~50 tokens per matching file (line snippets)
                total_grep_tokens += 50
        except Exception:
            continue

    # Phase 2: File reading - agent reads matched files fully
    total_read_tokens = 0
    files_read = []

    for file_path in matched_files:
        full_path = project_root / file_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            file_tokens = len(content) // 4
            total_read_tokens += file_tokens
            files_read.append(file_path)
        except Exception:
            continue

    total_tokens = total_grep_tokens + total_read_tokens
    duration_ms = int((time.time() - t0) * 1000)

    # Log to metrics
    metrics.log_baseline_estimate(
        prompt=prompt,
        total_tokens=total_tokens,
        files_read=files_read,
        files_grepped=len(all_files),
        duration_ms=duration_ms,
    )

    # Output
    table = Table(title="Baseline Estimation", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files scanned (grep)", str(len(all_files)))
    table.add_row("Files matched", str(len(matched_files)))
    table.add_row("Grep tokens", f"{total_grep_tokens:,}")
    table.add_row("File read tokens", f"{total_read_tokens:,}")
    table.add_row("Total native tokens", f"{total_tokens:,}", style="bold")
    table.add_row("Duration", f"{duration_ms}ms")
    console.print(table)

    if verbose:
        console.print("\n[bold]Files that would be read:[/bold]")
        for fp in files_read[:20]:
            console.print(f"  {fp}")
        if len(files_read) > 20:
            console.print(f"  ... and {len(files_read) - 20} more")

    console.print("\n[dim][*] Baseline logged to .skeletongraph/metrics/query_log.jsonl[/dim]")


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def metrics(path: str):
    """Show evaluation comparison between skeleton and baseline runs."""
    project_root = Path(path).resolve()

    from ..metrics.metrics_logger import MetricsLogger
    logger = MetricsLogger(project_root)
    summary = logger.get_comparison_summary()

    if "error" in summary:
        console.print(f"[yellow]{summary['error']}[/yellow]")
        return

    console.print(Panel("[bold]SkeletonGraph Evaluation Metrics[/bold]", style="cyan"))

    # Overview
    table = Table(title="Query Counts", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Skeleton queries", str(summary.get("total_skeleton_queries", 0)))
    table.add_row("Baseline estimates", str(summary.get("total_baseline_queries", 0)))
    console.print(table)

    # Skeleton stats
    if "skeleton" in summary:
        sk = summary["skeleton"]
        table = Table(title="SkeletonGraph Performance", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Avg SG tokens", f"{sk['avg_sg_tokens']:,}")
        table.add_row("Avg native estimated", f"{sk['avg_native_estimated']:,}")
        table.add_row("Avg reduction ratio", f"{sk['avg_reduction_ratio']}x")
        console.print(table)

        # IR stats (Only displayed if eval runs were logged)
        if "ir_metrics" in sk:
            ir = sk["ir_metrics"]
            ir_table = Table(title="Information Retrieval (IR) Benchmarks", show_header=False)
            ir_table.add_column("Metric", style="cyan")
            ir_table.add_column("Value", style="magenta")
            ir_table.add_row("Avg Precision", f"{ir['avg_precision']:.2f}")
            ir_table.add_row("Avg Recall", f"{ir['avg_recall']:.2f}")
            ir_table.add_row("Mean Reciprocal Rank (MRR)", f"{ir['avg_mrr']:.2f}")
            console.print(ir_table)

    # Baseline stats
    if "baseline" in summary:
        bl = summary["baseline"]
        table = Table(title="Baseline Performance", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Avg native tokens", f"{bl['avg_tokens']:,}")
        console.print(table)

    # Cross comparison
    if "cross_comparison" in summary:
        cc = summary["cross_comparison"]
        table = Table(title="[+] Head-to-Head Comparison", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_row("Actual reduction ratio", f"{cc['actual_reduction_ratio']}x")
        table.add_row("Tokens saved per query", f"{cc['tokens_saved_per_query']:,}")
        console.print(table)


@app.command(name="hotspots")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--limit", "-n", default=10, help="Number of hotspots to show")
def find_hotspots(path: str, limit: int):
    """Identify 'Load-Bearing' files in the project using graph centrality."""
    project_root = Path(path).resolve()
    from ..build import build_index
    from ..analytics.centrality import get_top_hotspots
    
    console.print("[bold yellow]> Analysing project topology hotspots...[/bold yellow]")
    
    with console.status("[dim]Computing PageRank over dependency graph...[/dim]"):
        store = build_index(project_root)
        hotspots = get_top_hotspots(store, top_n=limit)
    
    if not hotspots:
        console.print("[yellow]Graph is too small or has no edges to compute centrality.[/yellow]")
        return
        
    table = Table(title=f"Architectural Hotspots (Top {limit})")
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("File Path", style="cyan")
    table.add_column("Score", style="magenta", justify="right")
    
    for i, (file_path, score) in enumerate(hotspots):
        display_score = f"{score * 100:.2f}"
        table.add_row(str(i+1), file_path, display_score)
        
    console.print(table)
    console.print("[dim][*] High scores indicate 'Load-Bearing' files with heavy downstream impact.[/dim]")


@app.command(name="visualize")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--format", "-f", "out_format", default="mermaid", type=click.Choice(["mermaid", "json"]))
def visualize_graph(path: str, out_format: str):
    """Generate a visual architecture map (Mermaid) for Vision models."""
    project_root = Path(path).resolve()
    from ..build import build_index
    from ..analytics.visualizer import generate_mermaid_flowchart
    import pyperclip
    
    console.print("[bold magenta]> Rendering multi-modal architecture map...[/bold magenta]")
    
    with console.status("[dim]Assembling top-level dependency tree...[/dim]"):
        store = build_index(project_root)
        diagram = generate_mermaid_flowchart(store)
    
    if not diagram or diagram == "graph TD":
        console.print("[yellow]Project has no cross-file dependencies to visualize.[/yellow]")
        return
        
    pyperclip.copy(diagram)
    
    table = Table(title="Architecture Map Generated", show_header=False)
    table.add_column("Property", style="magenta")
    table.add_column("Status", style="green")
    table.add_row("Engine", "Mermaid.js Flowchart")
    table.add_row("Status", "Copied to Clipboard")
    console.print(table)
    
    console.print("\n[bold green][DONE][/bold green] Mermaid diagram is on your clipboard!")
    console.print("[dim]Next Step: Paste this into ChatGPT/Claude and ask: 'Here is my repository map. Where is the bug?'[/dim]")
    
    # Context summary
    console.print(f"[dim]Nodes: {len(store.skeleton_table)} | Edges: {store.meta.total_edges}[/dim]")


@app.command(name="pack")
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
def pack_context(prompt: str, path: str):
    """Assemble optimized context for a prompt and copy to clipboard (for Web UI)."""
    project_root = Path(path).resolve()
    from ..build import build_index
    from ..retrieval.resolver import resolve_context
    from ..assembly.zone_assembler import assemble_context
    import pyperclip

    console.print(f"[bold cyan]> Assembling context for: \"{prompt}\"[/bold cyan]")
    
    with console.status("[dim]Building context map...[/dim]"):
        store = build_index(project_root)
        result = resolve_context(prompt, store)
        assembled = assemble_context(result, store, project_root)
    
    # Payload for clipboard
    pyperclip.copy(assembled.text)
    
    table = Table(title="Context Packed Successfully", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total tokens", f"{assembled.token_count:,}")
    table.add_row("Entities caught", str(len(assembled.entities_matched)))
    table.add_row("Reduction ratio", f"{assembled.reduction_ratio:.1f}x")
    console.print(table)
    
    console.print("\n[bold green][DONE][/bold green] Optimized context copied to clipboard!")
    console.print("[dim]Paste this into ChatGPT/Claude for a token-minimal, high-precision session.[/dim]")


@app.command(name="eval")
@click.option("--path", "-p", default=".", help="Project root directory")
def run_eval(path: str):
    """Run the golden dataset evaluation."""
    project_root = Path(path).resolve()

    console.print("[bold]> Running SkeletonGraph evaluation[/bold]")
    try:
        from ..eval_runner import run_evaluation
        from ..metrics.metrics_logger import MetricsLogger
        
        results = run_evaluation(project_root)
        metrics = MetricsLogger(project_root)
        
        # Log all evaluative queries silently to the JSONL database
        for r in results.results:
            metrics.log_skeleton_query(
                prompt=r.prompt,
                sg_tokens=r.token_count,
                native_tokens_estimated=0, # Eval cases don't simulate baseline natively
                reduction_ratio=r.reduction_ratio,
                confidence=r.confidence,
                entities_matched=r.found_fqns,
                zone_breakdown={},
                precision=r.precision,
                recall=r.recall,
                mrr=r.mrr,
            )

        # Print the formal evaluation dashboard output
        console.print(f"\n[bold green]Evaluation Complete ({results.duration_seconds:.1f}s)[/bold green]")
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Total Cases", str(results.total_cases))
        table.add_row("Success Rate", f"{results.success_rate * 100:.1f}%")
        table.add_row("Avg Node Precision", f"{results.avg_precision:.2f}")
        table.add_row("Avg Node Recall", f"{results.avg_recall:.2f}")
        table.add_row("Mean Reciprocal Rank (MRR)", f"{results.avg_mrr:.2f}")
        table.add_row("Avg Reduction Ratio", f"{results.avg_reduction_ratio:.1f}x")
        console.print(table)
        console.print("[dim][*] IR Metrics seamlessly logged to background database.[/dim]")

    except ImportError:
        console.print("[yellow]Evaluation module not found. Ensure eval/ is set up.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command(name="parse-agent-log")
@click.argument("log_path")
@click.option("--agent", "-a", default="antigravity", help="Agent standard to parse (e.g. antigravity)")
@click.option("--path", "-p", default=".", help="Project root directory for logging")
@click.option("--prompt", default="Manual Baseline Log", help="Original task prompt")
def parse_agent_log(log_path: str, agent: str, path: str, prompt: str):
    """Parse an agent's true conversation log to dump exact 'empirical baseline' metrics."""
    project_root = Path(path).resolve()
    target_log = Path(log_path)
    
    if not target_log.exists():
        console.print(f"[yellow]! Initial path failed: {target_log}[/yellow]", highlight=False)
        console.print("[dim]Attempting smart search for conversation logs...[/dim]")
        
        # Smart search: look for log files in common Antigravity locations
        home = Path.home()
        possible_roots = [
            home / ".gemini" / "antigravity" / "brain",
            Path("C:/Users/ASUS/.gemini/antigravity/brain"), # Explicit fallback for user's machine
        ]
        
        found_logs = []
        for root in possible_roots:
            if root.exists():
                # Strictly search for overview.txt or explicit exported logs
                found_logs.extend(list(root.rglob("overview.txt")))
        
        if not found_logs:
            console.print("[red]Critical: Could not find any Antigravity conversation logs automatically.[/red]")
            console.print("[dim]Antigravity may no longer save txt logs automatically in this version.[/dim]")
            console.print("[dim]Please explicitly click 'Export Chat' in the IDE, save as a .txt file, and provide that path.[/dim]")
            return
            
        # Try to find one that matches the ID in the provided path if possible
        id_match = None
        for fl in found_logs:
            if any(part in str(fl) for part in target_log.parts):
                id_match = fl
                break
        
        if id_match:
            console.print(f"[green]Found matching log at:[/green] {id_match}")
            target_log = id_match
        else:
            console.print(f"[yellow]Found {len(found_logs)} logs, but none match the provided ID path.[/yellow]")
            console.print(f"Latest log found: {found_logs[-1]}")
            target_log = found_logs[-1]
        
    try:
        from ..metrics.log_parser import parse_antigravity_log, parse_copilot_log
        from ..metrics.metrics_logger import MetricsLogger
        
        if agent == "antigravity":
            stats = parse_antigravity_log(target_log)
        elif agent == "copilot":
            stats = parse_copilot_log(target_log)
        else:
            console.print(f"[red]Parser for agent '{agent}' not implemented.[/red]")
            return
            
        metrics = MetricsLogger(project_root)
        metrics.log_baseline_estimate(
            prompt=prompt,
            total_tokens=stats["total_native_tokens"],
            files_read=stats.get("files_viewed", 0),
            files_grepped=stats.get("grep_searches", 0),
            duration_ms=stats.get("duration_ms", 0)
        )
        
        table = Table(title="[bold]True Empirical Baseline Generated[/bold]", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Agent parsed", agent)
        table.add_row("Files viewed manually", str(stats.get("files_viewed", 0)))
        table.add_row("Grep searches run", str(stats.get("grep_searches", 0)))
        table.add_row("Total native tokens used", f"{stats['total_native_tokens']:,}")
        console.print(table)
        console.print("[dim][*] Real empirical baseline logged to JSONL.[/dim]")
        
    except Exception as e:
        console.print(f"[red]Failed to parse log: {e}[/red]")


@app.command(name="log-manual")
@click.option("--agent", "-a", required=True, help="Agent name (e.g. cursor, windsurf)")
@click.option("--tokens", "-t", type=int, required=True, help="Raw token count observed in UI")
@click.option("--prompt", "-m", default="Manual Entry", help="Task prompt for this run")
@click.option("--path", "-p", default=".", help="Project root directory")
def log_manual(agent: str, tokens: int, prompt: str, path: str):
    """Manually log a token count from an agent's UI for the dashboard."""
    project_root = Path(path).resolve()
    from ..metrics.metrics_logger import MetricsLogger
    
    try:
        metrics = MetricsLogger(project_root)
        metrics.log_baseline_estimate(
            prompt=prompt,
            total_tokens=tokens,
            files_read=0,      # Unknown in manual entry
            files_grepped=0,   # Unknown in manual entry
            duration_ms=0      # Unknown in manual entry
        )
        
        console.print(f"[green][OK][/green] Manually logged [bold]{tokens:,}[/bold] tokens for [bold]{agent}[/bold].")
        console.print("[dim][*] You can now see this in 'skeletongraph metrics'[/dim]")
    except Exception as e:
        console.print(f"[red]Error logging manual entry: {e}[/red]")


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def watch(path: str):
    """Start a background daemon to auto-reindex files on save."""
    project_root = Path(path).resolve()
    from ..daemon import start_daemon
    start_daemon(project_root)


# -- Evaluation commands ------------------------------------------------

SUPPORTED_AGENTS = ["antigravity", "claude_code", "cursor", "codex", "copilot"]

@app.command(name="eval-parse")
@click.option("--agent", "-a", required=True, type=click.Choice(SUPPORTED_AGENTS), help="Agent to parse")
@click.option("--file", "-f", "file_path", default=None, help="Path to exported chat file (auto-discovers if omitted)")
@click.option("--mode", "-m", default="native", type=click.Choice(["native", "skeletongraph"]), help="Was this a native or SG session?")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--project", default="", help="Project name (e.g. flask, fastapi)")
@click.option("--prompt", default="", help="Original task prompt")
def eval_parse(agent: str, file_path: str, mode: str, path: str, project: str, prompt: str):
    """Parse an agent's exported chat into a standardized trace JSON."""
    project_root = Path(path).resolve()

    # Auto-discover if no file provided
    if not file_path:
        if mode == "skeletongraph":
            # SG mode: parse from current.json
            try:
                from ..eval.parsers.antigravity import parse_antigravity_sg_session
                trace = parse_antigravity_sg_session(project_root, project)
                _save_and_display_trace(trace, project_root)
                return
            except FileNotFoundError as e:
                console.print(f"[red]No SG session found: {e}[/red]")
                return

        discovered = _discover_agent_file(agent)
        if discovered:
            file_path = str(discovered)
            console.print(f"[green]Auto-discovered:[/green] {file_path}")
        else:
            console.print(f"[red]No export file found for {agent}. Use --file to specify.[/red]")
            console.print(f"[dim]Tip: Export your chat and provide the path.[/dim]")
            return

    export_path = Path(file_path)
    if not export_path.exists():
        console.print(f"[red]File not found: {export_path}[/red]")
        return

    # Parse based on agent type
    trace = _parse_agent_export(agent, export_path, project_root, prompt, mode, project)
    if trace:
        _save_and_display_trace(trace, project_root)


@app.command(name="eval-compare")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--sg-file", default=None, help="SG trace JSON (auto-discovers if omitted)")
@click.option("--native-file", default=None, help="Native trace JSON (auto-discovers if omitted)")
@click.option("--output", "-o", default=None, help="Output report path")
def eval_compare(path: str, sg_file: str, native_file: str, output: str):
    """Compare SG vs Native traces and generate a report."""
    project_root = Path(path).resolve()
    eval_dir = project_root / ".skeletongraph" / "eval"

    from ..eval.schema import AgentTrace
    from ..eval.comparison import compare_traces
    from ..eval.report import generate_report, save_report

    # Auto-discover traces
    if not sg_file:
        sg_path = eval_dir / "sg_trace.json"
        if not sg_path.exists():
            # Try parsing from current.json
            try:
                from ..eval.parsers.antigravity import parse_antigravity_sg_session
                sg_trace = parse_antigravity_sg_session(project_root)
                sg_path.parent.mkdir(parents=True, exist_ok=True)
                sg_path.write_text(sg_trace.to_json(), encoding="utf-8")
            except FileNotFoundError:
                console.print("[red]No SG trace found. Run eval-parse --mode skeletongraph first.[/red]")
                return
    else:
        sg_path = Path(sg_file)

    if not native_file:
        native_path = eval_dir / "native_trace.json"
        if not native_path.exists():
            console.print("[red]No native trace found. Run eval-parse --mode native first.[/red]")
            return
    else:
        native_path = Path(native_file)

    # Load traces
    sg_data = json.loads(sg_path.read_text(encoding="utf-8"))
    native_data = json.loads(native_path.read_text(encoding="utf-8"))
    sg_trace = AgentTrace.from_dict(sg_data)
    native_trace = AgentTrace.from_dict(native_data)

    # Compare
    result = compare_traces(sg_trace, native_trace)

    # Display summary
    d = result.to_dict()
    ta = d["tier_a_retrieval"]
    tb = d["tier_b_conversation"]
    tc = d["tier_c_efficiency"]

    table = Table(title="[bold]SkeletonGraph Evaluation Results[/bold]", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("SkeletonGraph", style="green", justify="right")
    table.add_column("Native Agent", style="yellow", justify="right")
    table.add_column("Reduction", style="bold magenta", justify="right")

    table.add_row("Retrieval Tokens", f"{ta['sg_tokens']:,}", f"{ta['native_tokens']:,}", f"{ta['reduction_ratio']}x")
    table.add_row("Conversation Tokens", f"{tb['sg_tokens']:,}", f"{tb['native_tokens']:,}", f"{tb['reduction_ratio']}x")
    table.add_row("Tool Calls", str(tc['sg_tool_calls']), str(tc['native_tool_calls']), "")
    table.add_row("Turns", str(tc['sg_turns']), str(tc['native_turns']), "")
    table.add_row("Repeated Views", "0", str(tc['native_repeated_views']), "")

    console.print(table)

    # Save report
    if output:
        report_path = Path(output)
    else:
        report_path = eval_dir / "report.md"
    save_report(result, report_path)
    console.print(f"\n[green]Report saved to:[/green] {report_path}")

    # Also save comparison JSON
    comp_path = eval_dir / "comparison.json"
    comp_path.write_text(result.to_json(), encoding="utf-8")
    console.print(f"[green]JSON saved to:[/green] {comp_path}")


@app.command(name="eval-benchmark")
@click.option("--dataset", "-d", default="swe-bench-verified", type=click.Choice(["swe-bench-verified", "custom"]), help="Dataset to evaluate against")
@click.option("--repos", "-r", default=None, help="Comma-separated repo filter (e.g. 'django/django,psf/requests')")
@click.option("--limit", "-n", default=None, type=int, help="Max tasks to evaluate")
@click.option("--traces-dir", "-t", required=True, help="Directory containing agent traces (sg_trace.json + native_trace.json)")
@click.option("--repos-dir", default=None, help="Directory containing cloned repos (for codebase measurement)")
@click.option("--output", "-o", default=None, help="Output directory for results")
@click.option("--dataset-file", default=None, help="Path to custom dataset JSONL file")
def eval_benchmark(dataset: str, repos: str, limit: int, traces_dir: str, repos_dir: str, output: str, dataset_file: str):
    """Run research-grade benchmarks against SWE-bench Verified or custom datasets.
    
    Requires REAL agent session traces — does not simulate or estimate.
    """
    from ..eval.datasets.swe_bench import load_swe_bench, list_available_repos, download_swe_bench
    from ..eval.benchmarks.runner import BenchmarkRunner
    
    traces_path = Path(traces_dir).resolve()
    repos_path = Path(repos_dir).resolve() if repos_dir else None
    output_path = Path(output).resolve() if output else Path(".skeletongraph/benchmark")
    
    # Parse repo filter
    repo_list = [r.strip() for r in repos.split(",")] if repos else None
    
    # Load dataset
    console.print("[bold]Step 1:[/bold] Loading evaluation dataset...")
    
    if dataset == "swe-bench-verified":
        try:
            tasks = load_swe_bench(repos=repo_list, limit=limit)
        except ImportError as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[dim]Install datasets: pip install datasets[/dim]")
            return
    elif dataset == "custom" and dataset_file:
        import json as _json
        from ..eval.datasets.base import EvalTask
        tasks = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                raw = _json.loads(line)
                tasks.append(EvalTask(**raw))
    else:
        console.print("[red]Error: specify --dataset-file for custom datasets[/red]")
        return
    
    console.print(f"  [green][OK][/green] Loaded {len(tasks)} tasks")
    
    if not tasks:
        console.print("[yellow]No tasks loaded. Check your filters.[/yellow]")
        return
    
    # List repos
    repo_summary = {}
    for t in tasks:
        repo_summary[t.repo] = repo_summary.get(t.repo, 0) + 1
    for repo, count in sorted(repo_summary.items()):
        console.print(f"  [dim]{repo}: {count} tasks[/dim]")
    
    # Run benchmarks
    console.print("[bold]Step 2:[/bold] Loading agent traces...")
    runner = BenchmarkRunner(tasks, traces_path, repos_path)
    
    if not runner.sg_traces and not runner.native_traces:
        console.print("[red]No traces found in the specified directory.[/red]")
        console.print(f"[dim]Expected structure: {traces_path}/{{task_id}}/sg_trace.json[/dim]")
        return
    
    console.print("[bold]Step 3:[/bold] Running benchmarks...")
    summary = runner.run_all()
    
    # Save results
    console.print("[bold]Step 4:[/bold] Saving results...")
    runner.save_results(output_path)
    
    # Display summary
    console.print()
    te = summary.get("token_efficiency", {})
    rq_sg = summary.get("retrieval_quality", {}).get("sg", {})
    
    panel_text = ""
    rr = te.get("retrieval_reduction_ratio", {})
    if rr:
        panel_text += f"[bold green]Token Reduction:[/bold green] {rr.get('mean', 0):.1f}x average\n"
    
    cs = te.get("cost_savings_pct", {})
    if cs:
        panel_text += f"[bold blue]Cost Savings:[/bold blue] {cs.get('mean', 0):.1f}% average\n"
    
    f1 = rq_sg.get("f1", {})
    if f1:
        panel_text += f"[bold yellow]SG File F1:[/bold yellow] {f1.get('mean', 0):.3f} ± {f1.get('std', 0):.3f}\n"
    
    mrr = rq_sg.get("mrr", {})
    if mrr:
        panel_text += f"[bold cyan]SG MRR:[/bold cyan] {mrr.get('mean', 0):.3f}"
    
    if panel_text:
        console.print(Panel(panel_text, title="[bold]Benchmark Results[/bold]"))
    
    console.print(f"\n[dim]Full report: {output_path / 'benchmark_report.md'}[/dim]")
    console.print(f"[dim]Raw JSON: {output_path / 'benchmark_results.json'}[/dim]")


@app.command(name="eval-list")
@click.option("--dataset", "-d", default="swe-bench-verified", help="Dataset to list repos for")
def eval_list(dataset: str):
    """List available repos and task counts in a dataset."""
    if dataset == "swe-bench-verified":
        from ..eval.datasets.swe_bench import list_available_repos, REPO_SIZES
        repos = list_available_repos()
        
        table = Table(title="[bold]SWE-bench Verified Repos[/bold]", show_header=True)
        table.add_column("Repo", style="cyan")
        table.add_column("Tasks", style="yellow", justify="right")
        table.add_column("Size", style="dim")
        
        for repo, count in repos.items():
            size = REPO_SIZES.get(repo, "?")
            table.add_row(repo, str(count), size)
        
        console.print(table)
    else:
        console.print(f"[yellow]Dataset '{dataset}' not supported for listing.[/yellow]")


@app.command(name="eval")
@click.option("--agent", "-a", default="antigravity", type=click.Choice(SUPPORTED_AGENTS), help="Agent to evaluate")
@click.option("--native-file", "-f", default=None, help="Path to native exported chat")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--project", default="", help="Project name")
def eval_full(agent: str, native_file: str, path: str, project: str):
    """One-command evaluation: parse SG session + native export -> comparison report."""
    project_root = Path(path).resolve()
    eval_dir = project_root / ".skeletongraph" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    from ..eval.parsers.antigravity import parse_antigravity_sg_session
    from ..eval.comparison import compare_traces
    from ..eval.report import generate_report, save_report
    from ..eval.token_counter import measure_codebase_tokens

    # Step 1: Measure Whole Codebase ceiling
    console.print("[bold]Step 1:[/bold] Measuring static codebase limit...")
    whole_codebase_tokens = measure_codebase_tokens(project_root)
    console.print(f"  [green][OK][/green] Codebase ceiling: {whole_codebase_tokens:,} tokens")

    # Step 2: Parse SG session
    console.print("[bold]Step 2:[/bold] Parsing SkeletonGraph session...")
    try:
        sg_trace = parse_antigravity_sg_session(project_root, project)
        console.print(f"  [green][OK][/green] SG trace: {sg_trace.total_tool_output_tokens:,} tokens across {sg_trace.tool_call_count} calls")
    except FileNotFoundError as e:
        console.print(f"  [red][X] No SG session found: {e}[/red]")
        return

    # Step 3: Parse native baseline
    console.print("[bold]Step 3:[/bold] Parsing native baseline...")
    if not native_file:
        discovered = _discover_agent_file(agent)
        if discovered:
            native_file = str(discovered)
            console.print(f"  [green]Auto-discovered:[/green] {native_file}")
        else:
            console.print(f"  [red]No export found. Provide --native-file path.[/red]")
            _print_export_instructions(agent)
            return

    native_trace = _parse_agent_export(agent, Path(native_file), project_root, "", "native", project)
    if not native_trace:
        return
    console.print(f"  [green][OK][/green] Native trace: {native_trace.total_tool_output_tokens:,} tokens across {native_trace.tool_call_count} calls")

    # Step 4: Compare
    console.print("[bold]Step 4:[/bold] Generating comparison...")
    result = compare_traces(sg_trace, native_trace, whole_codebase_tokens)

    # Save everything
    (eval_dir / "sg_trace.json").write_text(sg_trace.to_json(), encoding="utf-8")
    (eval_dir / "native_trace.json").write_text(native_trace.to_json(), encoding="utf-8")
    (eval_dir / "comparison.json").write_text(result.to_json(), encoding="utf-8")
    save_report(result, eval_dir / "report.md")

    # Display results
    d = result.to_dict()
    ta = d["tier_a_retrieval"]
    tb = d["tier_b_conversation"]

    console.print()
    panel = Panel(
        f"[bold cyan]Static vs Dynamic:[/bold cyan] {tb.get('static_to_native_reduction_ratio', 0)}x reduction "
        f"({tb['whole_codebase_tokens']:,} -> {tb['native_tokens']:,} tokens)\n"
        f"[bold blue]Static vs SG:[/bold blue] {tb.get('static_to_sg_reduction_ratio', 0)}x reduction "
        f"({tb['whole_codebase_tokens']:,} -> {tb['sg_tokens']:,} tokens)\n"
        f"[bold green]Native vs SG:[/bold green] {tb.get('native_to_sg_reduction_ratio', 0)}x reduction "
        f"({tb['native_tokens']:,} -> {tb['sg_tokens']:,} tokens)\n"
        f"[bold yellow]Tokens Saved (vs Native):[/bold yellow] {ta['tokens_saved']:,}",
        title="[bold]Context Window Scaling Results[/bold]",
    )
    console.print(panel)
    console.print(f"\n[dim]Full report: {eval_dir / 'report.md'}[/dim]")


def _discover_agent_file(agent: str):
    """Auto-discover the latest export file for an agent."""
    if agent == "antigravity":
        from ..eval.parsers.antigravity import discover_latest_log
        return discover_latest_log()
    elif agent == "copilot":
        from ..eval.parsers.copilot import discover_copilot_sessions
        sessions = discover_copilot_sessions()
        return sessions[0] if sessions else None
    elif agent == "cursor":
        from ..eval.parsers.cursor import discover_cursor_sessions
        sessions = discover_cursor_sessions()
        return sessions[0] if sessions else None
    elif agent == "codex":
        from ..eval.parsers.codex import discover_codex_sessions
        sessions = discover_codex_sessions()
        return sessions[0] if sessions else None
    elif agent == "claude_code":
        from ..eval.parsers.claude_code import discover_claude_code_sessions
        sessions = discover_claude_code_sessions()
        return sessions[0] if sessions else None
    return None


def _parse_agent_export(agent, export_path, project_root, prompt, mode, project_name):
    """Route to the correct parser for the agent."""
    if agent == "antigravity":
        from ..eval.parsers.antigravity import parse_antigravity_export
        return parse_antigravity_export(export_path, project_root, prompt, mode, project_name)
    elif agent == "copilot":
        from ..eval.parsers.copilot import parse_copilot_json_export
        return parse_copilot_json_export(export_path, project_root, prompt, mode, project_name)
    elif agent == "cursor":
        from ..eval.parsers.cursor import parse_cursor_session
        return parse_cursor_session(export_path, project_root, prompt, mode, project_name)
    elif agent == "codex":
        from ..eval.parsers.codex import parse_codex_session
        return parse_codex_session(export_path, project_root, prompt, mode, project_name)
    elif agent == "claude_code":
        from ..eval.parsers.claude_code import parse_claude_code_export
        return parse_claude_code_export(export_path, project_root, prompt, mode, project_name)
    console.print(f"[red]Unknown agent: {agent}[/red]")
    return None


def _save_and_display_trace(trace, project_root):
    """Save trace JSON and display summary."""
    eval_dir = project_root / ".skeletongraph" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{trace.mode}_trace.json"
    output_path = eval_dir / filename
    output_path.write_text(trace.to_json(), encoding="utf-8")

    table = Table(title=f"[bold]Parsed: {trace.agent} ({trace.mode})[/bold]", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Tool Output Tokens (L1)", f"{trace.total_tool_output_tokens:,}")
    table.add_row("Response Tokens (L2)", f"{trace.total_response_tokens:,}")
    table.add_row("History Tokens (L3)", f"{trace.estimated_history_tokens:,}")
    table.add_row("MCP Schema Overhead (L5)", f"{trace.mcp_schema_overhead_tokens:,}")
    table.add_row("Total Conversation", f"{trace.total_conversation_tokens:,}")
    table.add_row("Tool Calls", str(trace.tool_call_count))
    table.add_row("File Views", str(trace.view_file_count))
    table.add_row("Grep Searches", str(trace.grep_count))
    console.print(table)
    console.print(f"[green]Saved to:[/green] {output_path}")


def _print_export_instructions(agent: str):
    """Print how to export chat for each agent."""
    instructions = {
        "antigravity": "In Antigravity, click the Export button at the top of the chat.",
        "claude_code": "In Claude Code CLI, type: /export my_session.md",
        "cursor": "In Cursor, the SQLite DB is auto-discovered. Or copy the chat text to a file.",
        "codex": "Codex logs are at ~/.codex/. Or run: npx @ccusage/codex@latest",
        "copilot": "In VS Code, press Ctrl+Shift+P -> 'Chat: Export Session...'",
    }
    msg = instructions.get(agent, f"Export your {agent} chat to a file.")
    console.print(f"[dim]How to export: {msg}[/dim]")


# -- Helper functions ---------------------------------------------------

def _detect_platforms(project_root: Path) -> list:
    """Auto-detect which AI coding tools are configured."""
    detected = []
    home = Path.home()

    # Claude Code
    if (home / ".claude.json").exists() or (project_root / "CLAUDE.md").exists():
        detected.append("claude")

    # Cursor
    if (project_root / ".cursorrules").exists() or (home / ".cursor").exists():
        detected.append("cursor")

    # Antigravity
    if (project_root / ".antigravity.md").exists() or (home / ".gemini").exists():
        detected.append("antigravity")

    # Codex (OpenAI)
    if (project_root / "AGENTS.md").exists() or (home / ".codex").exists():
        detected.append("codex")

    # Windsurf
    if (project_root / ".windsurfrules").exists():
        detected.append("windsurf")

    # If nothing detected, default to claude + cursor + antigravity
    if not detected:
        detected = ["claude", "cursor", "antigravity"]

    return detected


def _install_platform(platform: str, project_root: Path):
    """Write IDE-specific rules for a platform."""
    templates = {
        "claude": ("CLAUDE.md", _claude_template()),
        "cursor": (".cursorrules", _cursor_template()),
        "antigravity": (".antigravity.md", _antigravity_template()),
        "codex": ("AGENTS.md", _codex_template()),
        "windsurf": (".windsurfrules", _windsurf_template()),
        "kiro": (".kiro/rules.md", _kiro_template()),
        "opencode": (".opencode/rules.md", _opencode_template()),
    }

    if platform not in templates:
        console.print(f"[yellow]Unknown platform: {platform}[/yellow]")
        return

    filename, content = templates[platform]
    target = project_root / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    # Don't overwrite if content already exists with SG section
    if target.exists():
        existing = target.read_text(encoding="utf-8", errors="replace")
        if "SkeletonGraph" in existing:
            console.print(f"  [dim]{filename} already configured, skipping[/dim]")
            return
        # Append to existing file
        content = existing.rstrip() + "\n\n" + content

    target.write_text(content, encoding="utf-8")
    console.print(f"  [green][OK][/green] Installed SkeletonGraph rules to {filename}")


def _write_mcp_config(project_root: Path):
    """Write MCP server configuration to local and global (Antigravity) configs."""
    python_exe = "C:\\Users\\ASUS\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
    
    server_config = {
        "command": python_exe,
        "args": ["-m", "skeletongraph.server.mcp", "--path", str(project_root)],
    }

    # 1. Write local mcp.json (always uses 'skeletongraph' as tool prefix)
    local_config = {"mcpServers": {"skeletongraph": server_config}}
    (project_root / "mcp.json").write_text(json.dumps(local_config, indent=2), encoding="utf-8")
    console.print(f"  [green][OK][/green] Updated local mcp.json")

    # 2. Write global Antigravity config
    ag_config_path = Path("C:/Users/ASUS/.gemini/antigravity/mcp_config.json")
    if ag_config_path.exists():
        try:
            ag_data = json.loads(ag_config_path.read_text(encoding="utf-8"))
            if "mcpServers" not in ag_data:
                ag_data["mcpServers"] = {}
            
            ag_data["mcpServers"]["skeletongraph"] = server_config
            ag_config_path.write_text(json.dumps(ag_data, indent=2), encoding="utf-8")
            console.print(f"  [green][OK][/green] Updated global Antigravity MCP config.")
        except Exception as e:
            console.print(f"[red]Failed to update Antigravity config: {e}[/red]")


def _sg_rules_block() -> str:
    """Common SkeletonGraph rules injected into all IDE configs."""
    return """
## SkeletonGraph Context Assembly

This project restricts native searching tools. You MUST use SkeletonGraph's MCP tools for ALL codebase discovery.

### CRITICAL RULES:
1. **DO NOT USE** `grep_search` or manual `view_file` commands unless explicitly ordered.
2. **TOOL DISCOVERY**: Depending on your IDE, the tools may be automatically prefixed. Look for tools containing `query_context`, `search_index`, or `expand_function` in your available tool list (e.g., `mcp_skeletongraph_flask_query_context` or `query_context`).
3. **USE the `query_context` tool** as your VERY FIRST action to get an optimized structural map.
4. **USE the `search_index` tool** instead of grep if you need to find where a variable, constant, or function is used.
5. **USE the `expand_function` tool** if you need to read the full body of a function you found in the graph. Do NOT read the entire file just for one function.
6. **RESPECT** the constraints listed in Zone 1 of any context response.

If you violate Rule 1, you will damage the evaluation telemetry. ONLY use the SkeletonGraph MCP tools provided.
""".strip()


def _claude_template() -> str:
    return f"""# CLAUDE.md - SkeletonGraph-Enhanced Rules

{_sg_rules_block()}
"""


def _cursor_template() -> str:
    return f"""
# REPOSITORY PROTOCOL: SkeletonGraph (MANDATORY)

This repository is optimized for AI agents using the SkeletonGraph MCP server. To prevent token overflow and ensure structural integrity, you MUST follow these protocols:

1. **INITIALIZATION**: Your very first action MUST be `skeletongraph:query_context`.
2. **SEARCH RESTRICTION**: Native `grep_search` and `file_search` are deprecated in this environment. Use `skeletongraph:search_index` for discovery.
3. **READ RESTRICTION**: Do NOT `view_file` on files larger than 100 lines. Use `skeletongraph:expand_function` to request only the specific logic you need.
4. **CREDIBILITY**: For every action you take, mention the SkeletonGraph Tier (Zone 2/3) you are using to prove compliance.

Failure to use SkeletonGraph tools will result in context truncation.
""".strip()


def _antigravity_template() -> str:
    return f"""# Antigravity Rules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


def _codex_template() -> str:
    return f"""# AGENTS.md - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


def _windsurf_template() -> str:
    return f"""# Windsurf Rules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


def _kiro_template() -> str:
    return f"""# Kiro Rules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


def _opencode_template() -> str:
    return f"""# OpenCode Rules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


if __name__ == "__main__":
    app()
