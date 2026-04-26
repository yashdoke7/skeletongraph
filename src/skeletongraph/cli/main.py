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
        console.print(f"[red]Could not find log file at {log_path}[/red]")
        return
        
    try:
        from ..metrics.log_parser import parse_antigravity_log
        from ..metrics.metrics_logger import MetricsLogger
        
        if agent == "antigravity":
            stats = parse_antigravity_log(target_log)
        else:
            console.print(f"[red]Parser for agent '{agent}' not implemented.[/red]")
            return
            
        metrics = MetricsLogger(project_root)
        metrics.log_baseline_estimate(
            prompt=prompt,
            total_tokens=stats["total_native_tokens"],
            files_read=stats["files_involved"],
            files_grepped=stats["grep_searches"],
            duration_ms=stats["duration_ms"]
        )
        
        table = Table(title="[bold]True Empirical Baseline Generated[/bold]", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Agent parsed", agent)
        table.add_row("Files viewed manually", str(stats["files_viewed"]))
        table.add_row("Grep searches run", str(stats["grep_searches"]))
        table.add_row("Total native tokens used", f"{stats['total_native_tokens']:,}")
        console.print(table)
        console.print("[dim][*] Real empirical baseline logged to JSONL.[/dim]")
        
    except Exception as e:
        console.print(f"[red]Failed to parse log: {e}[/red]")


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def watch(path: str):
    """Start a background daemon to auto-reindex files on save."""
    project_root = Path(path).resolve()
    from ..daemon import start_daemon
    start_daemon(project_root)


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
    """Write MCP server configuration."""
    mcp_config = {
        "mcpServers": {
            "skeletongraph": {
                "command": "skeletongraph",
                "args": ["serve", "--path", str(project_root)],
            }
        }
    }

    mcp_file = project_root / "mcp.json"
    if not mcp_file.exists():
        mcp_file.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
        console.print(f"  [green][OK][/green] Created mcp.json")


def _sg_rules_block() -> str:
    """Common SkeletonGraph rules injected into all IDE configs."""
    return """
## SkeletonGraph Context Assembly

This project uses SkeletonGraph for intelligent, token-minimal context assembly.

### Rules:
1. **ALWAYS** use the `query_context` MCP tool before reading files manually.
   It returns attention-optimized context with constraints, target code, and structure.
2. **NEVER** read more than 3 files manually if SkeletonGraph is available.
3. **RESPECT** the constraints in Zone 1 of every context response.
4. If context confidence is LOW, use `search_index` to find the right entry point.
5. Use `expand_function` for page-fault expansion when you need a specific function body.
6. Use `review_delta` when reviewing code changes - it computes blast radius automatically.

### What SkeletonGraph provides:
- Zone 1: Project constraints (primacy position - always read these first)
- Zone 2: Target code bodies (near prompt - strongest attention)
- Zone 3: Structural context (compressed neighbors and dependencies)
- Zone 4: Your current task (recency position)
""".strip()


def _claude_template() -> str:
    return f"""# CLAUDE.md - SkeletonGraph-Enhanced Rules

{_sg_rules_block()}
"""


def _cursor_template() -> str:
    return f"""# Cursor Rules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


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
