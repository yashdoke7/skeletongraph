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
from typing import List

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


@app.command(name="init")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--agent", "-a", default=None, help="IDE agent preset name")
@click.option("--non-interactive", is_flag=True, help="Skip prompts and use defaults")
def init_cmd(path: str, agent: str | None, non_interactive: bool):
    """Initialize project metadata and IDE integration files."""
    project_root = Path(path).resolve()
    from .init import run_init
    run_init(project_root, non_interactive=non_interactive, agent=agent)


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
def build(path: str):
    """Build the full index for a project."""
    project_root = Path(path).resolve()

    if not project_root.is_dir():
        console.print(f"[red]Error:[/red] {project_root} is not a directory")
        sys.exit(1)

    console.print(f"[bold]> Building index for[/bold] {project_root.name}")

    # Auto-trigger sg init on first build if project.md doesn't exist
    sg_dir = project_root / ".skeletongraph"
    project_md = sg_dir / "project.md"
    if not project_md.exists():
        console.print("[cyan]First build detected — running sg init...[/cyan]")
        from .init import run_init
        run_init(project_root)
        console.print()

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

    # Post-build: update architecture.md with store data
    sg_dir = project_root / ".skeletongraph"
    if sg_dir.exists():
        from .init import _generate_architecture
        arch_path = sg_dir / "architecture.md"
        arch_md = _generate_architecture(project_root, store)
        arch_path.write_text(arch_md, encoding="utf-8")
        console.print(f"  [dim]Updated {arch_path.relative_to(project_root)}[/dim]")


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
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--json-output", is_flag=True, help="Print machine-readable JSON")
def doctor(path: str, json_output: bool):
    """Check index, routing, provider, and CLI/IDE configuration health."""
    project_root = Path(path).resolve()

    from ..config import load_config
    from ..storage.local import load_index

    config = load_config(project_root)
    store = load_index(project_root)
    checks = []

    def add(name: str, ok: bool, detail: str):
        checks.append({"name": name, "ok": ok, "detail": detail})

    add(
        "project_root",
        project_root.is_dir(),
        str(project_root),
    )
    add(
        "index",
        store is not None,
        store.status_summary() if store else "missing; run `sg build`",
    )
    add(
        "mcp_profile",
        config.mcp_tool_profile in {"compact", "minimal", "full"},
        config.mcp_tool_profile,
    )
    add(
        "dynamic_routing",
        bool(config.enable_dynamic_model_routing),
        "enabled" if config.enable_dynamic_model_routing else "disabled",
    )
    add(
        "cli_provider",
        bool(config.cli_provider),
        config.cli_provider,
    )
    key_envs = config.get_cli_key_envs()
    add(
        "cli_api_key",
        config.cli_api_key_configured(),
        (
            "not required for selected provider"
            if not key_envs
            else
            f"set via {', '.join(key_envs)}"
            if config.cli_api_key_configured()
            else f"missing; set one of: {', '.join(key_envs) or 'provider env var'}"
        ),
    )
    add(
        "cli_api_base",
        bool(config.get_cli_api_base()) or bool(key_envs),
        config.get_cli_api_base() or "provider default",
    )

    payload = {
        "ok": all(c["ok"] for c in checks if c["name"] != "cli_api_key"),
        "execution_ready": all(c["ok"] for c in checks),
        "checks": checks,
        "ide": {
            "agent": config.agent,
            "models": {
                "slm": config.slm_model,
                "mlm": config.mlm_model,
                "llm": config.llm_model,
            },
        },
        "cli": {
            "provider": config.cli_provider,
            "models": {
                "slm": config.cli_slm_model,
                "mlm": config.cli_mlm_model,
                "llm": config.cli_llm_model,
            },
            "api_key_env": key_envs,
            "api_base": config.get_cli_api_base(),
        },
    }

    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    table = Table(title="SkeletonGraph Doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detail")
    for check in checks:
        table.add_row(
            check["name"],
            "OK" if check["ok"] else "WARN",
            check["detail"],
        )
    console.print(table)

    model_table = Table(title="Configured Models")
    model_table.add_column("Surface", style="cyan")
    model_table.add_column("SLM")
    model_table.add_column("MLM")
    model_table.add_column("LLM")
    model_table.add_row("IDE", config.slm_model, config.mlm_model, config.llm_model)
    model_table.add_row("CLI", config.cli_slm_model, config.cli_mlm_model, config.cli_llm_model)
    console.print(model_table)


@app.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--budget", "-b", default=128000, help="Model context limit")
@click.option("--out", "-o", help="Save context to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def query(prompt: str, path: str, budget: int, out: str | None, verbose: bool):
    """Query the index with a natural language prompt."""
    project_root = Path(path).resolve()

    from ..engine import SGEngine

    t0 = time.time()
    engine = SGEngine(project_root=project_root)
    result = engine.query(prompt, delivery="cli")
    duration_ms = int((time.time() - t0) * 1000)

    if not result.success:
        console.print(f"[yellow]{result.error}[/yellow]")
        return

    # Output
    if verbose:
        console.print(f"\n[bold]Mode:[/bold] {result.query_mode.value}")
        console.print(f"[bold]Model tier:[/bold] {result.model_tier.value}")
        console.print(f"[bold]Base tier:[/bold] {result.base_model_tier.value}")
        console.print(f"[bold]Recommended model:[/bold] {result.recommended_model or 'default'}")
        console.print(f"[bold]Delivery:[/bold] {result.delivery}")
        console.print(f"[bold]Complexity:[/bold] {result.complexity_score:.2f}")
        console.print(f"[bold]Routing:[/bold] {result.routing_reason or 'static'}")
        console.print(f"[bold]Confidence:[/bold] {result.confidence}")
        console.print(f"[bold]Reason:[/bold] {result.confidence_reason}")
        console.print(f"[bold]Candidates:[/bold] {len(result.candidates)}")
        console.print(f"[bold]Pipeline:[/bold] {result.pipeline_path}")

        # Token budget table
        table = Table(title="Token Budget")
        table.add_column("Metric", style="cyan")
        table.add_column("Tokens", style="green")
        table.add_row("Context", str(result.context_tokens))
        table.add_row("SLM input", str(result.slm_input_tokens))
        table.add_row("SLM output", str(result.slm_output_tokens))
        table.add_row("Total", str(result.context_tokens), style="bold")
        console.print(table)

        if result.slm_used:
            console.print(
                f"\n[dim]SLM fallback used: {result.slm_entities_found} entities, "
                f"${result.slm_cost_usd:.6f}[/dim]"
            )

    if out:
        out_path = Path(out).resolve()
        out_path.write_text(result.context_text, encoding="utf-8")
        console.print(f"\n[green][OK][/green] Saved assembled context to {out_path}")
    else:
        console.print(f"\n[dim]--- Assembled Context ({result.context_tokens} tokens) ---[/dim]\n")
        console.print(result.context_text)

    console.print(f"\n[dim][*] Completed in {duration_ms}ms[/dim]")


@app.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--json-output", is_flag=True, help="Print machine-readable JSON")
def route(prompt: str, path: str, json_output: bool):
    """Show SG's deterministic model-routing decision for a task."""
    project_root = Path(path).resolve()

    from ..engine import SGEngine
    from ..retrieval.resolver import Tier

    t0 = time.time()
    engine = SGEngine(project_root=project_root)
    result = engine.query(prompt, delivery="cli")
    duration_ms = int((time.time() - t0) * 1000)

    if not result.success:
        console.print(f"[yellow]{result.error}[/yellow]")
        return

    targets = [
        c.skeleton.fqn for c in result.candidates
        if c.tier == Tier.TIER1
    ]
    payload = {
        "mode": result.query_mode.value,
        "tier": result.model_tier.value,
        "base_tier": result.base_model_tier.value,
        "recommended_model": result.recommended_model,
        "delivery": result.delivery,
        "complexity_score": result.complexity_score,
        "routing_reason": result.routing_reason,
        "confidence": result.confidence,
        "context_tokens": result.context_tokens,
        "candidate_count": len(result.candidates),
        "targets": targets[:5],
        "duration_ms": duration_ms,
    }
    config = engine.get_config()
    payload["cli_provider"] = config.cli_provider
    payload["api_key_env"] = config.get_cli_key_envs()
    payload["api_key_configured"] = config.cli_api_key_configured()
    payload["api_base"] = config.get_cli_api_base()

    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    table = Table(title="SG Route")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Mode", payload["mode"])
    table.add_row("Tier", payload["tier"])
    table.add_row("Base tier", payload["base_tier"])
    table.add_row("Recommended model", payload["recommended_model"] or "default")
    table.add_row("CLI provider", payload["cli_provider"])
    table.add_row("API key env", ", ".join(payload["api_key_env"]) or "none")
    table.add_row("API key configured", "yes" if payload["api_key_configured"] else "no")
    table.add_row("API base", payload["api_base"] or "provider default")
    table.add_row("Complexity", f"{payload['complexity_score']:.2f}")
    table.add_row("Reason", payload["routing_reason"])
    table.add_row("Confidence", payload["confidence"])
    table.add_row("Packet tokens", str(payload["context_tokens"]))
    table.add_row("Candidates", str(payload["candidate_count"]))
    table.add_row("Targets", ", ".join(targets[:3]) or "none")
    console.print(table)


@app.command()
@click.argument("prompt")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--dry-run", is_flag=True, help="Show execution plan without calling a model")
@click.option("--execute", is_flag=True, help="Call the configured provider and write model output")
@click.option("--auto-model", is_flag=True, help="Use SG dynamic routing (default unless --tier is set)")
@click.option("--tier", type=click.Choice(["slm", "mlm", "llm"]), default=None, help="Override routed tier")
@click.option("--out", "-o", default=None, help="Write prepared packet to a file")
@click.option("--response-out", default=None, help="Write provider output to this file")
@click.option("--max-output-tokens", type=int, default=2000, help="Max provider output tokens")
@click.option("--temperature", type=float, default=0.1, help="Provider sampling temperature")
@click.option("--plan-first/--no-plan-first", default=True, help="Use SLM tool planning before execution")
@click.option("--plan-max-tokens", type=int, default=800, help="Max tokens per planned expansion")
@click.option("--error-followup/--no-error-followup", default=False, help="Use last error-only follow-up instead of full context")
@click.option("--update-comments/--no-update-comments", default=True, help="Ask the model to update docstrings/comments if needed")
@click.option("--json-output", is_flag=True, help="Print machine-readable JSON")
def run(
    prompt: str,
    path: str,
    dry_run: bool,
    execute: bool,
    auto_model: bool,
    tier: str | None,
    out: str | None,
    response_out: str | None,
    max_output_tokens: int,
    temperature: float,
    plan_first: bool,
    plan_max_tokens: int,
    error_followup: bool,
    update_comments: bool,
    json_output: bool,
):
    """Plan or execute an SG CLI model-routed task."""
    if dry_run and execute:
        console.print("[red]Use either --dry-run or --execute, not both.[/red]")
        return

    project_root = Path(path).resolve()

    from ..engine import SGEngine
    from ..llm.provider import LLMConfig, complete
    from ..retrieval.resolver import Tier
    from .run_exec import (
        ErrorFollowup,
        RunPlan,
        build_execution_prompt,
        build_system_prompt,
        clear_error_followup,
        default_run_paths,
        load_error_followup,
        save_error_followup,
        write_run_log,
    )

    engine = SGEngine(project_root=project_root)
    result = engine.query(prompt, delivery="cli", force_slm=plan_first)
    config = engine.get_config()

    if not result.success:
        console.print(f"[yellow]{result.error}[/yellow]")
        return

    routing_mode = "manual" if tier else "auto"
    selected_tier = tier or result.model_tier.value
    selected_model = config.get_cli_model_for_tier(selected_tier)
    targets = [
        c.skeleton.fqn for c in result.candidates
        if c.tier == Tier.TIER1
    ]

    plan_payload = None
    extra_context = ""
    expansion_errors: List[str] = []
    if plan_first:
        try:
            from ..retrieval.slm_extractor import slm_plan_tools
            store = engine.get_store()
            session = engine.get_session()
            session_fqns = session.get_last_target_fqns() if session else set()
            sg_dir = project_root / ".skeletongraph"

            slm_plan = slm_plan_tools(
                prompt=prompt,
                store=store,
                sg_dir=sg_dir,
                config=config,
                session_fqns=session_fqns,
            )

            plan_payload = {
                "success": slm_plan.success,
                "reasoning": slm_plan.reasoning,
                "tool_calls": [
                    {
                        "type": c.call_type,
                        "target": c.target,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "include_neighbors": c.include_neighbors,
                    }
                    for c in slm_plan.tool_calls
                ],
                "input_tokens": slm_plan.input_tokens,
                "output_tokens": slm_plan.output_tokens,
                "cost": slm_plan.cost_usd,
                "latency_ms": slm_plan.latency_ms,
                "error": slm_plan.error,
            }

            if slm_plan.success and slm_plan.tool_calls:
                expansions = []
                for call in slm_plan.tool_calls:
                    try:
                        expanded = engine.expand(
                            target=call.target,
                            expand_type=call.call_type,
                            start_line=call.start_line or None,
                            end_line=call.end_line or None,
                            include_neighbors=call.include_neighbors,
                            max_tokens=plan_max_tokens,
                        )
                        expansions.append(f"### {call.call_type} {call.target}\n{expanded}")
                        if expanded.startswith(("File not found:", "Function not found:", "Unknown expand type:")):
                            expansion_errors.append(f"{call.call_type} {call.target}: {expanded}")
                    except Exception as e:
                        expansions.append(f"### {call.call_type} {call.target}\n[ERROR] {e}")
                        expansion_errors.append(f"{call.call_type} {call.target}: {e}")
                if expansions:
                    extra_context = "## Planned Expansions\n" + "\n\n".join(expansions)
        except Exception:
            plan_payload = {
                "success": False,
                "reasoning": "",
                "tool_calls": [],
                "error": "planner_failed",
            }

    if extra_context:
        result.context_text = f"{result.context_text}\n\n---\n\n{extra_context}"
        result.context_tokens = len(result.context_text) // 4

    if out:
        out_path = Path(out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.context_text, encoding="utf-8")
    else:
        out_path = None

    default_response_path, run_log_path = default_run_paths(project_root)
    response_path = Path(response_out).resolve() if response_out else default_response_path

    error_followup_saved = False
    if expansion_errors:
        save_error_followup(
            project_root,
            ErrorFollowup(
                prompt=prompt,
                timestamp=time.time(),
                source="planner_expand",
                errors=expansion_errors,
                context_path=str(out_path) if out_path else None,
                run_log_path=str(run_log_path),
            ),
        )
        error_followup_saved = True

    plan = RunPlan(
        prompt=prompt,
        mode=result.query_mode.value,
        routed_tier=result.model_tier.value,
        selected_tier=selected_tier,
        selected_model=selected_model,
        cli_provider=config.cli_provider,
        api_key_env=config.get_cli_key_envs(),
        api_key_configured=config.cli_api_key_configured(),
        api_base=config.get_cli_api_base(),
        context_tokens=result.context_tokens,
        confidence=result.confidence,
        complexity_score=result.complexity_score,
        routing_reason=result.routing_reason,
        targets=targets[:5],
        packet_path=str(out_path) if out_path else None,
    )

    payload = {
        "implemented": True,
        "dry_run": not execute,
        "execute": execute,
        "routing_mode": routing_mode,
        "auto_model": auto_model or tier is None,
        "plan_first": plan_first,
        "plan_max_tokens": plan_max_tokens,
        "error_followup_used": bool(error_followup_data),
        "error_followup_saved": error_followup_saved,
        **plan.to_dict(),
        "response_path": str(response_path),
        "run_log_path": str(run_log_path),
        "next_status": "dry-run only; pass --execute to call the provider",
    }

    if plan_payload is not None:
        payload["planner"] = plan_payload

    provider_response = None
    if execute:
        if not config.cli_api_key_configured():
            missing = ", ".join(config.get_cli_key_envs()) or "provider API key"
            console.print(f"[red]Missing CLI provider API key.[/red] Set one of: {missing}")
            return

        exec_prompt = build_execution_prompt(prompt, result.context_text)
        started = time.time()
        try:
            resp = complete(
                exec_prompt,
                system=build_system_prompt(update_comments=update_comments),
                config=LLMConfig(
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    api_base=config.get_cli_api_base(),
                ),
            )
        except Exception as e:
            save_error_followup(
                project_root,
                ErrorFollowup(
                    prompt=prompt,
                    timestamp=time.time(),
                    source="provider_execute",
                    errors=[str(e)],
                    context_path=str(out_path) if out_path else None,
                    run_log_path=str(run_log_path),
                ),
            )
            payload["next_status"] = f"provider error: {e}"
            write_run_log(run_log_path, {"timestamp": time.time(), **payload})
            console.print(f"[red]Provider error:[/red] {e}")
            return

        elapsed_ms = int((time.time() - started) * 1000)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(resp.text, encoding="utf-8")
        provider_response = {
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost": resp.cost,
            "duration_ms": elapsed_ms,
            "response_path": str(response_path),
        }
        payload["provider_response"] = provider_response
        payload["next_status"] = "provider output written; inspect/apply patch manually"

        if not expansion_errors:
            clear_error_followup(project_root)

        if config.auto_rebuild_on_completion:
            try:
                from ..build import update_index
                update_index(project_root)
                payload["index_updated"] = True
            except Exception:
                payload["index_updated"] = False

    write_run_log(run_log_path, {
        "timestamp": time.time(),
        **payload,
    })

    if json_output:
        console.print(json.dumps(payload, indent=2))
        return

    table = Table(title="SG Run" if execute else "SG Run Plan")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Mode", payload["mode"])
    table.add_row("Routed tier", payload["routed_tier"])
    table.add_row("Routing mode", payload["routing_mode"])
    table.add_row("Selected tier", payload["selected_tier"])
    table.add_row("Selected model", payload["selected_model"])
    table.add_row("CLI provider", payload["cli_provider"])
    table.add_row("API key configured", "yes" if payload["api_key_configured"] else "no")
    table.add_row("API base", config.get_cli_api_base() or "provider default")
    table.add_row("Packet tokens", str(payload["context_tokens"]))
    table.add_row("Confidence", payload["confidence"])
    table.add_row("Complexity", f"{payload['complexity_score']:.2f}")
    table.add_row("Targets", ", ".join(targets[:3]) or "none")
    if payload.get("error_followup_used"):
        table.add_row("Error followup", "used")
    if payload.get("error_followup_saved"):
        table.add_row("Error followup", "saved")
    if update_comments:
        table.add_row("Update comments", "yes")
    if plan_payload is not None:
        tool_count = len(plan_payload.get("tool_calls", []))
        status = "ok" if plan_payload.get("success") else "failed"
        table.add_row("Planner", f"{status} ({tool_count} calls)")
    if out_path:
        table.add_row("Packet path", str(out_path))
    table.add_row("Response path", str(response_path))
    table.add_row("Run log", str(run_log_path))
    if provider_response:
        table.add_row("Provider input tokens", str(provider_response["input_tokens"]))
        table.add_row("Provider output tokens", str(provider_response["output_tokens"]))
        table.add_row("Provider cost", f"${provider_response['cost']:.6f}")
    table.add_row("Status", payload["next_status"])
    console.print(table)


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--model", "-m", default="gemini/gemini-2.5-flash", help="LLM model")
@click.option("--force", is_flag=True, help="Re-summarize all functions")
@click.option("--api-key", envvar=["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"], default=None, help="API key for the LLM provider")
def summarize(path: str, model: str, force: bool, api_key: str):
    """Generate LLM summaries for indexed functions."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index
    from ..llm.summarizer import summarize_index
    from ..llm.provider import LLMConfig

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    cfg = LLMConfig(model=model, api_key=api_key or None)
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
    _write_mcp_config(project_root, platforms=platforms)
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


@app.command(name="config")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--agent", "-a", type=click.Choice(
    ["cursor", "copilot", "codex", "claude_code", "antigravity"],
    case_sensitive=False,
), default=None, help="Switch to agent preset")
@click.option("--cli-provider", type=click.Choice(
    ["anthropic", "openai", "google", "local"],
    case_sensitive=False,
), default=None, help="Set CLI execution provider preset")
@click.option("--slm", default=None, help="Override SLM model")
@click.option("--mlm", default=None, help="Override MLM model")
@click.option("--llm", default=None, help="Override LLM model")
@click.option("--cli-slm", default=None, help="Override CLI SLM provider model")
@click.option("--cli-mlm", default=None, help="Override CLI MLM provider model")
@click.option("--cli-llm", default=None, help="Override CLI LLM provider model")
@click.option("--cli-api-base", default=None, help="Override CLI provider API base URL")
@click.option("--dynamic-routing/--static-routing", default=None, help="Enable or disable dynamic model routing")
def config_cmd(
    path: str,
    show: bool,
    agent: str,
    cli_provider: str,
    slm: str,
    mlm: str,
    llm: str,
    cli_slm: str,
    cli_mlm: str,
    cli_llm: str,
    cli_api_base: str | None,
    dynamic_routing: bool | None,
):
    """View or update SkeletonGraph model configuration."""
    from ..config import AGENT_PRESETS, CLI_PROVIDER_PRESETS, load_config, save_config

    project_root = Path(path).resolve()
    config = load_config(project_root)

    # If no flags, run interactive mode
    if not any([
        show, agent, cli_provider, slm, mlm, llm, cli_slm, cli_mlm, cli_llm,
        cli_api_base, dynamic_routing is not None,
    ]):
        console.print("[bold cyan]SkeletonGraph Model Configuration[/bold cyan]\n")

        # Show current
        console.print(f"  IDE agent: [bold]{config.agent}[/bold]")
        console.print(f"  IDE SLM/MLM/LLM: {config.slm_model} / {config.mlm_model} / {config.llm_model}")
        console.print(f"  CLI provider: [bold]{config.cli_provider}[/bold]")
        console.print(
            f"  CLI SLM/MLM/LLM: {config.cli_slm_model} / "
            f"{config.cli_mlm_model} / {config.cli_llm_model}"
        )
        key_envs = config.get_cli_key_envs()
        key_status = (
            "not required"
            if not key_envs
            else "set"
            if config.cli_api_key_configured()
            else "missing"
        )
        console.print(f"  CLI API key: {key_status} ({', '.join(key_envs) or 'none'})")
        if config.get_cli_api_base():
            console.print(f"  CLI API base: {config.get_cli_api_base()}")

        console.print("\n[bold]What would you like to do?[/bold]")
        console.print("  [cyan]1[/cyan]. Switch IDE agent preset")
        console.print("  [cyan]2[/cyan]. Override IDE model label")
        console.print("  [cyan]3[/cyan]. Configure CLI provider preset")
        console.print("  [cyan]4[/cyan]. Override CLI provider model")
        console.print("  [cyan]5[/cyan]. Show available models")
        console.print("  [cyan]6[/cyan]. Exit")

        from rich.prompt import Prompt
        choice = Prompt.ask("\nSelect", choices=["1", "2", "3", "4", "5", "6"], default="6")

        if choice == "1":
            agent_names = list(AGENT_PRESETS.keys())
            console.print()
            for i, name in enumerate(agent_names, 1):
                p = AGENT_PRESETS[name]
                marker = " [green]< current[/green]" if name == config.agent else ""
                console.print(
                    f"  [cyan]{i}[/cyan]. {name}{marker}  "
                    f"[dim](SLM: {p['slm']}, MLM: {p['mlm']}, LLM: {p['llm']})[/dim]"
                )
            sel = Prompt.ask("\nSelect agent (number)", default="1")
            if sel.isdigit() and 1 <= int(sel) <= len(agent_names):
                agent = agent_names[int(sel) - 1]

        elif choice == "2":
            preset = AGENT_PRESETS.get(config.agent, {})
            models = preset.get("models_available", [])
            if models:
                console.print("\n[bold]Available models:[/bold]")
                for i, m in enumerate(models, 1):
                    console.print(f"  [cyan]{i}[/cyan]. {m}")

                tier = Prompt.ask("\nWhich tier to change?", choices=["slm", "mlm", "llm"])
                sel = Prompt.ask("Select model (number or name)")
                if sel.isdigit() and 1 <= int(sel) <= len(models):
                    model_name = models[int(sel) - 1]
                else:
                    model_name = sel

                if tier == "slm":
                    slm = model_name
                elif tier == "mlm":
                    mlm = model_name
                else:
                    llm = model_name
            else:
                console.print("[yellow]No model list available. Type model name directly.[/yellow]")
                tier = Prompt.ask("Which tier?", choices=["slm", "mlm", "llm"])
                model_name = Prompt.ask("Model name")
                if tier == "slm":
                    slm = model_name
                elif tier == "mlm":
                    mlm = model_name
                else:
                    llm = model_name

        elif choice == "3":
            provider_names = list(CLI_PROVIDER_PRESETS.keys())
            console.print()
            for i, name in enumerate(provider_names, 1):
                p = CLI_PROVIDER_PRESETS[name]
                marker = " [green]< current[/green]" if name == config.cli_provider else ""
                console.print(
                    f"  [cyan]{i}[/cyan]. {name}{marker}  "
                    f"[dim](SLM: {p['slm']}, MLM: {p['mlm']}, LLM: {p['llm']})[/dim]"
                )
            sel = Prompt.ask("\nSelect provider (number)", default="1")
            if sel.isdigit() and 1 <= int(sel) <= len(provider_names):
                cli_provider = provider_names[int(sel) - 1]

        elif choice == "4":
            preset = CLI_PROVIDER_PRESETS.get(config.cli_provider, {})
            models = preset.get("models_available", [])
            if models:
                console.print("\n[bold]Available CLI provider models:[/bold]")
                for i, m in enumerate(models, 1):
                    console.print(f"  [cyan]{i}[/cyan]. {m}")
            tier = Prompt.ask("\nWhich CLI tier to change?", choices=["slm", "mlm", "llm"])
            sel = Prompt.ask("Select model (number or name)")
            if sel.isdigit() and models and 1 <= int(sel) <= len(models):
                model_name = models[int(sel) - 1]
            else:
                model_name = sel
            if tier == "slm":
                cli_slm = model_name
            elif tier == "mlm":
                cli_mlm = model_name
            else:
                cli_llm = model_name

        elif choice == "5":
            preset = AGENT_PRESETS.get(config.agent, {})
            models = preset.get("models_available", [])
            if models:
                console.print(f"\n[bold]Models available in {config.agent}:[/bold]")
                for m in models:
                    console.print(f"  - {m}")
            else:
                console.print("[yellow]No model list available for this agent.[/yellow]")
            cli_preset = CLI_PROVIDER_PRESETS.get(config.cli_provider, {})
            cli_models = cli_preset.get("models_available", [])
            if cli_models:
                console.print(f"\n[bold]CLI models for {config.cli_provider}:[/bold]")
                for m in cli_models:
                    console.print(f"  - {m}")
            return

        else:
            return

    # Show current config
    if show:
        table = Table(title="SkeletonGraph Config", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Agent", config.agent)
        table.add_row("IDE SLM", config.slm_model)
        table.add_row("IDE MLM", config.mlm_model)
        table.add_row("IDE LLM", config.llm_model)
        table.add_row("CLI Provider", config.cli_provider)
        table.add_row("CLI SLM", config.cli_slm_model)
        table.add_row("CLI MLM", config.cli_mlm_model)
        table.add_row("CLI LLM", config.cli_llm_model)
        key_envs = config.get_cli_key_envs()
        key_status = (
            "not required"
            if not key_envs
            else "set"
            if config.cli_api_key_configured()
            else "missing"
        )
        table.add_row("CLI API Key Env", ", ".join(key_envs) or "none")
        table.add_row("CLI API Key", key_status)
        table.add_row("CLI API Base", config.get_cli_api_base() or "provider default")
        table.add_row("Dynamic Routing", "enabled" if config.enable_dynamic_model_routing else "disabled")
        table.add_row("Context Limit", f"{config.model_context_limit:,} tokens")
        table.add_row("MCP Profile", config.mcp_tool_profile)
        table.add_row("Session TTL", f"{config.session_ttl_minutes} min")
        console.print(table)

        preset = AGENT_PRESETS.get(config.agent, {})
        hint = preset.get("select_model_hint", "")
        if hint:
            console.print(f"\n[dim]Hint: {hint}[/dim]")
        return

    # Apply agent preset
    if agent:
        preset = AGENT_PRESETS.get(agent)
        if preset:
            config.agent = agent
            config.slm_model = preset["slm"]
            config.mlm_model = preset["mlm"]
            config.llm_model = preset["llm"]
            console.print(f"[green][OK][/green] Switched to [bold]{agent}[/bold] preset")
            hint = preset.get("select_model_hint", "")
            if hint:
                console.print(f"  [yellow]NOTE:[/yellow] {hint}")

    if cli_provider:
        config.apply_cli_provider_preset(cli_provider)
        envs = ", ".join(config.get_cli_key_envs()) or "provider API key env"
        console.print(f"[green][OK][/green] CLI provider -> {cli_provider}")
        if config.get_cli_key_envs():
            console.print(f"  [yellow]NOTE:[/yellow] Set {envs}; SG does not store API keys.")
        else:
            console.print(
                f"  [yellow]NOTE:[/yellow] No API key required; SG will use "
                f"{config.get_cli_api_base() or 'the configured local endpoint'}."
            )

    # Apply individual overrides
    if slm:
        config.slm_model = slm
        console.print(f"[green][OK][/green] IDE SLM -> {slm}")
    if mlm:
        config.mlm_model = mlm
        console.print(f"[green][OK][/green] IDE MLM -> {mlm}")
    if llm:
        config.llm_model = llm
        console.print(f"[green][OK][/green] IDE LLM -> {llm}")
    if cli_slm:
        config.cli_slm_model = cli_slm
        console.print(f"[green][OK][/green] CLI SLM -> {cli_slm}")
    if cli_mlm:
        config.cli_mlm_model = cli_mlm
        console.print(f"[green][OK][/green] CLI MLM -> {cli_mlm}")
    if cli_llm:
        config.cli_llm_model = cli_llm
        console.print(f"[green][OK][/green] CLI LLM -> {cli_llm}")
    if cli_api_base:
        config.cli_api_base = cli_api_base
        console.print(f"[green][OK][/green] CLI API base -> {cli_api_base}")
    if dynamic_routing is not None:
        config.enable_dynamic_model_routing = dynamic_routing
        state = "enabled" if dynamic_routing else "disabled"
        console.print(f"[green][OK][/green] Dynamic routing {state}")

    save_config(config, project_root)
    console.print(f"\n[dim]Config saved to .skeletongraph/config.json[/dim]")


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
        result = resolve_context(prompt, store, enable_keyword_fallback=False)
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


@app.command(name="eval-golden")
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


# Register additional commands from submodules
from .prepare import prepare as _prepare_command
app.add_command(_prepare_command)


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

    # SG mode parses the shared SG session format, either from an archived file
    # or from the current project session.
    if mode == "skeletongraph":
        try:
            session_path = Path(file_path) if file_path else None
            trace = _parse_sg_session(agent, project_root, project, session_path=session_path)
            _save_and_display_trace(trace, project_root)
            return
        except FileNotFoundError as e:
            console.print(f"[red]No SG session found: {e}[/red]")
            return

    # Auto-discover if no native export was provided.
    if not file_path:

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
                sg_trace = _parse_sg_session("antigravity", project_root, "")
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

    # Safe row helper
    def add_safe_row(label, sg_key, native_key, data):
        sg_val = data.get(sg_key, 0)
        native_val = data.get(native_key, 0)
        ratio = data.get("reduction_ratio", native_val / sg_val if sg_val > 0 else 1.0)
        table.add_row(label, f"{sg_val:,}", f"{native_val:,}", f"{ratio:.1f}x")

    add_safe_row("Retrieval Tokens", "sg_tokens", "native_tokens", ta)
    add_safe_row("Conversation Tokens", "sg_tokens", "native_tokens", tb)
    
    table.add_row("Tool Calls", str(tc.get('sg_tool_calls', 0)), str(tc.get('native_tool_calls', 0)), "")
    table.add_row("Turns", str(tc.get('sg_turns', 0)), str(tc.get('native_turns', 0)), "")
    table.add_row("Repeated Views", "0", str(tc.get('native_repeated_views', 0)), "")

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
@click.option("--dataset", "-d", default="swe-bench-verified", type=click.Choice(["swe-bench-verified", "crg-compat", "custom"]), help="Dataset to evaluate against")
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
    elif dataset == "crg-compat":
        from ..eval.datasets.crg_compat import load_crg_compat
        tasks = load_crg_compat(repos=repo_list)
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
    elif dataset == "crg-compat":
        from ..eval.datasets.crg_compat import CRG_CONFIGS, CRG_PUBLISHED_RESULTS
        
        table = Table(title="[bold]CRG-Compatible Repos[/bold]", show_header=True)
        table.add_column("Repo", style="cyan")
        table.add_column("Commits", style="yellow", justify="right")
        table.add_column("Language", style="dim")
        table.add_column("CRG Claimed", style="magenta", justify="right")
        
        for name, config in CRG_CONFIGS.items():
            crg_result = CRG_PUBLISHED_RESULTS.get(name, {})
            reduction = crg_result.get("reduction", "?")
            table.add_row(
                name,
                str(len(config["test_commits"])),
                config["language"],
                f"{reduction}x",
            )
        
        console.print(table)
        console.print("\n[dim]Note: CRG uses len(text)//4 for tokens. We use tiktoken BPE.[/dim]")
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
        sg_trace = _parse_sg_session(agent, project_root, project)
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


def _parse_sg_session(agent: str, project_root: Path, project: str, session_path: Path | None = None):
    """Parse the common SkeletonGraph session and attribute it to the active agent."""
    from ..eval.parsers.antigravity import parse_antigravity_sg_session

    return parse_antigravity_sg_session(project_root, project, agent=agent, session_path=session_path)


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
    if getattr(trace, "api_input_tokens", None) is not None:
        table.add_row("API Input Tokens", f"{trace.api_input_tokens:,}")
    if trace.reasoning_tokens is not None:
        table.add_row("Reasoning/API Output (L4)", f"{trace.reasoning_tokens:,}")
    table.add_row("MCP Schema Overhead (L5)", f"{trace.mcp_schema_overhead_tokens:,}")
    table.add_row("Measured Conversation", f"{trace.total_conversation_tokens:,}")
    if trace.reasoning_tokens is not None:
        table.add_row("Measured + L4", f"{trace.total_conversation_tokens + trace.reasoning_tokens:,}")
    if getattr(trace, "api_input_tokens", None) is not None:
        table.add_row("Exact API Total", f"{trace.api_input_tokens + (trace.reasoning_tokens or 0):,}")
    if getattr(trace, "model_turns", None) is not None:
        table.add_row("Model Turns", str(trace.model_turns))
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

    # Copilot (VS Code)
    if (project_root / ".vscode").exists() or (project_root / ".github" / "copilot-instructions.md").exists():
        detected.append("copilot")

    # Antigravity
    if (project_root / ".antigravity.md").exists() or (home / ".gemini").exists():
        detected.append("antigravity")

    # Codex (OpenAI)
    if (project_root / "AGENTS.md").exists() or (home / ".codex").exists():
        detected.append("codex")

    # Windsurf
    if (project_root / ".windsurfrules").exists():
        detected.append("windsurf")

    # If nothing detected, default to claude + cursor + copilot + antigravity
    if not detected:
        detected = ["claude", "cursor", "copilot", "antigravity"]

    return detected


def _install_platform(platform: str, project_root: Path):
    """Write IDE-specific rules for a platform."""
    templates = {
        "claude": ("CLAUDE.md", _claude_template()),
        "cursor": (".cursor/rules/skeletongraph.mdc", _cursor_template()),
        "copilot": (".github/copilot-instructions.md", _copilot_template()),
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

def _write_mcp_config(project_root: Path, platforms: list[str] = None):
    """Write MCP server configuration to local and global (Antigravity) configs."""
    python_exe = sys.executable  # Portable: use the Python that's running SG
    platforms = platforms or []
    
    server_config = {
        "command": python_exe,
        "args": ["-m", "skeletongraph.server.mcp", "--path", str(project_root)],
    }

    # 1. Write local mcp.json (generic)
    local_config = {"mcpServers": {"skeletongraph": server_config}}
    (project_root / "mcp.json").write_text(json.dumps(local_config, indent=2), encoding="utf-8")
    console.print(f"  [green][OK][/green] Updated local mcp.json")

    # 2. Write .vscode/mcp.json for Copilot (uses "servers" key)
    if "copilot" in platforms or not platforms:
        vscode_dir = project_root / ".vscode"
        vscode_dir.mkdir(parents=True, exist_ok=True)
        vscode_config = {"servers": {"skeletongraph": server_config}}
        (vscode_dir / "mcp.json").write_text(json.dumps(vscode_config, indent=2), encoding="utf-8")
        console.print(f"  [green][OK][/green] Updated .vscode/mcp.json")

    # 3. Write .cursor/mcp.json for Cursor (uses "mcpServers" key)
    if "cursor" in platforms or not platforms:
        cursor_dir = project_root / ".cursor"
        cursor_dir.mkdir(parents=True, exist_ok=True)
        cursor_config = {"mcpServers": {"skeletongraph": server_config}}
        (cursor_dir / "mcp.json").write_text(json.dumps(cursor_config, indent=2), encoding="utf-8")
        console.print(f"  [green][OK][/green] Updated .cursor/mcp.json")

    # 4. Write global Antigravity config
    if "antigravity" in platforms or not platforms:
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
## ⚡️ SkeletonGraph: MANDATORY WORKFLOW

> **CRITICAL DIRECTIVE: Follow these 3 steps for EVERY coding task. Violations waste tokens and reduce accuracy.**

This project uses SkeletonGraph for high-precision context assembly. The MCP server has indexed the entire codebase into a dependency graph with function signatures, summaries, and structural metadata.

### STEP 1: RETRIEVE THE FUNCTION INDEX (MANDATORY FIRST MOVE)
**DO NOT use grep, ls, find, or read_file as your first action.**
Your very first tool call MUST be `get_retrieval_context`:

```
get_retrieval_context(prompt="<the user's task>")
```

This returns:
- **Project summary** — what this project does
- **File map** — all source files with descriptions
- **Function index** — every function with a one-line summary
- **Session context** — what was done in recent turns

READ the function index carefully. Identify which FQNs (fully qualified names) are relevant to the user's task. Match the user's natural language description to the function SUMMARIES, not just names.

### STEP 2: QUERY WITH EXTRACTED ENTITIES (MANDATORY)
Call `query_context` with the entities YOU identified:

```
query_context(
  prompt="<the user's task>",
  entities=["file.py::ClassName.method_name", "other_file.py::function_name"]
)
```

This triggers graph-based expansion: SkeletonGraph fetches the full bodies of your target functions PLUS their callers, callees, test coverage, and structural neighbors — assembled into an attention-optimized 4-zone prompt.

**DO NOT skip Step 1 and call query_context with prompt only.** Without entities, the server falls back to regex keyword matching which is significantly less accurate.

### STEP 3: REPORT COMPLETION (MANDATORY FINAL MOVE)
After you finish the coding task, you MUST call `report_completion`:

```
report_completion(
  summary="Fixed the authentication bug in login handler",
  files_modified=["src/auth/handler.py", "tests/test_auth.py"],
  session_end=false
)
```

This updates session memory so future queries know what was changed. **Failure to call report_completion breaks cross-session continuity.**

### ZONE-BASED CONTEXT NAVIGATION
Every `query_context` response has 4 zones ordered by attention priority:
- **Zone 1 (Constraints):** READ FIRST. Mandatory project rules and guardrails.
- **Zone 2 (Target Code):** Full source of the functions you need to modify.
- **Zone 3 (Structural Context):** Callers, callees, neighbors. The "why" behind the code.
- **Zone 4 (Task):** Your mission statement at the attention boundary.

### REFINEMENT (ONLY IF NEEDED)
- Use `expand_context` if a skeleton wasn't enough and you need the full function body.
- Use native tools (grep, read_file) ONLY after you have the SkeletonGraph context and need surgical details.
""".strip()


def _claude_template() -> str:
    return f"""# CLAUDE.md - SkeletonGraph-Enhanced Rules

{_sg_rules_block()}
"""


def _cursor_template() -> str:
    return f"""# .cursorrules - SkeletonGraph-Enhanced

{_sg_rules_block()}
"""


def _copilot_template() -> str:
    return f"""# Copilot Instructions - SkeletonGraph-Enhanced

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


# ── Register v3 subcommands ────────────────────────────────────────────
from .prepare import prepare as prepare_cmd
from .init import init_command as init_cmd

@app.command("hook")
@click.argument("event_name")
@click.option("--prompt", "-p", default="", help="Prompt text for pre-prompt event")
@click.option("--tool-name", default="", help="Tool name for post-tool event")
@click.option("--tool-args", default="", help="Tool arguments for post-tool event")
@click.option("--tool-output", default="", help="Tool output for post-tool event")
@click.option("--path", default=".", help="Project root directory")
def hook_cmd(event_name: str, prompt: str, tool_name: str, tool_args: str, tool_output: str, path: str):
    """Internal hook execution for Claude Code."""
    project_root = Path(path).resolve()
    from ..hooks.claude_code import hook_session_start, hook_pre_prompt, hook_post_tool_use, hook_session_end
    
    result = ""
    if event_name == "session_start":
        result = hook_session_start(project_root)
    elif event_name == "pre_prompt":
        result = hook_pre_prompt(project_root, prompt)
    elif event_name == "post_tool":
        result = hook_post_tool_use(project_root, tool_name, tool_args, tool_output)
    elif event_name == "session_end":
        result = hook_session_end(project_root)
        
    if result:
        click.echo(result)

app.add_command(prepare_cmd)
app.add_command(init_cmd)
app.add_command(hook_cmd)
