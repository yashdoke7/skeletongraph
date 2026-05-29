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


@app.command(name="doctor")
@click.option("--path", default=".", help="Project root to check (for index status).")
def cmd_doctor(path: str) -> None:
    """Validate the SG install end-to-end.

    Checks every layer that silently failed in the v3 ablation run:
      - Python version
      - Core deps (tree-sitter, click, mmh3, rich, tiktoken)
      - sentence-transformers + huggingface_hub (the silent-failure pair)
      - tree-sitter language parsers
      - sg on PATH
      - .skeletongraph index present (if --path given)

    Exits non-zero if anything's broken so CI can rely on it.
    """
    import sys
    import shutil
    from pathlib import Path

    ok = True
    def check(label, cond, fix=""):
        nonlocal ok
        mark = "OK " if cond else "FAIL"
        click.echo(f"  [{mark}] {label}")
        if not cond:
            ok = False
            if fix:
                click.echo(f"        fix: {fix}")

    click.echo("\nSkeletonGraph — install doctor\n")

    # Python version
    pv = sys.version_info
    check(f"Python {pv.major}.{pv.minor}.{pv.micro} >= 3.10",
          pv >= (3, 10),
          "use Python 3.10 or newer")

    # Core deps
    for mod in ("tree_sitter", "click", "mmh3", "rich", "tiktoken", "numpy"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except ImportError as e:
            check(f"import {mod}", False, f"pip install {mod}  ({e})")

    # sentence-transformers + huggingface_hub (the silent-failure pair)
    try:
        import sentence_transformers
        check(f"sentence-transformers {sentence_transformers.__version__}", True)
    except ImportError as e:
        check("sentence-transformers", False,
              f"pip install 'sentence-transformers>=3.0,<4'  ({e})")
    except Exception as e:
        # This is the BucketNotFoundError class — import succeeds at top but
        # fails on sub-import. Surface it explicitly.
        check("sentence-transformers load", False,
              f"transitive import broken: {type(e).__name__}: {e}\n"
              f"        fix: pip install 'huggingface_hub>=0.20,<0.30'")

    try:
        import huggingface_hub
        ver = huggingface_hub.__version__
        ok_ver = ver.startswith(("0.20", "0.21", "0.22", "0.23", "0.24",
                                  "0.25", "0.26", "0.27", "0.28", "0.29"))
        check(f"huggingface_hub {ver} in supported range (>=0.20,<0.30)",
              ok_ver,
              f"pip install 'huggingface_hub>=0.20,<0.30'  "
              f"(newer versions remove BucketNotFoundError → breaks datasets)")
    except ImportError:
        check("huggingface_hub", False, "pip install huggingface_hub")

    # Try loading the actual default embedder — catches model-download issues
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=False)
        check("load default embedder (all-MiniLM-L6-v2)", True)
    except Exception as e:
        check("load default embedder", False,
              f"{type(e).__name__}: {str(e)[:200]}\n"
              f"        fix: check network / HF_HOME cache permissions")

    # tree-sitter language parsers (at least Python — most repos)
    for lang in ("python", "javascript", "typescript", "java", "go", "rust"):
        try:
            __import__(f"tree_sitter_{lang}")
            check(f"tree-sitter parser: {lang}", True)
        except ImportError:
            check(f"tree-sitter parser: {lang}", False,
                  f"pip install tree-sitter-{lang}")

    # sg on PATH (so IDE installers can use bare `sg` in hook commands)
    sg = shutil.which("sg")
    check(f"`sg` on PATH ({sg})" if sg else "`sg` on PATH", bool(sg),
          "ensure Python Scripts dir is on PATH (otherwise hooks use full python invocation)")

    # Project index status (only if --path is a real project)
    proj = Path(path).resolve()
    if (proj / ".skeletongraph").exists():
        meta = proj / ".skeletongraph" / "meta.json"
        emb = proj / ".skeletongraph" / "embeddings.npz"
        check(f"index present at {proj.name}/.skeletongraph", True)
        check("  → embeddings.npz built", emb.exists(),
              f"sg index --path '{proj}' --force  "
              f"(rebuild after fixing sentence-transformers)")
    else:
        click.echo(f"  [info] no .skeletongraph index at {proj.name} "
                   f"(run `sg index --path '{proj}'` to build)")

    click.echo()
    if ok:
        click.echo("  All checks passed.")
        sys.exit(0)
    else:
        click.echo("  Fix the FAIL items above, then re-run `sg doctor`.")
        sys.exit(1)


@app.command(name="init")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--agent", "-a", default=None, help="IDE agent preset name")
@click.option("--non-interactive", is_flag=True, help="Skip prompts and use defaults")
@click.option("--constraints", default="", help="Initial constraint text to seed constraints.md")
def init_cmd(path: str, agent: str | None, non_interactive: bool, constraints: str):
    """Initialize project metadata and IDE integration files."""
    project_root = Path(path).resolve()
    from .init import run_init
    run_init(project_root, non_interactive=non_interactive, agent=agent)

    # Seed initial constraints from --constraints flag (source 1 per plan)
    if constraints.strip():
        from ..assembly.constraint_store import ConstraintStore
        cs = ConstraintStore()
        cs.load(project_root)
        cs.add_constraint(constraints.strip(), provenance="init-arg", confirmed=True)
        cs.save_global(project_root)
        console.print(f"[green]Constraint added:[/green] {constraints.strip()[:80]}")


def _import_artifact(dest: Path, label: str) -> None:
    """Open an editor so the user can paste an existing doc into `dest`.

    Only ever called when the matching --project-summary / --constraints flag
    is passed to `sg build`. Bare `sg build` never reaches this.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.read_text(encoding="utf-8") if dest.exists() else ""
    seed = existing or f"# Paste your existing {label} here, then save and close.\n"
    pasted = click.edit(seed)
    if pasted is None or not pasted.strip() or pasted.strip().startswith("# Paste your"):
        console.print(f"  [yellow]Skipped {label} import (nothing pasted).[/yellow]")
        return
    dest.write_text(pasted, encoding="utf-8")
    console.print(f"  [green]Imported {label}[/green] -> {dest}")


def _seed_project_md_from_readme(project_root: Path) -> bool:
    """Derive a minimal project.md from the README when none exists.

    Heuristic only — no prompt, no LLM: first heading + first paragraph. SG's
    job here is compression — hand the agent a short project DNA so it does not
    spend a turn reading the whole README. The agent can refine it later.
    Returns True if a file was written.
    """
    sg_dir = project_root / ".skeletongraph"
    project_md = sg_dir / "project.md"
    if project_md.exists():
        return False
    readme = next((project_root / n for n in
                   ("README.md", "README.rst", "README.txt", "readme.md")
                   if (project_root / n).exists()), None)
    if readme is None:
        return False
    try:
        lines = [l.rstrip() for l in
                 readme.read_text(encoding="utf-8", errors="replace").splitlines()]
    except Exception:
        return False
    title, para = "", []
    for l in lines:
        s = l.strip().lstrip("#").strip()
        if s and not title:
            title = s
        elif title and s:
            para.append(s)
        elif title and para:
            break
    summary = " ".join(para)[:600] or "[Derived from README — refine as needed.]"
    sg_dir.mkdir(parents=True, exist_ok=True)
    project_md.write_text(
        f"# {project_root.name}\n"
        f"**Goal:** {title or project_root.name}\n\n"
        f"## Summary\n{summary}\n\n"
        f"<!-- Auto-derived from README. The agent may refine this. -->\n",
        encoding="utf-8",
    )
    return True


@app.command()
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--project-summary", "import_summary", is_flag=True,
              help="After indexing, open an editor to paste an existing "
                   "project summary into .skeletongraph/project.md")
@click.option("--constraints", "import_constraints", is_flag=True,
              help="After indexing, open an editor to paste existing "
                   "constraints into .skeletongraph/constraints.md")
def build(path: str, import_summary: bool, import_constraints: bool):
    """Build the code index for a project.

    Bare `sg build` indexes source files only — it never prompts and never
    creates project.md (a first build with no project.md is fully supported;
    SkeletonGraph derives a fallback, and the agent can refine it later).

    Pass --project-summary or --constraints ONLY if you already have those
    docs and want to import them; an editor opens for you to paste them.
    """
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

    # Post-build: update architecture.md with store data
    sg_dir = project_root / ".skeletongraph"
    if sg_dir.exists():
        from .init import _generate_architecture
        arch_path = sg_dir / "architecture.md"
        arch_md = _generate_architecture(project_root, store)
        arch_path.write_text(arch_md, encoding="utf-8")
        console.print(f"  [dim]Updated {arch_path.relative_to(project_root)}[/dim]")

    # Fallback project DNA: derive a minimal project.md from the README if none
    # exists (heuristic, no prompt, no LLM — the agent refines it later).
    if _seed_project_md_from_readme(project_root):
        console.print("  [dim]Derived project.md from README[/dim]")

    # Opt-in artifact import (only when the flag was explicitly passed)
    if import_summary:
        _import_artifact(sg_dir / "project.md", "project summary")
    if import_constraints:
        _import_artifact(sg_dir / "constraints.md", "constraints")


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

    # ── Ollama (Tier-0.5) ─────────────────────────────────────────────────
    ollama_base = getattr(config, "ollama_base_url", "http://localhost:11434")
    enable_local = getattr(config, "enable_local_summary", True)
    ollama_ok = False
    if enable_local:
        try:
            from ..summary.ollama import is_ollama_available, list_ollama_models
            ollama_ok = is_ollama_available(ollama_base)
        except Exception:
            pass
    add(
        "ollama",
        ollama_ok,
        f"available at {ollama_base}" if ollama_ok else (
            f"not reachable at {ollama_base} (optional — start with: ollama serve)"
            if enable_local else "disabled (enable_local_summary=False)"
        ),
    )

    # ── Summary coverage ──────────────────────────────────────────────────
    if store is not None:
        total_fn = store.meta.total_functions
        summarized = store.summaries.count
        pct = round(100 * summarized / total_fn, 1) if total_fn else 0
        sg_dir = project_root / ".skeletongraph"
        from ..summary.queue import queue_size
        pending = queue_size(sg_dir)
        add(
            "summaries",
            summarized > 0 or total_fn == 0,
            f"{summarized}/{total_fn} ({pct}%) summarized"
            + (f"; {pending} queued" if pending else ""),
        )

    payload = {
        "ok": all(c["ok"] for c in checks if c["name"] not in {"cli_api_key", "ollama"}),
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
        "ollama": {
            "available": ollama_ok,
            "base_url": ollama_base,
            "model": getattr(config, "ollama_summary_model", "qwen2.5-coder:1.5b"),
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
        status_str = "OK" if check["ok"] else "WARN"
        table.add_row(check["name"], status_str, check["detail"])
    console.print(table)

    model_table = Table(title="Configured Models")
    model_table.add_column("Surface", style="cyan")
    model_table.add_column("SLM")
    model_table.add_column("MLM")
    model_table.add_column("LLM")
    model_table.add_row("IDE", config.slm_model, config.mlm_model, config.llm_model)
    model_table.add_row("CLI", config.cli_slm_model, config.cli_mlm_model, config.cli_llm_model)
    console.print(model_table)

    if ollama_ok:
        try:
            models = list_ollama_models(ollama_base)
            if models:
                console.print(
                    f"\n[dim]Ollama models available: {', '.join(models[:6])}"
                    + (" ..." if len(models) > 6 else "") + "[/dim]"
                )
        except Exception:
            pass


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
@click.option("--plan-first/--no-plan-first", default=False, help="Use SLM tool planning before execution")
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
    config = engine.get_config()
    if dry_run and not plan_first:
        config.enable_slm_fallback = False

    result = engine.query(prompt, delivery="cli", force_slm=plan_first)

    if not result.success:
        console.print(f"[yellow]{result.error}[/yellow]")
        return

    routing_mode = "manual" if tier else "auto"
    selected_tier = tier or result.model_tier.value
    selected_model = config.get_cli_model_for_tier(selected_tier)
    error_followup_data = load_error_followup(project_root) if error_followup else None
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

    if error_followup_data:
        followup_text = "\n".join(f"- {err}" for err in error_followup_data.errors[:10])
        result.context_text = (
            f"{result.context_text}\n\n---\n\n"
            "## Previous Error Follow-up\n"
            f"Source: {error_followup_data.source}\n"
            f"Original prompt: {error_followup_data.prompt}\n"
            f"{followup_text}"
        )
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
@click.option(
    "--tier", "-t",
    type=click.Choice(["local", "cloud"], case_sensitive=False),
    default="cloud",
    help="Summary tier: local = Ollama (free, on-device), cloud = LLM API",
)
@click.option("--model", "-m", default=None, help="Override model (cloud: litellm name; local: ollama model name)")
@click.option("--force", is_flag=True, help="Re-summarize all functions, not just missing ones")
@click.option(
    "--api-key",
    envvar=["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    default=None,
    help="API key for the cloud LLM provider",
)
def summarize(path: str, tier: str, model: str | None, force: bool, api_key: str | None):
    """Generate summaries for indexed functions.

    \b
    sg summarize                       cloud tier (LLM API, default model)
    sg summarize --tier local          Ollama Tier-0.5 (free, no API key)
    sg summarize --tier local --model qwen2.5-coder:7b   larger Ollama model
    sg summarize --tier cloud --model gemini/gemini-2.5-flash
    sg summarize --force               re-summarize everything
    """
    project_root = Path(path).resolve()
    tier = tier.lower()

    from ..storage.local import load_index
    from ..config import load_config

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `skeletongraph build` first.")
        return

    config = load_config(project_root)

    # ── Tier: local (Ollama Tier-0.5) ────────────────────────────────────
    if tier == "local":
        from ..summary.ollama import is_ollama_available, batch_generate_ollama

        ollama_base = config.ollama_base_url
        ollama_model = model or config.ollama_summary_model

        console.print(f"[bold]> Ollama Tier-0.5 summarization[/bold]")
        console.print(f"  Server: {ollama_base}")
        console.print(f"  Model:  {ollama_model}")

        if not is_ollama_available(ollama_base):
            console.print(
                f"[red]Ollama not reachable at {ollama_base}.[/red] "
                "Start Ollama with: ollama serve"
            )
            return

        # Collect candidates
        candidates = []
        for fqn, sk in store.skeleton_table.items():
            if not force and store.summaries.get(fqn):
                continue
            if sk.body_token_estimate < 10:
                continue
            candidates.append(sk)

        if not candidates:
            console.print("[green]All functions already summarized.[/green]")
            return

        console.print(f"  Summarizing [cyan]{len(candidates)}[/cyan] functions...\n")

        import time as _time
        start = _time.time()
        summarized = 0
        errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Summarizing (Ollama)...", total=len(candidates))

            for i, sk in enumerate(candidates):
                short = sk.fqn.split("::")[-1] if "::" in sk.fqn else sk.fqn
                progress.update(task, completed=i, description=f"[{i+1}/{len(candidates)}] {short}")

                # Read body
                file_path = project_root / sk.file_path
                body = ""
                try:
                    if file_path.exists():
                        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        body = "\n".join(lines[max(0, sk.line_start - 1):sk.line_end])
                except Exception:
                    pass

                from ..summary.ollama import generate_summary_ollama
                summary = generate_summary_ollama(
                    fqn=sk.fqn,
                    signature=sk.signature,
                    body=body,
                    model=ollama_model,
                    base_url=ollama_base,
                    timeout=config.ollama_timeout,
                )
                if summary:
                    store.summaries.set(sk.fqn, summary)
                    summarized += 1
                else:
                    errors += 1

            progress.update(task, completed=len(candidates))

        elapsed = _time.time() - start

        if summarized > 0:
            sg_dir = project_root / ".skeletongraph"
            store.summaries.save(sg_dir)

        table = Table(title="Ollama Summarization Complete", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Tier", "0.5 (Ollama local)")
        table.add_row("Model", ollama_model)
        table.add_row("Summarized", str(summarized))
        table.add_row("Skipped (already done)", str(len(store.skeleton_table) - len(candidates)))
        table.add_row("Errors / timeouts", str(errors))
        table.add_row("Duration", f"{elapsed:.1f}s")
        table.add_row("API cost", "$0.00")
        console.print(table)
        return

    # ── Tier: cloud (LLM API, Tier-1) ────────────────────────────────────
    from ..llm.summarizer import summarize_index
    from ..llm.provider import LLMConfig

    cloud_model = model or config.summary_model or "gemini/gemini-2.5-flash"
    cfg = LLMConfig(model=cloud_model, api_key=api_key or None)
    console.print(f"[bold]> Cloud Tier-1 summarization with {cloud_model}[/bold]")

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
    table.add_row("Tier", "1 (cloud LLM)")
    table.add_row("Model", cloud_model)
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
@click.option("--port", default=3500, help="Server port (unused in stdio mode)")
def serve(path: str, port: int):
    """Start the MCP server for IDE integration (stdio transport)."""
    import sys as _sys
    from rich.console import Console as _Console
    stderr_console = _Console(stderr=True)

    project_root = Path(path).resolve()

    stderr_console.print(f"[bold]> SkeletonGraph MCP server[/bold]  (stdio)")
    stderr_console.print(f"  Project: {project_root}")
    stderr_console.print(f"  Tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log")

    from ..server.mcp import serve as mcp_serve
    from ..config import load_config
    cfg = load_config(project_root)
    mcp_serve(project_root, cfg)


@app.command()
@click.argument("platform", required=False, default=None)
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--ide", default=None,
              type=click.Choice([
                  "claude-code", "cursor",
                  "cline", "roo", "zed", "continue", "copilot", "windsurf",
                  # legacy names still accepted
                  "claude", "antigravity", "codex", "kiro", "opencode",
              ], case_sensitive=False),
              help="Target IDE (overrides auto-detection)")
@click.option("--all-detected", is_flag=True,
              help="Install for all detected IDEs automatically")
def install(platform: str, path: str, ide: str, all_detected: bool):
    """Configure IDE integrations: hooks + MCP server registration.

    \b
    sg install --ide claude-code    write Claude Code hooks + MCP
    sg install --ide cursor         write Cursor hooks + MCP
    sg install --ide cline          write MCP config + rules block
    sg install --all-detected       auto-detect and configure all IDEs
    sg install                      same as --all-detected
    """
    project_root = Path(path).resolve()

    # --ide takes priority; --all-detected or no args → auto-detect
    if ide:
        targets = [ide]
    elif platform:
        targets = [platform]
    else:
        from ..install.detect import detect_ides
        targets = detect_ides(project_root) or ["claude-code"]

    if not targets:
        console.print("[yellow]No supported IDEs detected.[/yellow]")
        console.print("Run `sg install --ide claude-code` to install manually.")
        return

    for target in targets:
        _run_installer(target, project_root)

    console.print(
        f"\n[bold green][OK] Installation complete.[/bold green] "
        f"Restart your editor to activate SkeletonGraph."
    )


def _run_installer(ide_name: str, project_root: Path) -> None:
    """Dispatch to the correct installer module."""
    # Normalize legacy names
    _aliases = {
        "claude": "claude-code",
        "antigravity": "copilot",
        "codex": "copilot",
        "kiro": "cline",
        "opencode": "cline",
    }
    ide_name = _aliases.get(ide_name, ide_name)

    console.print(f"\n[bold]Installing for:[/bold] {ide_name}")

    try:
        if ide_name == "claude-code":
            from ..install.claude_code import install as cc_install
            written = cc_install(project_root)
        elif ide_name == "cursor":
            from ..install.cursor import install as cur_install
            written = cur_install(project_root)
        elif ide_name in ("cline", "roo", "zed", "continue", "copilot", "windsurf"):
            from ..install.mcp_only import install as mcp_install
            written = mcp_install(ide_name, project_root)
        else:
            console.print(f"  [yellow]No installer for IDE '{ide_name}'[/yellow]")
            return

        for f in written:
            console.print(f"  [green][OK][/green] {f}")

    except Exception as e:
        console.print(f"  [red][FAIL][/red] {e}")


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
def watch(path: str):
    """Start a background daemon to auto-reindex files on save."""
    project_root = Path(path).resolve()
    from ..daemon import start_daemon
    start_daemon(project_root)


# Register additional commands from submodules
from .prepare import prepare as _prepare_command
app.add_command(_prepare_command)


# ── New canonical commands (P1/P2) ──────────────────────────────────────


@app.command(name="index")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--incremental", "-i", is_flag=True, help="Only re-index changed files")
def index_cmd(path: str, incremental: bool):
    """Index the project (alias for build / update)."""
    project_root = Path(path).resolve()
    if incremental:
        from ..build import update_index
        store = update_index(project_root)
        console.print(f"[green][OK][/green] {store.meta.total_functions} functions, {store.meta.total_edges} edges")
    else:
        from ..build import build_index, discover_files
        files = discover_files(project_root)
        if not files:
            console.print("[yellow]No supported files found.[/yellow]")
            return
        console.print(f"  Found [cyan]{len(files)}[/cyan] source files")
        store = build_index(project_root)
        console.print(f"[green][OK][/green] {store.meta.total_files} files, {store.meta.total_functions} functions, {store.meta.total_edges} edges")


@app.command(name="overview")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--top-n", "-n", default=20, help="Number of top functions to show")
@click.option("--json-output", is_flag=True, help="Machine-readable JSON")
def overview_cmd(path: str, top_n: int, json_output: bool):
    """Show project skeleton: top functions, constraints, session digest."""
    project_root = Path(path).resolve()
    sg_dir = project_root / ".skeletongraph"

    from ..storage.local import load_index
    from ..session.log import read_log, format_log_digest

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `sg index` first.")
        return

    if json_output:
        scores = store.pagerank_scores or {}
        top_fqns = sorted(scores, key=lambda f: -scores[f])[:top_n]
        if not top_fqns:
            top_fqns = list(store.skeleton_table.keys())[:top_n]
        payload = {
            "files": store.meta.total_files,
            "functions": store.meta.total_functions,
            "edges": store.meta.total_edges,
            "languages": list(store.meta.languages),
            "top_functions": [
                {"fqn": fqn, "pagerank": round(scores.get(fqn, 0), 4)}
                for fqn in top_fqns
            ],
            "constraints": store.constraints.get_all_constraints() if store.constraints else "",
        }
        console.print(json.dumps(payload, indent=2))
        return

    table = Table(title="Project Overview", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files", str(store.meta.total_files))
    table.add_row("Functions", str(store.meta.total_functions))
    table.add_row("Edges", str(store.meta.total_edges))
    table.add_row("Languages", ", ".join(store.meta.languages))
    console.print(table)

    if store.constraints and store.constraints.has_constraints:
        console.print(Panel(store.constraints.get_all_constraints(), title="Constraints (Zone 1)", border_style="yellow"))

    scores = store.pagerank_scores or {}
    top_fqns = sorted(scores, key=lambda f: -scores[f])[:top_n]
    if not top_fqns:
        top_fqns = list(store.skeleton_table.keys())[:top_n]

    fn_table = Table(title=f"Top {top_n} Functions by PageRank")
    fn_table.add_column("FQN", style="cyan")
    fn_table.add_column("Signature")
    fn_table.add_column("PageRank", style="dim")
    for fqn in top_fqns:
        sk = store.skeleton_table.get(fqn)
        if sk:
            fn_table.add_row(fqn, sk.signature[:60], f"{scores.get(fqn, 0):.4f}")
    console.print(fn_table)

    entries = read_log(sg_dir, last_n=5)
    digest = format_log_digest(entries, max_turns=5)
    if digest:
        console.print(Panel(digest, title="Recent turns", border_style="dim"))


@app.command(name="search")
@click.argument("query")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--top-n", "-n", default=15, help="Max results")
@click.option("--file-filter", "-f", default="", help="Restrict to files matching this substring")
@click.option("--json-output", is_flag=True, help="Machine-readable JSON")
def search_cmd(query: str, path: str, top_n: int, file_filter: str, json_output: bool):
    """Hybrid search: BM25 + graph centrality. PREFERRED over grep."""
    project_root = Path(path).resolve()

    from ..engine import SGEngine

    engine = SGEngine(project_root=project_root)
    try:
        result = engine.heuristic_query(query, top_n=top_n, file_filter=file_filter or None)
    except RuntimeError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    candidates = result.candidates
    if not candidates:
        console.print(f"No results for: [cyan]{query!r}[/cyan]")
        return

    if json_output:
        payload = [
            {
                "fqn": c.skeleton.fqn,
                "file": c.skeleton.file_path,
                "line": c.skeleton.line_start,
                "signature": c.skeleton.signature,
                "score": round(c.score, 4),
            }
            for c in candidates
        ]
        console.print(json.dumps(payload, indent=2))
        return

    table = Table(title=f"Search: {query!r}")
    table.add_column("#", style="dim")
    table.add_column("FQN", style="cyan")
    table.add_column("Signature")
    table.add_column("Score", style="dim")
    for i, c in enumerate(candidates[:top_n], 1):
        sk = c.skeleton
        table.add_row(str(i), sk.fqn, sk.signature[:70], f"{c.score:.3f}")
    console.print(table)


@app.command(name="get")
@click.argument("fqn")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--no-callers", is_flag=True, help="Skip caller listing")
def get_cmd(fqn: str, path: str, no_callers: bool):
    """Get a specific function by FQN: signature, summary, callers."""
    project_root = Path(path).resolve()

    from ..storage.local import load_index

    store = load_index(project_root)
    if store is None:
        console.print("[yellow]No index found.[/yellow] Run `sg index` first.")
        return

    sk = store.skeleton_table.get(fqn)
    if not sk:
        for k, v in store.skeleton_table.items():
            if k.endswith(fqn) or fqn in k:
                sk, fqn = v, k
                break
    if not sk:
        console.print(f"[red]Not found:[/red] {fqn}")
        console.print("Tip: use `sg search <name>` to find the exact FQN.")
        return

    console.print(f"[bold]{fqn}[/bold]")
    console.print(f"  File: {sk.file_path}:{sk.line_start}")
    console.print(f"  Signature: {sk.signature}")
    if sk.docstring:
        console.print(f"  Docstring: {sk.docstring.strip()[:200]}")
    summary = store.summaries.get(fqn) or ""
    if summary:
        console.print(f"  Summary: {summary[:200]}")

    if not no_callers:
        callers = [
            k for k, v in store.skeleton_table.items()
            if fqn in getattr(v, "callees", [])
        ]
        if callers:
            console.print(f"\n  Callers ({len(callers)}):")
            for c in callers[:6]:
                csk = store.skeleton_table.get(c)
                if csk:
                    console.print(f"    {csk.signature}")

    callees = list(getattr(sk, "callees", []))
    if callees:
        console.print(f"\n  Calls ({len(callees)}):")
        for callee in callees[:6]:
            csk = store.skeleton_table.get(callee)
            if csk:
                console.print(f"    {csk.signature}")
            else:
                console.print(f"    {callee}")


@app.command(name="expand")
@click.argument("target")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--max-tokens", default=4000, help="Token budget")
@click.option("--out", "-o", default=None, help="Write output to file")
def expand_cmd(target: str, path: str, max_tokens: int, out: str | None):
    """Expand a function, file, or line range. PREFERRED over reading full files.

    TARGET formats:
      src/file.py::MyClass.my_method  (function body)
      src/file.py                     (full file, token-capped)
      src/file.py:42-80               (line range)
    """
    project_root = Path(path).resolve()

    from ..engine import SGEngine

    engine = SGEngine(project_root=project_root)

    # Parse range syntax
    start_line = end_line = None
    expand_target = target
    if ":" in target and "::" not in target:
        parts = target.rsplit(":", 1)
        range_part = parts[-1]
        if "-" in range_part and all(p.strip().isdigit() for p in range_part.split("-", 1)):
            start_line, end_line = (int(p.strip()) for p in range_part.split("-", 1))
            expand_target = parts[0]

    try:
        text = engine.expand(
            target=expand_target,
            start_line=start_line,
            end_line=end_line,
            max_tokens=max_tokens,
        )
    except RuntimeError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    if out:
        Path(out).write_text(text, encoding="utf-8")
        console.print(f"[green][OK][/green] Written to {out}")
    else:
        console.print(text)


@app.command(name="constraint")
@click.argument("action", type=click.Choice(["list", "propose", "confirm", "remove", "aggregate"]))
@click.argument("value", default="")
@click.option("--path", "-p", default=".", help="Project root directory")
def constraint_cmd(action: str, value: str, path: str):
    """Manage project constraints.

    \b
    sg constraint list               — show all constraints
    sg constraint propose "text"     — add a proposal
    sg constraint confirm <id>       — promote a proposal to confirmed
    sg constraint remove <id>        — remove a constraint
    sg constraint aggregate          — import from IDE rule files
    """
    project_root = Path(path).resolve()
    sg_dir = project_root / ".skeletongraph"

    from ..assembly.constraint_store import ConstraintStore

    cs = ConstraintStore()
    cs.load(project_root)

    if action == "list":
        items = cs.list_constraints(include_proposed=True)
        if not items:
            raw = cs.get_all_constraints()
            if raw:
                console.print(Panel(raw, title="Constraints"))
            else:
                console.print("[dim]No constraints defined.[/dim]")
            return
        table = Table(title="Constraints")
        table.add_column("ID", style="dim")
        table.add_column("Status", style="cyan")
        table.add_column("Provenance", style="dim")
        table.add_column("Text")
        for c in items:
            table.add_row(
                c.id,
                "[green]confirmed[/green]" if c.confirmed else "[yellow]proposed[/yellow]",
                c.provenance,
                c.text.strip()[:80],
            )
        console.print(table)

    elif action == "propose":
        if not value:
            console.print("[red]Error:[/red] provide constraint text as argument")
            return
        c = cs.propose_constraint(value)
        cs.save_global(project_root)
        console.print(f"[green]Proposed[/green] id={c.id}  Run `sg constraint confirm {c.id}` to promote.")

    elif action == "confirm":
        if not value:
            console.print("[red]Error:[/red] provide constraint ID as argument")
            return
        ok = cs.confirm_constraint(value, project_root=project_root)
        if ok:
            cs.save_global(project_root)
            console.print(f"[green]Confirmed[/green] id={value}  (promoted to decisions.md)")
        else:
            console.print(f"[yellow]Not found:[/yellow] {value}")

    elif action == "remove":
        if not value:
            console.print("[red]Error:[/red] provide constraint ID as argument")
            return
        ok = cs.remove_constraint(value)
        if ok:
            cs.save_global(project_root)
            console.print(f"[green]Removed[/green] id={value}")
        else:
            console.print(f"[yellow]Not found:[/yellow] {value}")

    elif action == "aggregate":
        n = cs.aggregate_from_ide_rules(project_root)
        cs.save_global(project_root)
        console.print(f"[green]Imported[/green] {n} constraints from IDE rule files.")


@app.command(name="log")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--last-n", "-n", default=10, help="Number of entries to show")
@click.option("--session-id", default="", help="Specific session ID")
def log_cmd(path: str, last_n: int, session_id: str):
    """Show recent session log entries."""
    project_root = Path(path).resolve()
    sg_dir = project_root / ".skeletongraph"

    from ..session.log import read_log

    entries = read_log(sg_dir, session_id=session_id or None, last_n=last_n)
    if not entries:
        console.print("[dim]No session log entries found.[/dim]")
        return

    table = Table(title="Session Log")
    table.add_column("Turn", style="dim")
    table.add_column("Prompt", style="cyan")
    table.add_column("Files touched")
    table.add_column("Summary")
    for e in entries:
        table.add_row(
            str(e.turn_index),
            e.user_prompt[:60],
            ", ".join(e.files_touched[:3]) or "—",
            e.summary[:60] or "—",
        )
    console.print(table)


@app.command("hook")
@click.argument("event_name",
                type=click.Choice([
                    "session_start", "user_prompt_submit",
                    "post_tool_use", "file_changed",
                    # legacy aliases kept for backward compat
                    "pre_prompt", "post_tool", "session_end",
                ]))
@click.option("--path", default=".", help="Project root directory")
def hook_cmd(event_name: str, path: str):
    """Handle Claude Code / Cursor hook events.

    Reads JSON event data from stdin, writes JSON response to stdout.
    Always exits 0 — never blocks the agent.

    \b
    Events:
      session_start       → inject "use SG" systemMessage
      user_prompt_submit  → inject sg_overview as additionalContext
      post_tool_use       → append to session log
      file_changed        → background incremental re-index
    """
    import json as _json
    import sys as _sys

    project_root = Path(path).resolve()

    # Read event data from stdin (Claude Code sends JSON)
    event_data: dict = {}
    try:
        raw = _sys.stdin.read()
        if raw.strip():
            event_data = _json.loads(raw)
    except Exception:
        pass  # proceed with empty event_data

    from ..hooks.claude_code import (
        hook_session_start,
        hook_user_prompt_submit,
        hook_post_tool_use,
        hook_file_changed,
    )

    try:
        if event_name in ("session_start",):
            result = hook_session_start(project_root, event_data)
        elif event_name in ("user_prompt_submit", "pre_prompt"):
            result = hook_user_prompt_submit(project_root, event_data)
        elif event_name in ("post_tool_use", "post_tool"):
            result = hook_post_tool_use(project_root, event_data)
        elif event_name in ("file_changed",):
            result = hook_file_changed(project_root, event_data)
        elif event_name in ("session_end",):
            result = {}
        else:
            result = {}
    except Exception:
        result = {}  # never block the agent

    if result:
        click.echo(_json.dumps(result))


if __name__ == "__main__":
    app()
