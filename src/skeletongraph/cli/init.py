"""
CLI init command: generates project.md (L0) and architecture.md (L1).

Auto-triggered by `sg build` on first run if project.md doesn't exist.
Can also be run standalone: `sg init`

- project.md: User answers 4 prompts (goal, constraints, phase, decisions)
- architecture.md: Auto-generated from directory structure + IndexStore
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def run_init(project_root: Path, store=None, non_interactive: bool = False) -> bool:
    """Run init logic. Called from sg build on first run.

    Args:
        project_root: Project root directory.
        store: Optional IndexStore (if build already ran).
        non_interactive: If True, skip prompts and use defaults.

    Returns:
        True if project.md was created/updated.
    """
    sg_dir = project_root / ".skeletongraph"
    sg_dir.mkdir(parents=True, exist_ok=True)

    project_path = sg_dir / "project.md"
    arch_path = sg_dir / "architecture.md"

    # Create session and domain directories
    (sg_dir / "session").mkdir(exist_ok=True)
    (sg_dir / "domain").mkdir(exist_ok=True)
    (sg_dir / "eval").mkdir(exist_ok=True)

    # ── project.md ───────────────────────────────────────────────────
    project_name = project_root.name

    if non_interactive:
        goal = f"{project_name} project"
        constraints = "[Add constraints here]"
        phase = _detect_phase(project_root, store)
        decisions = "[Add key architectural decisions here]"
    else:
        console.print()
        console.print("[bold cyan]SkeletonGraph Init[/bold cyan]")
        console.print("Setting up L0 (project DNA) — loaded every turn, every mode.\n")

        # Prompt 1: REQUIRED
        goal = Prompt.ask(
            "[bold]What does this project do?[/bold] (1-2 sentences)",
        )
        while not goal.strip():
            console.print("[red]Goal is required — it's loaded on every single turn.[/red]")
            goal = Prompt.ask("[bold]What does this project do?[/bold]")

        # Prompt 2: Optional
        constraints = Prompt.ask(
            "[bold]What are 2-3 constraints that must never be violated?[/bold]\n"
            "  (e.g., 'Public API backward compat', 'No external deps without approval')\n"
            "  Press Enter to skip",
            default="",
        )
        if not constraints.strip():
            constraints = "[Add constraints here]"

        # Prompt 3: With default
        phase = _detect_phase(project_root, store)
        phase_input = Prompt.ask(
            f"[bold]Project phase?[/bold] [start/active/maintenance/refactor]",
            default=phase,
        )
        if phase_input.strip():
            phase = phase_input.strip()

        # Prompt 4: Optional
        decisions = Prompt.ask(
            "[bold]Any key architectural decisions to preserve?[/bold]\n"
            "  Press Enter to skip",
            default="",
        )
        if not decisions.strip():
            decisions = "[Add key architectural decisions here]"

    # Detect stack
    stack = _detect_stack(project_root)
    project_type = _detect_project_type(project_root)

    # Write project.md
    project_md = (
        f"# {project_name}\n"
        f"**Type:** {project_type}\n"
        f"**Goal:** {goal}\n"
        f"**Stack:** {stack}\n"
        f"**Phase:** {phase}\n"
        f"\n"
        f"## Fundamental Constraints\n"
        f"- {constraints}\n"
        f"\n"
        f"## Key Architectural Decisions\n"
        f"- {decisions}\n"
        f"\n"
        f"## Current Focus\n"
        f"[Updated by session memory]\n"
    )
    project_path.write_text(project_md, encoding="utf-8")

    # ── architecture.md ──────────────────────────────────────────────
    arch_md = _generate_architecture(project_root, store)
    arch_path.write_text(arch_md, encoding="utf-8")

    if not non_interactive:
        console.print(f"\n[green]✓[/green] Created {project_path.relative_to(project_root)}")
        console.print(f"[green]✓[/green] Created {arch_path.relative_to(project_root)}")

    return True


def _detect_phase(project_root: Path, store=None) -> str:
    """Infer project phase from index size."""
    if store and hasattr(store, 'meta'):
        if store.meta.total_functions > 50:
            return "active"
        return "start"
    # Check if there are many source files
    py_count = len(list(project_root.rglob("*.py")))
    js_count = len(list(project_root.rglob("*.js"))) + len(list(project_root.rglob("*.ts")))
    total = py_count + js_count
    if total > 20:
        return "active"
    return "start"


def _detect_stack(project_root: Path) -> str:
    """Detect tech stack from config files."""
    parts = []

    # Python
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        if "python" in text.lower():
            # Try to find python version
            import re
            match = re.search(r'python\s*[><=]+\s*"?(\d+\.\d+)', text)
            if match:
                parts.append(f"Python {match.group(1)}")
            else:
                parts.append("Python")

    requirements = project_root / "requirements.txt"
    setup_py = project_root / "setup.py"
    if not parts and (requirements.exists() or setup_py.exists()):
        parts.append("Python")

    # Node.js
    package_json = project_root / "package.json"
    if package_json.exists():
        parts.append("Node.js")
        try:
            import json
            pkg = json.loads(package_json.read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "react" in deps:
                parts.append("React")
            elif "vue" in deps:
                parts.append("Vue")
            elif "next" in deps:
                parts.append("Next.js")
        except Exception:
            pass

    # Go
    if (project_root / "go.mod").exists():
        parts.append("Go")

    # Rust
    if (project_root / "Cargo.toml").exists():
        parts.append("Rust")

    return ", ".join(parts) if parts else "Unknown"


def _detect_project_type(project_root: Path) -> str:
    """Detect project type."""
    if (project_root / "pyproject.toml").exists():
        text = (project_root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
        if "library" in text.lower() or "[project]" in text:
            return "Library/Package"
    if (project_root / "manage.py").exists():
        return "Django Web App"
    if (project_root / "package.json").exists():
        return "Node.js Application"
    if (project_root / "Dockerfile").exists():
        return "Containerized Service"
    return "Application"


def _generate_architecture(project_root: Path, store=None) -> str:
    """Auto-generate architecture.md from directory structure."""
    lines = [f"# Architecture — {project_root.name}\n"]

    # Module list from directory structure
    lines.append("## Modules\n")

    # Scan top-level directories
    src_dirs = []
    for child in sorted(project_root.iterdir()):
        if child.is_dir() and not child.name.startswith((".")) and child.name not in (
            "node_modules", "__pycache__", ".git", "venv", "env", ".venv",
            "dist", "build", ".tox", ".pytest_cache", ".mypy_cache",
        ):
            # Count source files
            py_files = list(child.rglob("*.py"))
            js_files = list(child.rglob("*.js")) + list(child.rglob("*.ts"))
            total = len(py_files) + len(js_files)
            if total > 0:
                src_dirs.append((child.name, total))

    if src_dirs:
        for name, count in src_dirs:
            lines.append(f"- **{name}/** ({count} source files)")
    else:
        lines.append("- [No source directories detected]")

    # If we have a store, add function count per file
    if store and hasattr(store, 'file_skeletons'):
        lines.append("\n## Key Files\n")
        # Top 10 files by function count
        file_counts = [
            (fp, len(fs.all_skeletons))
            for fp, fs in store.file_skeletons.items()
        ]
        file_counts.sort(key=lambda x: -x[1])
        for fp, count in file_counts[:10]:
            lines.append(f"- `{fp}` ({count} functions)")

    lines.append("\n## Interfaces\n[Add module interfaces here]\n")
    lines.append("## Data Flow\n[Add data flow description here]\n")

    return "\n".join(lines)


@click.command("init")
@click.option("--path", "-p", default=".", help="Project root directory")
@click.option("--non-interactive", is_flag=True, help="Skip prompts, use defaults")
def init_command(path: str, non_interactive: bool):
    """Initialize project.md and architecture.md for SkeletonGraph."""
    project_root = Path(path).resolve()

    if not project_root.is_dir():
        console.print(f"[red]Error:[/red] {project_root} is not a directory")
        sys.exit(1)

    run_init(project_root, non_interactive=non_interactive)
