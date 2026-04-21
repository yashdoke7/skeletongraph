"""
Constraint store with hierarchical per-directory scoping.

Unique feature: No competitor has scoped constraint injection.

Reads constraints from:
  1. Project root: .skeletongraph/constraints.md (global)
  2. Subdirectories: <dir>/.skeletongraph/constraints.md (scoped)

When assembling context for a file in services/auth/, the output includes:
  - Global constraints (from project root)
  - Scoped constraints (from services/auth/ if they exist)

Constraints are injected into Zone 1 (primacy position) to ensure
they receive maximum LLM attention.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional


class ConstraintStore:
    """Hierarchical constraint manager.

    Load once at build time, query per-file at assembly time.
    """

    def __init__(self) -> None:
        self._global_constraints: str = ""
        self._scoped_constraints: Dict[str, str] = {}  # dir_path → constraints

    def load(self, project_root: Path) -> None:
        """Scan project for constraint files and load them.

        Looks for:
          - {project_root}/.skeletongraph/constraints.md (global)
          - {project_root}/<any_dir>/.skeletongraph/constraints.md (scoped)
        """
        # Global constraints
        global_file = project_root / ".skeletongraph" / "constraints.md"
        if global_file.exists():
            self._global_constraints = global_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

        # Scoped constraints — scan for nested .skeletongraph/constraints.md
        for constraints_file in project_root.rglob(".skeletongraph/constraints.md"):
            # Skip the root one (already loaded)
            if constraints_file == global_file:
                continue

            # Get the directory this constraint applies to
            # e.g., services/auth/.skeletongraph/constraints.md → services/auth
            sg_dir = constraints_file.parent  # .skeletongraph/
            scope_dir = sg_dir.parent         # services/auth/
            rel_scope = scope_dir.relative_to(project_root).as_posix()

            self._scoped_constraints[rel_scope] = constraints_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

    def get_constraints_for_file(self, file_path: str) -> str:
        """Get merged constraints for a specific file.

        Combines global constraints with any scoped constraints
        that apply to this file's directory.

        Args:
            file_path: Relative file path (e.g., "services/auth/handler.py").

        Returns:
            Merged constraint text, ready for Zone 1 injection.
        """
        parts = []

        if self._global_constraints:
            parts.append(self._global_constraints)

        # Find all scoped constraints that apply to this file
        file_dir = str(PurePosixPath(file_path).parent)
        for scope_dir, constraints in sorted(self._scoped_constraints.items()):
            if file_dir == scope_dir or file_dir.startswith(scope_dir + "/"):
                parts.append(f"# [{scope_dir}/ scope]\n{constraints}")

        return "\n\n".join(parts)

    def get_all_constraints(self) -> str:
        """Get all constraints (global only). Used when file scope is unknown."""
        return self._global_constraints

    @property
    def has_constraints(self) -> bool:
        return bool(self._global_constraints) or bool(self._scoped_constraints)

    @property
    def scope_count(self) -> int:
        """Number of scoped constraint files found."""
        return len(self._scoped_constraints)

    def set_global(self, text: str) -> None:
        """Set global constraints programmatically (for API/test use)."""
        self._global_constraints = text.strip()

    def save_global(self, project_root: Path) -> None:
        """Persist global constraints to disk."""
        sg_dir = project_root / ".skeletongraph"
        sg_dir.mkdir(parents=True, exist_ok=True)
        target = sg_dir / "constraints.md"
        target.write_text(self._global_constraints, encoding="utf-8")
