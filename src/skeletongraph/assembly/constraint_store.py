"""
Constraint store: hierarchical scoping + IDE rule aggregation + propose/confirm.

Sources (three paths per the plan):
  1. sg init --constraints "..." (init-arg)
  2. IDE rule files aggregated at index time → constraints.md
  3. Model-driven proposals via sg_constraint(action="propose")

Constraints are stored in .skeletongraph/constraints.md with structured markers
so SG can manage individual items (confirm/remove) while keeping the file
human-readable and editable.

Never strictly enforced — Zone 1 visibility is the mechanism.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
import time
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional


# ── Constraint dataclass ─────────────────────────────────────────────────


@dataclass
class Constraint:
    """Single constraint entry."""
    id: str
    text: str
    provenance: str        # e.g. "CLAUDE.md", "init-arg", "model-proposed"
    confirmed: bool = True

    def to_block(self) -> str:
        confirmed_str = "true" if self.confirmed else "false"
        return (
            f"<!-- sg:constraint id={self.id} confirmed={confirmed_str}"
            f" provenance={self.provenance} -->\n"
            f"{self.text.strip()}\n"
            f"<!-- /sg:constraint -->"
        )


# ── ConstraintStore ──────────────────────────────────────────────────────


class ConstraintStore:
    """Hierarchical constraint manager with IDE rule aggregation.

    Load once at build time, query per-file at assembly time.
    The existing load()/get_constraints_for_file()/get_all_constraints() interface
    is preserved for storage/local.py compatibility.
    """

    def __init__(self) -> None:
        self._global_constraints: str = ""
        self._scoped_constraints: Dict[str, str] = {}   # dir_path → raw text
        self._items: List[Constraint] = []              # structured items

    # ── Load / Save ──────────────────────────────────────────────────────

    def load(self, project_root: Path) -> None:
        """Scan project for constraint files and load them.

        Looks for:
          - {project_root}/.skeletongraph/constraints.md (global)
          - {project_root}/<any_dir>/.skeletongraph/constraints.md (scoped)
        """
        global_file = project_root / ".skeletongraph" / "constraints.md"
        if global_file.exists():
            raw = global_file.read_text(encoding="utf-8", errors="replace").strip()
            self._items = _parse_items(raw)
            self._global_constraints = raw

        for constraints_file in project_root.rglob(".skeletongraph/constraints.md"):
            if constraints_file == global_file:
                continue
            sg_dir = constraints_file.parent
            scope_dir = sg_dir.parent
            rel_scope = scope_dir.relative_to(project_root).as_posix()
            self._scoped_constraints[rel_scope] = constraints_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

    def save_global(self, project_root: Path) -> None:
        """Persist global constraints to disk."""
        sg_dir = project_root / ".skeletongraph"
        sg_dir.mkdir(parents=True, exist_ok=True)
        target = sg_dir / "constraints.md"
        # Rebuild from structured items + any free-form preamble
        blocks = [item.to_block() for item in self._items]
        target.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        self._global_constraints = target.read_text(encoding="utf-8")

    # ── Query interface (used by assembly) ───────────────────────────────

    def get_constraints_for_file(self, file_path: str) -> str:
        """Get merged constraints for a specific file (global + scoped)."""
        parts = []

        confirmed_text = self._confirmed_text()
        if confirmed_text:
            parts.append(confirmed_text)

        file_dir = str(PurePosixPath(file_path).parent)
        for scope_dir, constraints in sorted(self._scoped_constraints.items()):
            if file_dir == scope_dir or file_dir.startswith(scope_dir + "/"):
                parts.append(f"# [{scope_dir}/ scope]\n{constraints}")

        return "\n\n".join(parts)

    def get_all_constraints(self) -> str:
        """Get confirmed global constraints. Used when file scope is unknown."""
        return self._confirmed_text() or self._global_constraints

    def get_all_for_overview(self) -> str:
        """Human-readable list for sg_overview Zone 1, includes proposed."""
        if not self._items:
            return self._global_constraints
        lines = []
        for item in self._items:
            prefix = "✓" if item.confirmed else "?"
            lines.append(f"[{prefix}] ({item.provenance}) {item.text.strip()}")
        return "\n".join(lines)

    @property
    def has_constraints(self) -> bool:
        return bool(self._global_constraints) or bool(self._scoped_constraints)

    @property
    def scope_count(self) -> int:
        return len(self._scoped_constraints)

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add_constraint(
        self,
        text: str,
        provenance: str = "manual",
        confirmed: bool = True,
    ) -> Constraint:
        """Add a confirmed constraint (from init-arg or manual CLI)."""
        cid = _make_id(text)
        c = Constraint(id=cid, text=text.strip(), provenance=provenance, confirmed=confirmed)
        self._items.append(c)
        return c

    def propose_constraint(self, text: str, source: str = "model-proposed") -> Constraint:
        """Add an unconfirmed proposal (from sg_constraint action=propose)."""
        return self.add_constraint(text, provenance=source, confirmed=False)

    def confirm_constraint(
        self,
        constraint_id: str,
        project_root: Optional[Path] = None,
    ) -> bool:
        """Mark a proposed constraint as confirmed. Returns True if found.

        If project_root is given, also promotes to decisions.md.
        """
        for item in self._items:
            if item.id == constraint_id or item.id.startswith(constraint_id):
                item.confirmed = True
                if project_root is not None:
                    _promote_to_decisions(item, project_root)
                return True
        return False

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint by id. Returns True if found."""
        before = len(self._items)
        self._items = [
            c for c in self._items
            if not (c.id == constraint_id or c.id.startswith(constraint_id))
        ]
        return len(self._items) < before

    def list_constraints(self, include_proposed: bool = True) -> List[Constraint]:
        """Return all constraints, optionally filtering out proposals."""
        if include_proposed:
            return list(self._items)
        return [c for c in self._items if c.confirmed]

    # ── IDE rule aggregation ──────────────────────────────────────────────

    def aggregate_from_ide_rules(self, project_root: Path) -> int:
        """Scan IDE rule files and import their content as confirmed constraints.

        Files checked (in order):
          CLAUDE.md, AGENTS.md, .github/copilot-instructions.md,
          .windsurfrules, .roorules, .rules,
          .cursor/rules/*.mdc  (all matched)

        Returns number of new constraints added.
        """
        sources: List[tuple[str, Path]] = [
            ("CLAUDE.md", project_root / "CLAUDE.md"),
            ("AGENTS.md", project_root / "AGENTS.md"),
            ("copilot-instructions.md", project_root / ".github" / "copilot-instructions.md"),
            (".windsurfrules", project_root / ".windsurfrules"),
            (".roorules", project_root / ".roorules"),
            (".rules", project_root / ".rules"),
        ]

        # .cursor/rules/*.mdc — all files
        cursor_rules_dir = project_root / ".cursor" / "rules"
        if cursor_rules_dir.exists():
            for mdc in sorted(cursor_rules_dir.glob("*.mdc")):
                sources.append((f".cursor/rules/{mdc.name}", mdc))

        added = 0
        existing_texts = {c.text.strip() for c in self._items}

        for provenance, path in sources:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            # Deduplicate — skip if same text already stored
            if text in existing_texts:
                continue
            c = Constraint(
                id=_make_id(text),
                text=text,
                provenance=provenance,
                confirmed=True,
            )
            self._items.append(c)
            existing_texts.add(text)
            added += 1

        return added

    # ── Legacy helpers ────────────────────────────────────────────────────

    def set_global(self, text: str) -> None:
        """Set global constraints programmatically (for API/test use)."""
        self._global_constraints = text.strip()
        # Also parse items from provided text
        self._items = _parse_items(text)

    # ── Internal ──────────────────────────────────────────────────────────

    def _confirmed_text(self) -> str:
        confirmed = [c.text.strip() for c in self._items if c.confirmed]
        return "\n\n".join(confirmed)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_id(text: str) -> str:
    """Stable 8-char id from text hash."""
    return hashlib.sha1(text.strip().encode()).hexdigest()[:8]


_BLOCK_RE = re.compile(
    r"<!-- sg:constraint\s+id=(\S+)\s+confirmed=(\S+)\s+provenance=(\S+)\s*-->\n(.*?)\n<!-- /sg:constraint -->",
    re.DOTALL,
)


def _promote_to_decisions(item: Constraint, project_root: Path) -> None:
    """Append a confirmed constraint to decisions.md as a persistent decision record."""
    sg_dir = project_root / ".skeletongraph"
    sg_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = sg_dir / "decisions.md"

    date_str = time.strftime("%Y-%m-%d")
    entry = (
        f"\n## [{item.id}] {date_str}  —  {item.provenance}\n\n"
        f"{item.text.strip()}\n"
    )

    if decisions_path.exists():
        existing = decisions_path.read_text(encoding="utf-8", errors="replace")
        # Skip if already promoted
        if item.id in existing:
            return
        decisions_path.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
    else:
        header = "# Decisions\n\nPromoted constraints and architectural decisions.\n"
        decisions_path.write_text(header + entry, encoding="utf-8")


def _parse_items(raw: str) -> List[Constraint]:
    """Parse structured constraint blocks from constraints.md text."""
    items = []
    for m in _BLOCK_RE.finditer(raw):
        cid, confirmed_str, provenance, text = m.groups()
        items.append(Constraint(
            id=cid,
            text=text.strip(),
            provenance=provenance,
            confirmed=(confirmed_str.lower() == "true"),
        ))
    return items
