"""
Stale-index detection: check if the index is outdated vs. current disk state.

Used by:
  - MCP server: warn agent before serving stale context
  - sg doctor: report index health
  - sg route / sg prepare: show staleness in CLI output

Lightweight check — only hashes files, doesn't re-parse anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .local import IndexStore


@dataclass
class StalenessReport:
    """Result of a staleness check."""
    stale: bool
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    age_hours: float = 0.0
    recommendation: str = ""  # "ok", "update", "rebuild"

    @property
    def total_stale(self) -> int:
        return self.new_files + self.modified_files + self.deleted_files

    def warning_text(self) -> str:
        """One-line warning string for MCP/CLI output."""
        if not self.stale:
            return ""
        parts = []
        if self.new_files:
            parts.append(f"{self.new_files} new")
        if self.modified_files:
            parts.append(f"{self.modified_files} modified")
        if self.deleted_files:
            parts.append(f"{self.deleted_files} deleted")
        files_str = ", ".join(parts)
        return (
            f"Index is stale: {files_str} files since last build "
            f"({self.age_hours:.1f}h ago). "
            f"Run `sg build --update` to refresh."
        )

    def to_dict(self) -> dict:
        return {
            "stale": self.stale,
            "stale_files": self.total_stale,
            "new_files": self.new_files,
            "modified_files": self.modified_files,
            "deleted_files": self.deleted_files,
            "age_hours": round(self.age_hours, 1),
            "recommendation": self.recommendation,
        }


def check_staleness(
    store: IndexStore,
    project_root: Path,
    current_files: Optional[list] = None,
) -> StalenessReport:
    """Check if the index is stale relative to current disk state.

    Args:
        store: The loaded index.
        project_root: Root directory of the project.
        current_files: Optional pre-computed file list (avoids re-discovery).

    Returns:
        StalenessReport with counts and recommendation.
    """
    if current_files is None:
        from ..build import discover_files
        current_files = discover_files(project_root)

    new, modified, deleted = store.dirty_tracker.get_changed_files(
        project_root, current_files,
    )

    age_hours = 0.0
    if store.meta.build_timestamp > 0:
        age_hours = (time.time() - store.meta.build_timestamp) / 3600

    total = len(new) + len(modified) + len(deleted)

    if total == 0:
        recommendation = "ok"
    elif total > 10 or len(deleted) > 3:
        recommendation = "rebuild"
    else:
        recommendation = "update"

    return StalenessReport(
        stale=total > 0,
        new_files=len(new),
        modified_files=len(modified),
        deleted_files=len(deleted),
        age_hours=age_hours,
        recommendation=recommendation,
    )
