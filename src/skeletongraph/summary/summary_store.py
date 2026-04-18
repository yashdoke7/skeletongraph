"""
Summary store: FQN → summary (separated from SkeletonCore).

Two-layer design:
  - Layer 1 (SkeletonCore): Always loaded. Used for graph traversal + ranking.
  - Layer 2 (SummaryStore): Loaded on demand. Only needed for Tier 2 assembly.

This separation means graph traversal never loads summaries → less memory,
and assembly only loads summaries for the ~10-20 Tier 2 candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class SummaryStore:
    """Maps FQN → summary string. Separate file on disk.

    Summaries are 1-line semantic descriptions:
        'Validates JWT token. Returns False if expired or malformed.'

    NOT included in SkeletonCore. Loaded only during context assembly
    for Tier 2 (expanded skeleton) entries.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def set(self, fqn: str, summary: str) -> None:
        """Set or update a summary."""
        self._store[fqn] = summary

    def get(self, fqn: str) -> Optional[str]:
        """Get summary for a single FQN. Returns None if not summarized."""
        return self._store.get(fqn)

    def batch_get(self, fqns: List[str]) -> Dict[str, str]:
        """Get summaries for multiple FQNs. Skips missing entries."""
        return {fqn: self._store[fqn] for fqn in fqns if fqn in self._store}

    def remove(self, fqn: str) -> None:
        """Remove a summary (when function is deleted)."""
        self._store.pop(fqn, None)

    def remove_by_file(self, file_path: str) -> int:
        """Remove all summaries for functions in a file.

        Returns:
            Number of summaries removed.
        """
        prefix = file_path + "::"
        to_remove = [fqn for fqn in self._store if fqn.startswith(prefix)]
        for fqn in to_remove:
            del self._store[fqn]
        return len(to_remove)

    def has(self, fqn: str) -> bool:
        return fqn in self._store

    @property
    def count(self) -> int:
        return len(self._store)

    @property
    def pending_count(self) -> int:
        """Number of entries where summary is a placeholder."""
        return sum(1 for s in self._store.values() if s.startswith("[pending"))

    def all_fqns(self) -> List[str]:
        """All FQNs that have summaries."""
        return list(self._store.keys())

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Save to disk as summaries.json."""
        path = directory / "summaries.json"
        path.write_text(
            json.dumps(self._store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> SummaryStore:
        """Load from disk. Returns empty store if file doesn't exist."""
        store = cls()
        path = directory / "summaries.json"
        if path.exists():
            store._store = json.loads(path.read_text(encoding="utf-8"))
        return store
