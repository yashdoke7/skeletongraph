"""
SHA256-based dirty tracking for incremental updates.

Tracks hashes at two levels:
  - File-level: has the file changed at all?
  - Function-level: which specific functions changed within a file?

Only re-parse files that changed, only re-summarize functions that changed.
Typical edit cycle: 1 file changed → 1-3 functions re-summarized → <1 second.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def hash_text(text: str) -> str:
    """SHA256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(file_path: Path) -> str:
    """SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class DirtyTracker:
    """Tracks file and function hashes for incremental update detection.

    Usage:
        tracker = DirtyTracker.load(skeletongraph_dir)

        # Check what changed
        changed = tracker.get_changed_files(project_root, file_paths)
        # Returns: (new_files, modified_files, deleted_files)

        # After re-parsing a file
        tracker.update_file("auth/middleware.py", file_hash, function_hashes)

        # Save
        tracker.save(skeletongraph_dir)
    """

    def __init__(self) -> None:
        # file_path → file content hash
        self._file_hashes: Dict[str, str] = {}
        # fqn → function body hash
        self._function_hashes: Dict[str, str] = {}

    def get_changed_files(
        self,
        project_root: Path,
        current_files: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Compare current files against stored hashes.

        Args:
            project_root: Root directory of the project.
            current_files: List of relative file paths currently in the project.

        Returns:
            Tuple of (new_files, modified_files, deleted_files).
        """
        current_set = set(current_files)
        tracked_set = set(self._file_hashes.keys())

        new_files = sorted(current_set - tracked_set)
        deleted_files = sorted(tracked_set - current_set)
        modified_files = []

        for file_path in current_set & tracked_set:
            full_path = project_root / file_path
            if full_path.exists():
                current_hash = hash_file(full_path)
                if current_hash != self._file_hashes.get(file_path):
                    modified_files.append(file_path)

        return new_files, sorted(modified_files), deleted_files

    def get_changed_functions(
        self,
        file_path: str,
        current_functions: Dict[str, str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Compare function hashes within a file.

        Args:
            file_path: Relative file path.
            current_functions: {fqn: body_hash} for all functions in the file.

        Returns:
            Tuple of (new_fqns, modified_fqns, deleted_fqns).
        """
        current_fqns = set(current_functions.keys())

        # Get all tracked FQNs for this file
        tracked_fqns = {
            fqn for fqn in self._function_hashes
            if fqn.startswith(file_path + "::")
        }

        new_fqns = sorted(current_fqns - tracked_fqns)
        deleted_fqns = sorted(tracked_fqns - current_fqns)
        modified_fqns = []

        for fqn in current_fqns & tracked_fqns:
            if current_functions[fqn] != self._function_hashes.get(fqn):
                modified_fqns.append(fqn)

        return new_fqns, sorted(modified_fqns), deleted_fqns

    def update_file(
        self,
        file_path: str,
        file_hash: str,
        function_hashes: Dict[str, str],
    ) -> None:
        """Update tracked hashes for a file and its functions.

        Args:
            file_path: Relative file path.
            file_hash: SHA256 of the file content.
            function_hashes: {fqn: body_hash} for all functions.
        """
        self._file_hashes[file_path] = file_hash

        # Remove any old function hashes for this file
        old_fqns = [
            fqn for fqn in self._function_hashes
            if fqn.startswith(file_path + "::")
        ]
        for fqn in old_fqns:
            del self._function_hashes[fqn]

        # Add new function hashes
        self._function_hashes.update(function_hashes)

    def remove_file(self, file_path: str) -> Set[str]:
        """Remove all tracking for a deleted file.

        Returns:
            Set of FQNs that were removed (for graph cleanup).
        """
        self._file_hashes.pop(file_path, None)
        removed_fqns = {
            fqn for fqn in self._function_hashes
            if fqn.startswith(file_path + "::")
        }
        for fqn in removed_fqns:
            del self._function_hashes[fqn]
        return removed_fqns

    def is_file_tracked(self, file_path: str) -> bool:
        return file_path in self._file_hashes

    def get_file_hash(self, file_path: str) -> Optional[str]:
        return self._file_hashes.get(file_path)

    def get_function_hash(self, fqn: str) -> Optional[str]:
        return self._function_hashes.get(fqn)

    @property
    def tracked_file_count(self) -> int:
        return len(self._file_hashes)

    @property
    def tracked_function_count(self) -> int:
        return len(self._function_hashes)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Save hashes to disk."""
        data = {
            "file_hashes": self._file_hashes,
            "function_hashes": self._function_hashes,
        }
        path = directory / "hashes.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> DirtyTracker:
        """Load hashes from disk. Returns empty tracker if file doesn't exist."""
        tracker = cls()
        path = directory / "hashes.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            tracker._file_hashes = data.get("file_hashes", {})
            tracker._function_hashes = data.get("function_hashes", {})
        return tracker
