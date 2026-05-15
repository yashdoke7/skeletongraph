"""
IDE auto-detection.

Checks for indicator files/directories to decide which IDEs are present.
Returns a list of IDE names the installer should configure.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


# Map: ide_name → (check_fn, description)
# check_fn receives (project_root, home) and returns bool


def _has_claude(project_root: Path, home: Path) -> bool:
    return (
        (home / ".claude.json").exists()
        or (home / ".claude").is_dir()
        or (project_root / "CLAUDE.md").exists()
        or (project_root / ".claude").is_dir()
    )


def _has_cursor(project_root: Path, home: Path) -> bool:
    return (
        (project_root / ".cursorrules").exists()
        or (project_root / ".cursor").is_dir()
        or (home / ".cursor").is_dir()
        # Windows: AppData/Roaming/Cursor
        or (home / "AppData" / "Roaming" / "Cursor").exists()
    )


def _has_cline(project_root: Path, home: Path) -> bool:
    # Cline is a VS Code extension — detect via .vscode
    return (project_root / ".vscode").is_dir()


def _has_roo(project_root: Path, home: Path) -> bool:
    return (project_root / ".roorules").exists() or (project_root / ".roo").is_dir()


def _has_windsurf(project_root: Path, home: Path) -> bool:
    return (
        (project_root / ".windsurfrules").exists()
        or (home / ".codeium").is_dir()
        or (home / "AppData" / "Roaming" / "Windsurf").exists()
    )


def _has_zed(project_root: Path, home: Path) -> bool:
    return (
        (home / ".config" / "zed").is_dir()
        or (home / "AppData" / "Roaming" / "Zed").exists()
        or (home / "Library" / "Application Support" / "Zed").exists()
    )


def _has_continue(project_root: Path, home: Path) -> bool:
    return (
        (home / ".continue").is_dir()
        or (project_root / ".continue").is_dir()
    )


def _has_copilot(project_root: Path, home: Path) -> bool:
    return (
        (project_root / ".github" / "copilot-instructions.md").exists()
        or (project_root / ".vscode").is_dir()
    )


_DETECTORS = [
    ("claude-code", _has_claude, "Claude Code"),
    ("cursor",      _has_cursor, "Cursor"),
    ("cline",       _has_cline,  "Cline (VS Code extension)"),
    ("roo",         _has_roo,    "Roo"),
    ("windsurf",    _has_windsurf, "Windsurf"),
    ("zed",         _has_zed,    "Zed"),
    ("continue",    _has_continue, "Continue"),
    ("copilot",     _has_copilot, "GitHub Copilot"),
]


def detect_ides(project_root: Path) -> List[str]:
    """Return list of detected IDE names for this project + machine."""
    home = Path.home()
    found = []
    for name, check_fn, _ in _DETECTORS:
        try:
            if check_fn(project_root, home):
                found.append(name)
        except Exception:
            pass
    return found


def describe_ides(project_root: Path) -> List[tuple[str, str, bool]]:
    """Return (name, label, detected) for all known IDEs."""
    home = Path.home()
    result = []
    for name, check_fn, label in _DETECTORS:
        try:
            detected = check_fn(project_root, home)
        except Exception:
            detected = False
        result.append((name, label, detected))
    return result
