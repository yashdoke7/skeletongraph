"""
Cross-file import resolution with alias tracking.

Resolves import statements to actual FQNs in the project:
  - `from auth.middleware import validate_token` → `auth/middleware.py::validate_token`
  - `from . import utils` → relative to current package
  - `import jwt as j` → tracks alias `j` → `jwt`
  - `from utils import *` → resolves all exports

This is the #1 edge quality bottleneck — better resolution = better edges.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Set, Tuple


class ImportResolver:
    """Resolves import statements to concrete FQNs.

    Initialized with the project's file structure and all known FQNs,
    then used per-file to resolve that file's imports.
    """

    def __init__(
        self,
        all_fqns: Set[str],
        all_files: Set[str],
        short_name_index: Dict[str, List[str]],
    ) -> None:
        self._all_fqns = all_fqns
        self._all_files = all_files
        self._short_name_index = short_name_index

        # Build package map: directory → list of files
        self._package_map: Dict[str, List[str]] = {}
        for fp in all_files:
            parent = str(PurePosixPath(fp).parent)
            if parent not in self._package_map:
                self._package_map[parent] = []
            self._package_map[parent].append(fp)

        # Build module → file path map for Python-style imports
        # e.g., "auth.middleware" → "auth/middleware.py"
        self._module_to_file: Dict[str, str] = {}
        for fp in all_files:
            # Remove extension, replace / with .
            mod = fp.rsplit(".", 1)[0].replace("/", ".").replace("\\", ".")
            self._module_to_file[mod] = fp

            # Also register without __init__
            if mod.endswith(".__init__"):
                pkg = mod[:-9]  # Remove .__init__
                self._module_to_file[pkg] = fp

    def resolve_file_imports(
        self,
        file_path: str,
        imports: list,
    ) -> Dict[str, str]:
        """Resolve all imports in a file to FQN targets.

        Args:
            file_path: The importing file's relative path.
            imports: List of RawImport objects from the AST extractor.

        Returns:
            Dict mapping local alias → target FQN.
            e.g., {"validate_token": "auth/middleware.py::validate_token"}
        """
        import_map: Dict[str, str] = {}

        for imp in imports:
            resolved = self._resolve_single_import(file_path, imp)
            import_map.update(resolved)

        return import_map

    def _resolve_single_import(
        self, file_path: str, imp
    ) -> Dict[str, str]:
        """Resolve a single import statement."""
        result: Dict[str, str] = {}

        # Handle relative imports (Python: from . import X, from ..utils import Y)
        if imp.is_relative or imp.module.startswith("."):
            target_module = self._resolve_relative_module(
                file_path, imp.module
            )
        else:
            target_module = imp.module

        # Handle wildcard imports
        if "*" in imp.names:
            resolved = self._resolve_wildcard(target_module)
            result.update(resolved)
            return result

        # Handle specific name imports
        for name in imp.names:
            if not name or name == "*":
                continue

            alias = imp.aliases.get(name, name)
            fqn = self._resolve_name_to_fqn(name, target_module)
            if fqn:
                result[alias] = fqn

        # Handle bare module import: `import jwt`
        if not imp.names:
            alias = list(imp.aliases.values())[0] if imp.aliases else imp.module
            # Track that `alias` refers to module `target_module`
            # When we see alias.decode(), resolve to module::decode
            result[f"__module__{alias}"] = target_module

        return result

    def _resolve_relative_module(
        self, current_file: str, module: str
    ) -> str:
        """Resolve a relative module path to an absolute module path.

        Examples:
            current_file="auth/middleware.py", module="." → "auth"
            current_file="auth/middleware.py", module=".models" → "auth.models"
            current_file="auth/sub/handler.py", module="..models" → "auth.models"
        """
        current_dir = str(PurePosixPath(current_file).parent)
        parts = current_dir.split("/")

        # Count leading dots
        dots = 0
        rest = module
        while rest.startswith("."):
            dots += 1
            rest = rest[1:]

        # Go up (dots - 1) levels (first dot = current package)
        levels_up = max(dots - 1, 0)
        if levels_up >= len(parts):
            # Can't go above project root
            base_parts = []
        else:
            base_parts = parts[:len(parts) - levels_up]

        base = ".".join(base_parts) if base_parts else ""

        if rest:
            return f"{base}.{rest}" if base else rest
        return base

    def _resolve_name_to_fqn(
        self, name: str, module: str
    ) -> Optional[str]:
        """Resolve an imported name to a concrete FQN.

        Strategy (priority order):
          1. Direct: module_file_path::name exists in all_fqns
          2. Short name: name has exactly one match in short_name_index
          3. Module-filtered: name exists in short_name_index, filtered by module path
          4. Fallback: first candidate
        """
        # Try to find the module's file path
        module_file = self._module_to_file.get(module)

        # Strategy 1: Direct FQN match
        if module_file:
            direct_fqn = f"{module_file}::{name}"
            if direct_fqn in self._all_fqns:
                return direct_fqn

        # Strategy 2: Short name index (unique match)
        candidates = self._short_name_index.get(name, [])
        if len(candidates) == 1:
            return candidates[0]

        # Strategy 3: Filter by module path
        if candidates and module:
            # Convert module to path-like for matching
            mod_path = module.replace(".", "/")
            matched = [c for c in candidates if mod_path in c]
            if matched:
                return matched[0]

        # Strategy 4: Fallback to first candidate
        if candidates:
            return candidates[0]

        return None

    def _resolve_wildcard(self, module: str) -> Dict[str, str]:
        """Resolve `from X import *` by finding all exports of X."""
        result: Dict[str, str] = {}
        module_file = self._module_to_file.get(module)

        if not module_file:
            return result

        # Find all FQNs in this file
        prefix = f"{module_file}::"
        for fqn in self._all_fqns:
            if fqn.startswith(prefix):
                name = fqn[len(prefix):]
                # Skip private names
                if not name.startswith("_"):
                    result[name] = fqn

        return result

    def resolve_call_target(
        self,
        call_name: str,
        import_map: Dict[str, str],
        current_file_fqns: Set[str],
    ) -> Optional[str]:
        """Resolve a call expression to a target FQN.

        Handles:
          - Simple: `validate_token(...)` → check import_map, then local file
          - Dotted: `self.method(...)` → class method resolution
          - Module: `jwt.decode(...)` → module.function resolution

        Args:
            call_name: The raw call expression (e.g., "validate_token", "jwt.decode").
            import_map: The file's resolved import map.
            current_file_fqns: FQNs defined in the current file.

        Returns:
            Resolved FQN or None.
        """
        # Direct import map match
        if call_name in import_map:
            return import_map[call_name]

        # Dotted call: obj.method
        if "." in call_name:
            parts = call_name.split(".")
            obj_name = parts[0]
            method_name = parts[-1]

            # Check if obj is an imported module
            module_key = f"__module__{obj_name}"
            if module_key in import_map:
                module = import_map[module_key]
                module_file = self._module_to_file.get(module)
                if module_file:
                    fqn = f"{module_file}::{method_name}"
                    if fqn in self._all_fqns:
                        return fqn

            # Check if it's a method call on a known class
            candidates = self._short_name_index.get(method_name, [])
            if len(candidates) == 1:
                return candidates[0]

        # Local file match
        for fqn in current_file_fqns:
            if fqn.endswith(f"::{call_name}") or fqn.endswith(f".{call_name}"):
                return fqn

        # Short name fallback
        candidates = self._short_name_index.get(call_name, [])
        if len(candidates) == 1:
            return candidates[0]

        return None
