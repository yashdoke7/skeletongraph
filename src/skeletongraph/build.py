"""
Build orchestrator: coordinates full project indexing and incremental updates.

Entry point for `skeletongraph build` and `skeletongraph update`.
Ties together: file discovery → AST parsing → edge extraction → graph construction
→ Bloom filter → inverted index → persistence.
"""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .parser.ast_extractor import (
    extract_file,
    result_to_file_skeleton,
    FileExtractionResult,
)
from .parser.edge_extractor import build_short_name_index, extract_edges
from .parser.import_resolver import ImportResolver
from .parser.skeleton import FileSkeleton
from .graph.bloom import BloomFilter
from .graph.dependency import DependencyGraph
from .graph.inverted_index import InvertedIndex
from .storage.dirty import DirtyTracker, hash_file
from .storage.local import (
    BuildMeta,
    IndexStore,
    create_empty_index,
    load_index,
    save_index,
    VERSION,
)
from .summary.summary_store import SummaryStore
from .assembly.constraint_store import ConstraintStore
from .config import SGConfig, load_config


# Default ignore patterns (always excluded)
_DEFAULT_IGNORE = [
    "node_modules/", "dist/", "build/", "__pycache__/",
    ".git/", ".svn/", ".hg/",
    "*.pyc", "*.pyo", "*.egg-info/",
    ".venv/", "venv/", "env/",
    ".tox/", ".mypy_cache/", ".ruff_cache/",
    "*.min.js", "*.min.css", "*.map",
    ".skeletongraph/",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "*.lock",
]

# Supported file extensions
_SUPPORTED_EXTENSIONS = {
    ".py",                                      # Python
    ".js", ".mjs", ".cjs", ".jsx",              # JavaScript
    ".ts", ".tsx",                               # TypeScript
    ".java",                                     # Java
    ".go",                                       # Go
    ".rs",                                       # Rust
    ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp",  # C/C++
    ".cs",                                       # C#
    ".rb",                                       # Ruby
    ".php",                                      # PHP
}


def discover_files(
    project_root: Path,
    extra_ignore: Optional[List[str]] = None,
) -> List[str]:
    """Discover all supported source files in the project.

    Args:
        project_root: Root directory.
        extra_ignore: Additional gitignore-style patterns from .skeletongraphignore.

    Returns:
        List of relative file paths.
    """
    ignore_patterns = list(_DEFAULT_IGNORE)
    if extra_ignore:
        ignore_patterns.extend(extra_ignore)

    # Load .skeletongraphignore if present
    ignore_file = project_root / ".skeletongraphignore"
    if ignore_file.exists():
        for line in ignore_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.append(line)

    files: List[str] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue

        rel = path.relative_to(project_root).as_posix()

        # Check against ignore patterns
        if _should_ignore(rel, ignore_patterns):
            continue

        files.append(rel)

    return sorted(files)


def _should_ignore(rel_path: str, patterns: List[str]) -> bool:
    """Check if a file matches any ignore pattern."""
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            if rel_path.startswith(pattern) or f"/{pattern}" in f"/{rel_path}":
                return True
            # Also check each path segment
            clean = pattern.rstrip("/")
            parts = rel_path.split("/")
            if clean in parts[:-1]:  # Only match directory segments
                return True
        elif fnmatch.fnmatch(rel_path, pattern):
            return True
        elif fnmatch.fnmatch(rel_path.split("/")[-1], pattern):
            return True
    return False


def build_index(
    project_root: Path,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    config: Optional[SGConfig] = None,
) -> IndexStore:
    """Full project build. Creates or replaces the index.

    Args:
        project_root: Root directory of the project.
        on_progress: Optional callback(file_path, current, total) for progress.
        config: Optional configuration override.

    Returns:
        Complete IndexStore.
    """
    start_time = time.time()
    cfg = config or load_config(project_root)

    store = create_empty_index()
    files = discover_files(project_root)
    languages_seen: Set[str] = set()

    # Phase 1: Parse all files — cache results for reuse in edge extraction
    results: Dict[str, FileExtractionResult] = {}
    for i, file_path in enumerate(files):
        if on_progress:
            on_progress(file_path, i + 1, len(files))

        result = extract_file(file_path, project_root)
        if result is None:
            continue

        results[file_path] = result
        languages_seen.add(result.language)

        # Convert to FileSkeleton and register
        file_skel = result_to_file_skeleton(result)
        store.file_skeletons[file_path] = file_skel

        # Register all skeletons
        for sk in file_skel.all_skeletons:
            store.skeleton_table[sk.fqn] = sk
            store.graph.add_node(sk.fqn)

        # Update dirty tracker
        store.dirty_tracker.update_file(
            file_path,
            result.file_hash,
            {sk.fqn: sk.sha256 for sk in file_skel.all_skeletons},
        )

    # Phase 2: Extract edges using the import resolver (reuses cached results)
    all_fqns = set(store.skeleton_table.keys())
    short_name_index = build_short_name_index(all_fqns)

    # Build the import resolver with full project knowledge
    import_resolver = ImportResolver(all_fqns, set(files), short_name_index)

    for file_path, result in results.items():
        # Resolve imports using the proper import resolver
        import_map = import_resolver.resolve_file_imports(
            file_path, result.imports
        )

        # Extract call sites from the CACHED parse result
        func_ranges = [
            (sk.fqn, sk.line_start, sk.line_end)
            for sk in store.file_skeletons[file_path].all_skeletons
        ]

        source = (project_root / file_path).read_text(
            encoding="utf-8", errors="replace"
        )
        source_bytes = source.encode("utf-8")

        from .parser.ast_extractor import _get_parser, detect_language
        lang = detect_language(file_path)
        if lang:
            parser = _get_parser(lang)
            tree = parser.parse(source_bytes)

            if lang == "python":
                from .parser.languages.python import extract_call_sites
                call_sites = extract_call_sites(
                    file_path, source_bytes, tree, func_ranges,
                )
            elif lang in ("javascript", "js", "typescript", "ts", "tsx"):
                from .parser.languages.typescript import extract_call_sites_ts
                call_sites = extract_call_sites_ts(
                    file_path, source_bytes, tree, func_ranges,
                )
            else:
                call_sites = result.call_sites if hasattr(result, 'call_sites') else []

            # Convert to DependencyEdges
            edges = extract_edges(
                result, call_sites, all_fqns, short_name_index, import_map,
            )
            store.graph.add_edges(edges)

    # Phase 3: Build auxiliary structures
    # Bloom filter
    store.bloom = BloomFilter(expected_items=max(len(all_fqns), 100))
    store.bloom.add_all(all_fqns)

    # Also add short names to bloom for prompt matching
    for fqn in all_fqns:
        if "::" in fqn:
            short = fqn.split("::")[-1]
            store.bloom.add(short)
            if "." in short:
                store.bloom.add(short.split(".")[-1])

    # Inverted index
    store.inverted_index = InvertedIndex()
    for fqn, sk in store.skeleton_table.items():
        name = fqn.split("::")[-1] if "::" in fqn else fqn
        store.inverted_index.add(fqn, name, sk.signature)

    # Load constraints
    store.constraints = ConstraintStore()
    store.constraints.load(project_root)

    # Metadata
    store.meta = BuildMeta(
        version=VERSION,
        build_timestamp=time.time(),
        total_files=len(store.file_skeletons),
        total_functions=len(store.skeleton_table),
        total_edges=store.graph.edge_count,
        languages=sorted(languages_seen),
        build_duration_seconds=time.time() - start_time,
    )

    # Persist
    save_index(store, project_root)

    return store


def update_index(
    project_root: Path,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    config: Optional[SGConfig] = None,
) -> IndexStore:
    """Incremental update. Only re-processes changed files.

    Args:
        project_root: Root directory.
        on_progress: Optional progress callback.
        config: Optional configuration override.

    Returns:
        Updated IndexStore.
    """
    store = load_index(project_root)
    if store is None:
        return build_index(project_root, on_progress, config)

    start_time = time.time()
    current_files = discover_files(project_root)

    new_files, modified_files, deleted_files = store.dirty_tracker.get_changed_files(
        project_root, current_files,
    )

    changed_count = len(new_files) + len(modified_files) + len(deleted_files)
    if changed_count == 0:
        return store

    # Remove deleted files
    for file_path in deleted_files:
        removed_fqns = store.dirty_tracker.remove_file(file_path)
        for fqn in removed_fqns:
            store.graph.remove_node(fqn)
            store.skeleton_table.pop(fqn, None)
            store.inverted_index.remove(fqn)
            store.summaries.remove(fqn)
        store.file_skeletons.pop(file_path, None)

    # Process new and modified files — cache results for edge phase
    cached_results: Dict[str, FileExtractionResult] = {}
    files_to_process = new_files + modified_files

    for i, file_path in enumerate(files_to_process):
        if on_progress:
            on_progress(file_path, i + 1, len(files_to_process))

        # Remove old data for modified files
        if file_path in modified_files:
            old_fqns = store.dirty_tracker.remove_file(file_path)
            for fqn in old_fqns:
                store.graph.remove_node(fqn)
                store.skeleton_table.pop(fqn, None)
                store.inverted_index.remove(fqn)
            store.file_skeletons.pop(file_path, None)

        # Parse new/modified file
        result = extract_file(file_path, project_root)
        if result is None:
            continue

        cached_results[file_path] = result  # Cache for edge phase

        file_skel = result_to_file_skeleton(result)
        store.file_skeletons[file_path] = file_skel

        for sk in file_skel.all_skeletons:
            store.skeleton_table[sk.fqn] = sk
            store.graph.add_node(sk.fqn)

            # Update inverted index
            name = sk.fqn.split("::")[-1] if "::" in sk.fqn else sk.fqn
            store.inverted_index.add(sk.fqn, name, sk.signature)

        store.dirty_tracker.update_file(
            file_path,
            result.file_hash,
            {sk.fqn: sk.sha256 for sk in file_skel.all_skeletons},
        )

    # Rebuild edges for changed files using CACHED results (no re-parse)
    all_fqns = set(store.skeleton_table.keys())
    short_name_index = build_short_name_index(all_fqns)
    import_resolver = ImportResolver(all_fqns, set(current_files), short_name_index)

    for file_path, result in cached_results.items():
        if file_path not in store.file_skeletons:
            continue

        import_map = import_resolver.resolve_file_imports(
            file_path, result.imports
        )
        func_ranges = [
            (sk.fqn, sk.line_start, sk.line_end)
            for sk in store.file_skeletons[file_path].all_skeletons
        ]

        source = (project_root / file_path).read_text(
            encoding="utf-8", errors="replace"
        )
        source_bytes = source.encode("utf-8")
        from .parser.ast_extractor import _get_parser, detect_language
        lang = detect_language(file_path)
        if lang:
            parser = _get_parser(lang)
            tree = parser.parse(source_bytes)

            if lang == "python":
                from .parser.languages.python import extract_call_sites
                call_sites = extract_call_sites(
                    file_path, source_bytes, tree, func_ranges,
                )
            elif lang in ("javascript", "js", "typescript", "ts", "tsx"):
                from .parser.languages.typescript import extract_call_sites_ts
                call_sites = extract_call_sites_ts(
                    file_path, source_bytes, tree, func_ranges,
                )
            else:
                call_sites = result.call_sites if hasattr(result, 'call_sites') else []

            edges = extract_edges(
                result, call_sites, all_fqns, short_name_index, import_map,
            )
            store.graph.add_edges(edges)

    # Rebuild Bloom filter
    store.bloom = BloomFilter(expected_items=max(len(all_fqns), 100))
    store.bloom.add_all(all_fqns)
    for fqn in all_fqns:
        if "::" in fqn:
            short = fqn.split("::")[-1]
            store.bloom.add(short)
            if "." in short:
                store.bloom.add(short.split(".")[-1])

    # Reload constraints
    store.constraints = ConstraintStore()
    store.constraints.load(project_root)

    # Update meta
    store.meta.build_timestamp = time.time()
    store.meta.total_files = len(store.file_skeletons)
    store.meta.total_functions = len(store.skeleton_table)
    store.meta.total_edges = store.graph.edge_count
    store.meta.build_duration_seconds = time.time() - start_time

    save_index(store, project_root)
    return store
