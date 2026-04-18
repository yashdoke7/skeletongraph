"""
Edge extraction: convert raw call sites and imports into DependencyEdge entries.

This is the bridge between the AST parser (which extracts raw data) and the
DependencyGraph (which only knows about FQNs and edge types).

Resolution strategy:
  1. Exact FQN match (highest confidence)
  2. Short name match within the same file
  3. Import-resolved match (cross-file)
  4. Class method resolution (obj.method → Class.method)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from ..graph.dependency import DependencyEdge, EdgeType
from .ast_extractor import FileExtractionResult, RawCallSite, RawImport
from .skeleton import FileSkeleton
from .node_kinds import NodeKind


def extract_edges(
    result: FileExtractionResult,
    call_sites: List[RawCallSite],
    all_fqns: Set[str],
    fqn_by_short_name: Dict[str, List[str]],
    import_map: Dict[str, str],
) -> List[DependencyEdge]:
    """Convert raw extraction data into typed DependencyEdges.

    Args:
        result: The file's extraction result (imports, classes, etc.).
        call_sites: Raw call sites from the file.
        all_fqns: Set of all known FQNs in the project.
        fqn_by_short_name: Maps short names → list of FQNs.
            e.g., {"validate_token": ["auth/m.py::validate_token"]}
        import_map: Maps local alias → target FQN for this file.
            Built by import_resolver.

    Returns:
        List of DependencyEdge entries.
    """
    edges: List[DependencyEdge] = []

    # 1. Import edges (file-level)
    edges.extend(_resolve_import_edges(result))

    # 2. Inheritance / implements edges
    edges.extend(_resolve_inheritance_edges(result, all_fqns, fqn_by_short_name))

    # 3. Call edges
    edges.extend(_resolve_call_edges(
        call_sites, result.file_path, all_fqns,
        fqn_by_short_name, import_map,
    ))

    # 4. Decorator edges
    edges.extend(_resolve_decorator_edges(result, all_fqns, fqn_by_short_name))

    return edges


def _resolve_import_edges(result: FileExtractionResult) -> List[DependencyEdge]:
    """Create IMPORTS edges from import statements."""
    edges = []
    for imp in result.imports:
        # Create a file-level import edge
        # The target is the module path — will be resolved to a file FQN later
        if imp.names and imp.names != ["*"]:
            for name in imp.names:
                target = f"{imp.module}::{name}" if imp.module else name
                edges.append(DependencyEdge(
                    source_fqn=f"{result.file_path}::__file__",
                    target_fqn=target,
                    edge_type=EdgeType.IMPORTS,
                    source_line=imp.line,
                    confidence=0.8,  # Will be refined by import resolver
                ))
        elif not imp.names:
            edges.append(DependencyEdge(
                source_fqn=f"{result.file_path}::__file__",
                target_fqn=f"{imp.module}::__file__",
                edge_type=EdgeType.IMPORTS,
                source_line=imp.line,
                confidence=0.8,
            ))
    return edges


def _resolve_inheritance_edges(
    result: FileExtractionResult,
    all_fqns: Set[str],
    fqn_by_short_name: Dict[str, List[str]],
) -> List[DependencyEdge]:
    """Create INHERITS/IMPLEMENTS edges from class base classes."""
    edges = []
    for cls in result.classes:
        for base in cls.bases:
            target_fqns = _resolve_name(
                base, result.file_path, all_fqns, fqn_by_short_name,
            )
            edge_type = (
                EdgeType.IMPLEMENTS if cls.kind == NodeKind.INTERFACE
                else EdgeType.INHERITS
            )
            for target_fqn, conf in target_fqns:
                edges.append(DependencyEdge(
                    source_fqn=cls.fqn,
                    target_fqn=target_fqn,
                    edge_type=edge_type,
                    source_line=cls.line_start,
                    confidence=conf,
                ))
    return edges


def _resolve_call_edges(
    call_sites: List[RawCallSite],
    file_path: str,
    all_fqns: Set[str],
    fqn_by_short_name: Dict[str, List[str]],
    import_map: Dict[str, str],
) -> List[DependencyEdge]:
    """Convert raw call sites to CALLS/CONSTRUCTS edges."""
    edges = []

    for call in call_sites:
        callee = call.callee_name
        edge_type = EdgeType.CONSTRUCTS if call.is_constructor else EdgeType.CALLS

        # Try import map first (highest confidence for cross-file calls)
        if callee in import_map:
            edges.append(DependencyEdge(
                source_fqn=call.caller_fqn,
                target_fqn=import_map[callee],
                edge_type=edge_type,
                source_line=call.line,
                confidence=1.0,
            ))
            continue

        # Handle dotted calls: obj.method()
        if "." in callee:
            parts = callee.split(".")
            # Try: import_map[obj_name] → module, then resolve method
            obj_name = parts[0]
            method_name = parts[-1]

            if obj_name in import_map:
                # imported_module.function()
                base_fqn = import_map[obj_name]
                target = f"{base_fqn}.{method_name}"
                if target in all_fqns:
                    edges.append(DependencyEdge(
                        source_fqn=call.caller_fqn,
                        target_fqn=target,
                        edge_type=edge_type,
                        source_line=call.line,
                        confidence=0.9,
                    ))
                    continue

            # Try same-file class method: ClassName.method
            same_file_cls = f"{file_path}::{obj_name}.{method_name}"
            if same_file_cls in all_fqns:
                edges.append(DependencyEdge(
                    source_fqn=call.caller_fqn,
                    target_fqn=same_file_cls,
                    edge_type=edge_type,
                    source_line=call.line,
                    confidence=0.85,
                ))
                continue

            # self.method() — resolve within parent class
            if obj_name == "self" and "::" in call.caller_fqn:
                # caller is File::Class.method, target is File::Class.callee
                caller_parts = call.caller_fqn.split("::")
                if len(caller_parts) == 2 and "." in caller_parts[1]:
                    class_name = caller_parts[1].split(".")[0]
                    target_fqn = f"{file_path}::{class_name}.{method_name}"
                    if target_fqn in all_fqns:
                        edges.append(DependencyEdge(
                            source_fqn=call.caller_fqn,
                            target_fqn=target_fqn,
                            edge_type=EdgeType.CALLS,
                            source_line=call.line,
                            confidence=0.95,
                        ))
                        continue

        # Simple name resolution
        targets = _resolve_name(callee, file_path, all_fqns, fqn_by_short_name)
        for target_fqn, conf in targets:
            edges.append(DependencyEdge(
                source_fqn=call.caller_fqn,
                target_fqn=target_fqn,
                edge_type=edge_type,
                source_line=call.line,
                confidence=conf,
            ))

    return edges


def _resolve_decorator_edges(
    result: FileExtractionResult,
    all_fqns: Set[str],
    fqn_by_short_name: Dict[str, List[str]],
) -> List[DependencyEdge]:
    """Create DECORATES edges from decorator usage."""
    edges = []

    all_raw_fns = list(result.functions)
    for cls in result.classes:
        all_raw_fns.extend(cls.methods)

    for fn in all_raw_fns:
        for dec in fn.decorators:
            dec_name = dec.lstrip("@")
            targets = _resolve_name(
                dec_name, result.file_path, all_fqns, fqn_by_short_name,
            )
            for target_fqn, conf in targets:
                edges.append(DependencyEdge(
                    source_fqn=target_fqn,
                    target_fqn=fn.fqn,
                    edge_type=EdgeType.DECORATES,
                    source_line=fn.line_start,
                    confidence=conf,
                ))

    return edges


def _resolve_name(
    name: str,
    current_file: str,
    all_fqns: Set[str],
    fqn_by_short_name: Dict[str, List[str]],
) -> List[tuple[str, float]]:
    """Resolve a name to possible FQNs with confidence scores.

    Strategy (in priority order):
      1. Same-file exact match (confidence=1.0)
      2. Cross-file unique match (confidence=0.9)
      3. Cross-file ambiguous match (confidence=0.7, includes all candidates)
      4. No match → empty list (function not in our index)

    Returns:
        List of (fqn, confidence) tuples. Usually 1 entry; multiple if ambiguous.
    """
    # 1. Try exact FQN match
    exact = f"{current_file}::{name}"
    if exact in all_fqns:
        return [(exact, 1.0)]

    # 2. Look up by short name
    candidates = fqn_by_short_name.get(name, [])

    if not candidates:
        # Name not in our codebase (likely stdlib or external library)
        return []

    if len(candidates) == 1:
        return [(candidates[0], 0.9)]

    # Multiple candidates — prefer same-file
    same_file = [c for c in candidates if c.startswith(current_file + "::")]
    if same_file:
        return [(same_file[0], 0.95)]

    # Ambiguous — return all with lower confidence
    return [(c, 0.7) for c in candidates[:3]]  # Cap at 3 to avoid explosion


def build_short_name_index(all_fqns: Set[str]) -> Dict[str, List[str]]:
    """Build a short name → FQN lookup table.

    "auth/m.py::AuthMiddleware.validate_token" → "validate_token" → [fqn]
    """
    index: Dict[str, List[str]] = {}
    for fqn in all_fqns:
        if "::" in fqn:
            full_name = fqn.split("::")[-1]
            # Add the full name (e.g., "AuthMiddleware.validate_token")
            index.setdefault(full_name, []).append(fqn)
            # Also add the last part (e.g., "validate_token")
            if "." in full_name:
                short = full_name.split(".")[-1]
                index.setdefault(short, []).append(fqn)
    return index
