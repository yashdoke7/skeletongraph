"""
SkeletonGraph — Token-minimal, constraint-preserving context assembly for AI coding agents.

Quick start:
    from skeletongraph import build_index, resolve_context, assemble_context
    from pathlib import Path

    store = build_index(Path("."))
    result = resolve_context("fix validate_token", store)
    context = assemble_context(result, store, Path("."))
    print(context.text)

Features:
    - 4-zone attention-aware assembly (constraints, target, structure, prompt)
    - Elastic token budget with progressive compression
    - Session memory for cross-turn context deduplication
    - Per-directory constraint scoping
    - 10-language support via Tree-sitter
    - MCP server with 11 tools for IDE integration
    - PR blast-radius analysis with risk scoring
"""

__version__ = "0.1.0"

from .build import build_index, update_index
from .retrieval.resolver import resolve_context
from .assembly.zone_assembler import assemble_context
from .config import SGConfig, load_config

__all__ = [
    "build_index",
    "update_index",
    "resolve_context",
    "assemble_context",
    "SGConfig",
    "load_config",
    "__version__",
]
