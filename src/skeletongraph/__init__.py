"""
SkeletonGraph — Token-minimal, constraint-preserving context assembly for AI coding agents.

Quick start:
    from skeletongraph import build_index, resolve_context, assemble_context
    from pathlib import Path

    store = build_index(Path("."))
    result = resolve_context("fix validate_token", store)
    context = assemble_context(result, store, Path("."))
    print(context.text)
"""

__version__ = "0.1.0"

from .build import build_index, update_index
from .retrieval.resolver import resolve_context
from .assembly.zone_assembler import assemble_context

__all__ = [
    "build_index",
    "update_index",
    "resolve_context",
    "assemble_context",
    "__version__",
]
