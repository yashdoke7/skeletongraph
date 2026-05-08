"""
SkeletonGraph — Token-minimal, constraint-preserving context assembly for AI coding agents.

Quick start:
    from skeletongraph import SGEngine
    from pathlib import Path

    engine = SGEngine(Path("."))
    result = engine.query("fix validate_token")
    print(result.context_text)

Features:
    - 5-layer attention-aware SLM-orchestrated context assembly
    - Elastic token budget with progressive compression
    - Session memory for cross-turn context deduplication
    - Per-directory constraint scoping
    - 10-language support via Tree-sitter
    - MCP server with 11 tools for IDE integration
    - PR blast-radius analysis with risk scoring
"""

__version__ = "0.1.0"

from .build import build_index, update_index
from .engine import SGEngine
from .config import SGConfig, load_config

__all__ = [
    "build_index",
    "update_index",
    "SGEngine",
    "SGConfig",
    "load_config",
    "__version__",
]
