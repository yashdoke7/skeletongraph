"""
Central configuration for SkeletonGraph.

Priority order (highest wins):
  1. Function arguments (code-level override)
  2. Environment variables (SG_*)
  3. Project-level config (.skeletongraph/config.json)
  4. Defaults defined here
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SGConfig:
    """SkeletonGraph runtime configuration."""

    # ── LLM ────────────────────────────────────────────────────────────
    default_model: str = "gemini/gemini-2.0-flash"
    summary_model: str = "gemini/gemini-2.0-flash"
    summary_batch_size: int = 20

    # ── Context Assembly ───────────────────────────────────────────────
    model_context_limit: int = 128_000
    soft_target_ratio: float = 0.25        # % of context limit we AIM to use
    max_graph_depth: int = 2               # BFS depth for blast-radius/deps
    max_zone2_body_tokens: int = 5000      # Max tokens for a single function body
    focused_extraction_threshold: int = 200  # Lines before focused extraction kicks in
    default_detail_level: str = "compact"  # compact | full
    compact_body_line_limit: int = 60      # Lines to include for compact bodies

    # ── Build & Indexing ───────────────────────────────────────────────
    ignore_patterns: List[str] = field(default_factory=list)  # Extra ignore patterns
    parallel_parse: bool = False            # Reserved for future concurrent parsing
    auto_summarize: bool = False            # Summarize on build?

    # ── Session ────────────────────────────────────────────────────────
    session_ttl_minutes: int = 60          # Session memory TTL
    session_max_turns: int = 50            # Maximum turns to remember
    enable_session: bool = True            # Enable cross-turn deduplication

    # ── MCP Server ─────────────────────────────────────────────────────
    server_port: int = 3500

    # ── Display ────────────────────────────────────────────────────────
    show_attention_map: bool = True        # Include attention heatmap in output


def load_config(project_root: Optional[Path] = None) -> SGConfig:
    """Load configuration with env var + file merging.

    Args:
        project_root: Project root to look for .skeletongraph/config.json.

    Returns:
        Merged SGConfig instance.
    """
    config = SGConfig()

    # Layer 1: Project-level config file
    if project_root:
        config_file = project_root / ".skeletongraph" / "config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                _apply_dict(config, data)
            except (json.JSONDecodeError, OSError):
                pass  # Silently ignore bad config

    # Layer 2: Environment variables (override file config)
    _env_overrides = {
        "SG_MODEL": ("default_model", str),
        "SG_SUMMARY_MODEL": ("summary_model", str),
        "SG_CONTEXT_LIMIT": ("model_context_limit", int),
        "SG_SOFT_TARGET": ("soft_target_ratio", float),
        "SG_MAX_DEPTH": ("max_graph_depth", int),
        "SG_SESSION_TTL": ("session_ttl_minutes", int),
        "SG_SERVER_PORT": ("server_port", int),
        "SG_BATCH_SIZE": ("summary_batch_size", int),
    }

    for env_key, (attr, type_fn) in _env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(config, attr, type_fn(val))
            except (ValueError, TypeError):
                pass

    return config


def save_config(config: SGConfig, project_root: Path) -> None:
    """Save current config to project's .skeletongraph/config.json."""
    sg_dir = project_root / ".skeletongraph"
    sg_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "default_model": config.default_model,
        "summary_model": config.summary_model,
        "model_context_limit": config.model_context_limit,
        "soft_target_ratio": config.soft_target_ratio,
        "max_graph_depth": config.max_graph_depth,
        "focused_extraction_threshold": config.focused_extraction_threshold,
        "default_detail_level": config.default_detail_level,
        "compact_body_line_limit": config.compact_body_line_limit,
        "session_ttl_minutes": config.session_ttl_minutes,
        "show_attention_map": config.show_attention_map,
    }
    config_file = sg_dir / "config.json"
    config_file.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _apply_dict(config: SGConfig, data: dict) -> None:
    """Apply a dictionary of values to a config object."""
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
