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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Force LiteLLM to use v1beta for Gemini to avoid 404s on 3.1 models
os.environ["GEMINI_API_VERSION"] = "v1beta"


# ── Agent presets for first-install defaults ──────────────────────────
AGENT_PRESETS: Dict[str, Dict[str, str]] = {
    "claude_code": {
        "slm": "claude-haiku-4-5",
        "mlm": "claude-sonnet-4-6",
        "llm": "claude-opus-4-7",
        "integration": "hooks+mcp",
    },
    "cursor": {
        "slm": "claude-haiku-4-5",
        "mlm": "claude-sonnet-4-6",
        "llm": "claude-opus-4-7",
        "integration": "mcp",
    },
    "codex": {
        "slm": "gpt-5.4-mini",
        "mlm": "gpt-5.5",
        "llm": "gpt-5.5",
        "integration": "mcp+agents_md",
    },
    "copilot": {
        "slm": "gemini/gemini-3.1-flash-lite",
        "mlm": "copilot-default",
        "llm": "copilot-default",
        "integration": "extension",
    },
    "antigravity": {
        "slm": "gemini/gemini-3.1-flash-lite",
        "mlm": "gemini/gemini-2.5-pro",
        "llm": "claude-opus-4-7",
        "integration": "mcp",
    },
}

# Default per-mode tier routing
_DEFAULT_TIER_ROUTING: Dict[str, str] = {
    "retrieval_fast": "slm",
    "debug_targeted": "mlm",
    "debug_investigate": "mlm",
    "build_guided": "mlm",
    "build_greenfield": "llm",
    "refactor": "mlm",
    "explain": "mlm",
    "architecture": "llm",
    "review": "mlm",
    "test": "mlm",
    "document": "slm",
    "migrate": "mlm",
}


@dataclass
class SGConfig:
    """SkeletonGraph runtime configuration."""

    # ── Model Tiers (v4) ───────────────────────────────────────────────
    slm_model: str = "gemini/gemini-3.1-flash-lite"     # Entity extraction, summarization, docs
    mlm_model: str = "claude-sonnet-4-6"            # Planning, code review, standard tasks
    llm_model: str = "claude-opus-4-7"              # Architecture, complex debugging

    # ── Legacy LLM (kept for backward compat) ─────────────────────────
    default_model: str = "gemini/gemini-3.1-flash-lite"
    summary_model: str = "gemini/gemini-3.1-flash-lite"
    summary_batch_size: int = 2

    # ── SLM Fallback ───────────────────────────────────────────────────
    enable_slm_fallback: bool = True        # SLM entity extraction on LOW/MISS
    slm_timeout: int = 3                    # Seconds before SLM call times out
    slm_max_fqns_in_prompt: int = 200       # Max FQN names sent to SLM for matching

    # ── Tier Routing ───────────────────────────────────────────────────
    tier_routing: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_TIER_ROUTING)
    )

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
    auto_summarize_on_build: bool = True    # Auto-summarize top 20% by PageRank
    enable_embeddings: bool = True          # Use embeddings if available

    # ── Session ────────────────────────────────────────────────────────
    session_ttl_minutes: int = 60          # Session memory TTL
    session_max_turns: int = 50            # Maximum turns to remember
    enable_session: bool = True            # Enable cross-turn deduplication
    enable_slm_turn_summary: bool = True   # SLM summarizes turns >500 chars ($0.002/session)

    # ── MCP Server ─────────────────────────────────────────────────────
    server_port: int = 3500
    mcp_tool_profile: str = "compact"  # compact | minimal | full

    # ── Cost & Display ─────────────────────────────────────────────────
    show_attention_map: bool = True        # Include attention heatmap in output
    show_cost_per_query: bool = True       # Display token cost after each query
    show_session_total: bool = True        # Running session cost total

    # ── Helpers ─────────────────────────────────────────────────────────

    def get_model_for_tier(self, tier: str) -> str:
        """Get the configured model name for a tier ('slm', 'mlm', 'llm')."""
        return {
            "slm": self.slm_model,
            "mlm": self.mlm_model,
            "llm": self.llm_model,
        }.get(tier, self.mlm_model)

    def get_tier_for_mode(self, mode: str) -> str:
        """Get the tier assigned to a query mode."""
        return self.tier_routing.get(mode, "mlm")

    def get_model_for_mode(self, mode: str) -> str:
        """Get the actual model name for a query mode."""
        tier = self.get_tier_for_mode(mode)
        return self.get_model_for_tier(tier)

    @classmethod
    def from_agent_preset(cls, agent: str) -> "SGConfig":
        """Create config pre-filled with agent-specific model defaults."""
        preset = AGENT_PRESETS.get(agent, AGENT_PRESETS["claude_code"])
        return cls(
            slm_model=preset["slm"],
            mlm_model=preset["mlm"],
            llm_model=preset["llm"],
        )


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
        "SG_MCP_PROFILE": ("mcp_tool_profile", str),
        # v4 tier overrides
        "SG_SLM_MODEL": ("slm_model", str),
        "SG_MLM_MODEL": ("mlm_model", str),
        "SG_LLM_MODEL": ("llm_model", str),
        "SG_ENABLE_SLM": ("enable_slm_fallback", lambda v: v.lower() in ("1", "true", "yes")),
        "SG_SHOW_COST": ("show_cost_per_query", lambda v: v.lower() in ("1", "true", "yes")),
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
        # v4 tiers
        "slm_model": config.slm_model,
        "mlm_model": config.mlm_model,
        "llm_model": config.llm_model,
        "enable_slm_fallback": config.enable_slm_fallback,
        "tier_routing": config.tier_routing,
        # Legacy
        "default_model": config.default_model,
        "summary_model": config.summary_model,
        # Assembly
        "model_context_limit": config.model_context_limit,
        "soft_target_ratio": config.soft_target_ratio,
        "max_graph_depth": config.max_graph_depth,
        "focused_extraction_threshold": config.focused_extraction_threshold,
        "default_detail_level": config.default_detail_level,
        "compact_body_line_limit": config.compact_body_line_limit,
        # Session
        "session_ttl_minutes": config.session_ttl_minutes,
        # Display
        "show_attention_map": config.show_attention_map,
        "show_cost_per_query": config.show_cost_per_query,
        "mcp_tool_profile": config.mcp_tool_profile,
        # Build
        "auto_summarize_on_build": config.auto_summarize_on_build,
        "enable_embeddings": config.enable_embeddings,
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
