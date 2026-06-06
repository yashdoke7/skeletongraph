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
# Models listed here are the EXACT names from each IDE's model selector.
# SLM/MLM/LLM tiers are recommendations for which model the user should
# select in the IDE for different task complexities. SG does NOT call
# these models directly — the IDE handles all model invocations.
#
# Last verified: 2026-05-08
AGENT_PRESETS: Dict[str, Dict] = {
    "cursor": {
        "slm": "GPT-5 Mini",
        "mlm": "Sonnet 4.6",
        "llm": "Opus 4.7",
        "integration": "mcp",
        "select_model_hint": (
            "In Cursor: Settings > Models > toggle ON the models you want. "
            "Select Sonnet 4.6 in Composer for standard tasks, Opus 4.7 for complex tasks."
        ),
        "models_available": [
            "Codex 5.3", "Composer 2", "Sonnet 4.6", "GPT 5.5", "Opus 4.7",
            "GPT-5.4", "Opus 4.6", "Opus 4.5", "GPT 5.2", "Gemini 3.1 Pro",
            "GPT-5.4 Mini", "GPT 5.4 Nano", "Haiku 4.5", "Codex 5.3 Spark",
            "Grok 4.3", "Sonnet 4.5", "Codex 5.2", "Codex 5.1 Max", "GPT-5.1",
            "Gemini 3 Flash", "Codex 5.1 Mini", "Sonnet 4", "GPT-5 Mini",
            "Gemini 2.5 Flash", "Kimi K2.5",
        ],
    },
    "copilot": {
        "slm": "GPT-5 mini",
        "mlm": "GPT-5.2",
        "llm": "Gemini 3.1 Pro",
        "integration": "mcp",
        "select_model_hint": (
            "In Copilot Chat: click the model dropdown and select GPT-5.2 for standard tasks. "
            "Use 'Other models' for GPT-5.2 Codex or Gemini 3.1 Pro."
        ),
        "models_available": [
            "Claude Haiku 4.5", "Gemini 2.5 Pro", "Gemini 3 Flash",
            "Gemini 3.1 Pro", "GPT-4.1", "GPT-4o", "GPT-5 mini",
            "GPT-5.2", "GPT-5.2 Codex", "GPT-5.4 mini",
            "Grok Code Fast 1", "Raptor mini",
        ],
    },
    "codex": {
        "slm": "GPT-5.4-Mini",
        "mlm": "GPT-5.5",
        "llm": "GPT-5.5",
        "integration": "mcp+agents_md",
        "select_model_hint": (
            "In Codex App: click the model dropdown to select GPT-5.5 (default). "
            "GPT-5.3-Codex is available under 'Other models'."
        ),
        "models_available": [
            "GPT-5.5", "GPT-5.4", "GPT-5.4-Mini",
            "GPT-5.3-Codex", "GPT-5.2",
        ],
    },
    "claude_code": {
        "slm": "Haiku 4.5",
        "mlm": "Sonnet 4.6",
        "llm": "Opus 4.7",
        "integration": "hooks+mcp",
        "select_model_hint": (
            "Claude Code defaults to Sonnet 4.6. "
            "Use /model opus-4.7 for architecture tasks, /model haiku-4.5 for quick lookups."
        ),
        "models_available": [
            "Haiku 4.5", "Sonnet 4.6", "Opus 4.7",
        ],
    },
    "antigravity": {
        "slm": "Gemini 3 Flash",
        "mlm": "Gemini 3.1 Pro",
        "llm": "Claude Opus 4.6",
        "integration": "mcp",
        "select_model_hint": (
            "In Antigravity: select Gemini 3.1 Pro for standard tasks, "
            "Claude Opus 4.6 (Thinking) for complex architecture work."
        ),
        "models_available": [
            "Gemini 3.1 Pro (High)", "Gemini 3.1 Pro (Low)", "Gemini 3 Flash",
            "Claude Sonnet 4.6 (Thinking)", "Claude Opus 4.6 (Thinking)",
            "GPT-OSS 120B (Medium)",
        ],
    },
}

# ── CLI provider presets ───────────────────────────────────────────────
# These are API-facing defaults for future SG CLI execution. API keys are
# never stored in config; SG only records which environment variable to read.
CLI_PROVIDER_PRESETS: Dict[str, Dict] = {
    "anthropic": {
        "api_key_env": ["ANTHROPIC_API_KEY"],
        "slm": "claude-haiku-4-5",
        "mlm": "claude-sonnet-4-6",
        "llm": "claude-opus-4-7",
        "models_available": [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-7",
        ],
    },
    "openai": {
        "api_key_env": ["OPENAI_API_KEY"],
        "slm": "gpt-5.4-mini",
        "mlm": "gpt-5.5",
        "llm": "gpt-5.5",
        "models_available": [
            "gpt-5.4-mini",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.3-codex",
            "gpt-5.2",
        ],
    },
    "google": {
        "api_key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "slm": "gemini-3-flash",
        "mlm": "gemini-3.1-pro",
        "llm": "gemini-3.1-pro",
        "models_available": [
            "gemini-3-flash",
            "gemini-3.1-pro",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    },
    "local": {
        "api_key_env": [],
        "api_base": "http://localhost:11434",
        "slm": "ollama/qwen3-coder:latest",
        "mlm": "ollama/qwen3-coder:30b",
        "llm": "ollama/qwen3-coder:30b",
        "models_available": [
            "ollama/qwen3-coder:latest",
            "ollama/qwen3-coder:30b",
            "ollama/qwen3-coder-next:latest",
            "ollama/deepseek-coder-v2:latest",
        ],
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
    # These are the DEFAULT recommended models. They are overridden by
    # the agent preset selected during `sg init --agent <name>`.
    # SG does NOT call these models directly — the IDE does.
    # These names are stored as recommendations in .skeletongraph/config.json.
    slm_model: str = "GPT-5 Mini"               # Quick lookups, navigation
    mlm_model: str = "Sonnet 4.6"               # Standard coding, debugging, review
    llm_model: str = "Opus 4.7"                 # Architecture, complex reasoning

    # ── Active agent ───────────────────────────────────────────────────
    agent: str = "cursor"                       # Active IDE agent preset name

    # ── CLI provider routing ───────────────────────────────────────────
    cli_provider: str = "anthropic"
    cli_slm_model: str = "claude-haiku-4-5"
    cli_mlm_model: str = "claude-sonnet-4-6"
    cli_llm_model: str = "claude-opus-4-7"
    cli_api_base: Optional[str] = None

    # ── Legacy LLM (kept for backward compat with sg summarize) ───────
    default_model: str = "gemini/gemini-2.5-flash"  # Used by sg summarize only
    summary_model: str = "gemini/gemini-2.5-flash"  # Used by sg summarize only
    summary_batch_size: int = 2

    # ── SLM Fallback ───────────────────────────────────────────────────
    enable_slm_fallback: bool = True        # SLM entity extraction on LOW/MISS
    slm_timeout: int = 3                    # Seconds before SLM call times out
    slm_max_fqns_in_prompt: int = 200       # Max FQN names sent to SLM for matching

    # ── Retrieval Fallbacks ─────────────────────────────────────────────
    enable_bm25_fallback: bool = True       # Use BM25 over comment/token corpus when no entity matches
    enable_keyword_fallback: bool = False   # Allow inverted-index fallback when no entity matches
    # Gated recall booster (default ON as of eval v2). When an entity match is
    # AMBIGUOUS (a common token short-name-matched >6 FQNs → likely coincidental),
    # ALSO run full-corpus BM25 and add its top-3 hits as seeds. Gated on
    # ambiguity so PRECISE matches (1-3 FQNs) are never diluted. Promoted from
    # ablation arm sg-weakfallback after confirming: precision +85% (0.18→0.34),
    # same recall (0.68), rank 1.0 vs 2.0, −37% input tokens on 7B/30-task eval.
    enable_weak_entity_fallback: bool = True
    # sg-rerank (DEFAULT product retrieval). BM25 supplies the wide RECALL pool
    # always (not just as a fallback), then the structural ranker reorders it by
    # centrality + entity bonus + lexical score, and read_symbol fetches only the
    # chosen function bodies. This is the eval-v2 WINNER (best file+function recall
    # at the lowest token cost). Set SG_BM25_PRIMARY=0 to fall back to the lean
    # entity-first `sg` path.
    bm25_primary: bool = True

    # ── Tier Routing ───────────────────────────────────────────────────
    enable_dynamic_model_routing: bool = True  # Adjust tier by complexity/confidence
    tier_routing: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_TIER_ROUTING)
    )

    # ── Context Assembly ───────────────────────────────────────────────
    use_4zone: bool = True               # Use 4-zone assembler (canonical). False → legacy layered.
    model_context_limit: int = 128_000
    soft_target_ratio: float = 0.25        # % of context limit we AIM to use
    max_graph_depth: int = 2               # BFS depth for blast-radius/deps
    max_zone2_body_tokens: int = 5000      # Max tokens for a single function body
    focused_extraction_threshold: int = 200  # Lines before focused extraction kicks in
    default_detail_level: str = "compact"  # compact | full
    compact_body_line_limit: int = 60      # Lines to include for compact bodies

    # ── Tier-0.5 Local LLM (Ollama) ───────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"          # Ollama server URL
    ollama_summary_model: str = "qwen2.5-coder:1.5b"        # Model for Tier-0.5 summaries
    enable_local_summary: bool = True       # Try Ollama Tier-0.5 if available
    ollama_timeout: int = 15                # Timeout per Ollama request (seconds)

    # ── Post-turn summary queue ────────────────────────────────────────────
    summary_queue_enabled: bool = True      # Enable async post-turn summary queue
    summary_queue_max_batch: int = 10       # Max functions to process per drain run

    # ── IDE push/pull integration ─────────────────────────────────────────
    # When SG is installed as an MCP server (the pull path), the agent calls
    # sg_search itself. If the UserPromptSubmit hook ALSO runs heuristic_query
    # and injects the results (push), the same retrieval fires twice → doubled
    # tokens. Default OFF: the hook injects only AMBIENT memory (constraints,
    # session digest, the use-SG reminder) and leaves task retrieval to
    # sg_search. Set True ONLY for hook-only installs with no MCP (then the push
    # is the sole retrieval path). See docs/RESEARCH.md §5d.
    hook_push_retrieval: bool = False

    # ── Ablation toggles (eval only) ──────────────────────────────────────
    # These default to True (full SG); the eval harness flips them off one at
    # a time for the stage-2 ablation. NEVER ship them False in production.
    enable_graph_expansion: bool = True   # Tier-2/Tier-3 graph neighbor inclusion
    graph_expansion_policy: str = "gated"  # gated | always | off
    enable_centrality_rerank: bool = True # PageRank/hub-score reranking signal
    enable_summaries: bool = True         # Include Tier-2 summaries in assembled context

    # ── Build & Indexing ───────────────────────────────────────────────
    ignore_patterns: List[str] = field(default_factory=list)  # Extra ignore patterns
    parallel_parse: bool = False            # Reserved for future concurrent parsing
    auto_build_on_query: bool = True        # Build index on first query if missing (no "run sg build" wall)
    auto_summarize: bool = False            # Summarize on build?
    auto_summarize_on_build: bool = False   # Keep build LLM-free by default (docstring + BM25 path)
    auto_summarize_on_update: bool = False  # Keep update LLM-free by default (docstring + BM25 path)
    summary_use_docstrings: bool = True     # Seed summaries from docstrings/comments
    summary_use_local_heuristics: bool = True  # Seed missing summaries without API calls
    summary_min_words: int = 6              # Minimum word count to accept a summary
    auto_rebuild_on_completion: bool = True # Rebuild index after task completion
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
        """Get the IDE-facing configured model name for a tier."""
        return {
            "slm": self.slm_model,
            "mlm": self.mlm_model,
            "llm": self.llm_model,
        }.get(tier, self.mlm_model)

    def get_cli_model_for_tier(self, tier: str) -> str:
        """Get the CLI/provider-facing configured model name for a tier."""
        return {
            "slm": self.cli_slm_model,
            "mlm": self.cli_mlm_model,
            "llm": self.cli_llm_model,
        }.get(tier, self.cli_mlm_model)

    def get_cli_key_envs(self) -> List[str]:
        """Return accepted API-key environment variable names for CLI execution."""
        preset = CLI_PROVIDER_PRESETS.get(self.cli_provider, {})
        return list(preset.get("api_key_env", []))

    def cli_api_key_configured(self) -> bool:
        """Return True if CLI execution has the required credentials."""
        key_envs = self.get_cli_key_envs()
        if not key_envs:
            return True
        return any(os.environ.get(name) for name in key_envs)

    def get_cli_api_base(self) -> Optional[str]:
        """Return provider API base for local/OpenAI-compatible execution."""
        if self.cli_api_base:
            return self.cli_api_base
        preset = CLI_PROVIDER_PRESETS.get(self.cli_provider, {})
        return preset.get("api_base")

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

    def apply_cli_provider_preset(self, provider: str) -> None:
        """Set CLI provider and default API-facing models."""
        preset = CLI_PROVIDER_PRESETS.get(provider)
        if not preset:
            raise ValueError(f"Unknown CLI provider: {provider}")
        self.cli_provider = provider
        self.cli_slm_model = preset["slm"]
        self.cli_mlm_model = preset["mlm"]
        self.cli_llm_model = preset["llm"]
        self.cli_api_base = preset.get("api_base")


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
        "SG_SUMMARY_MIN_WORDS": ("summary_min_words", int),
        "SG_SUMMARY_DOCSTRINGS": (
            "summary_use_docstrings",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_AUTO_SUMMARY_UPDATE": (
            "auto_summarize_on_update",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_LOCAL_SUMMARIES": (
            "summary_use_local_heuristics",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_AUTO_REBUILD": (
            "auto_rebuild_on_completion",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        # v4 tier overrides
        "SG_SLM_MODEL": ("slm_model", str),
        "SG_MLM_MODEL": ("mlm_model", str),
        "SG_LLM_MODEL": ("llm_model", str),
        "SG_CLI_PROVIDER": ("cli_provider", str),
        "SG_CLI_SLM_MODEL": ("cli_slm_model", str),
        "SG_CLI_MLM_MODEL": ("cli_mlm_model", str),
        "SG_CLI_LLM_MODEL": ("cli_llm_model", str),
        "SG_CLI_API_BASE": ("cli_api_base", str),
        "SG_DYNAMIC_ROUTING": (
            "enable_dynamic_model_routing",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_ENABLE_SLM": ("enable_slm_fallback", lambda v: v.lower() in ("1", "true", "yes")),
        "SG_ENABLE_KEYWORD_FALLBACK": (
            "enable_keyword_fallback",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_ENABLE_BM25_FALLBACK": (
            "enable_bm25_fallback",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_BM25_PRIMARY": (
            "bm25_primary",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_GRAPH_POLICY": ("graph_expansion_policy", str),
        "SG_SHOW_COST": ("show_cost_per_query", lambda v: v.lower() in ("1", "true", "yes")),
        # Tier-0.5 / queue
        "SG_OLLAMA_URL": ("ollama_base_url", str),
        "SG_OLLAMA_MODEL": ("ollama_summary_model", str),
        "SG_LOCAL_SUMMARY": (
            "enable_local_summary",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_OLLAMA_TIMEOUT": ("ollama_timeout", int),
        "SG_SUMMARY_QUEUE": (
            "summary_queue_enabled",
            lambda v: v.lower() in ("1", "true", "yes"),
        ),
        "SG_SUMMARY_QUEUE_BATCH": ("summary_queue_max_batch", int),
    }

    for env_key, (attr, type_fn) in _env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(config, attr, type_fn(val))
            except (ValueError, TypeError):
                pass

    # If the CLI provider is selected via environment, use its default model
    # trio unless the caller also supplied explicit per-tier CLI model env vars.
    cli_provider_env = os.environ.get("SG_CLI_PROVIDER")
    if cli_provider_env in CLI_PROVIDER_PRESETS:
        explicit_cli_models = {
            "slm": os.environ.get("SG_CLI_SLM_MODEL"),
            "mlm": os.environ.get("SG_CLI_MLM_MODEL"),
            "llm": os.environ.get("SG_CLI_LLM_MODEL"),
            "api_base": os.environ.get("SG_CLI_API_BASE"),
        }
        config.apply_cli_provider_preset(cli_provider_env)
        if explicit_cli_models["slm"]:
            config.cli_slm_model = explicit_cli_models["slm"]
        if explicit_cli_models["mlm"]:
            config.cli_mlm_model = explicit_cli_models["mlm"]
        if explicit_cli_models["llm"]:
            config.cli_llm_model = explicit_cli_models["llm"]
        if explicit_cli_models["api_base"]:
            config.cli_api_base = explicit_cli_models["api_base"]

    return config


def save_config(config: SGConfig, project_root: Path) -> None:
    """Save current config to project's .skeletongraph/config.json."""
    sg_dir = project_root / ".skeletongraph"
    sg_dir.mkdir(parents=True, exist_ok=True)

    data = {
        # Active agent
        "agent": config.agent,
        # v4 tiers
        "slm_model": config.slm_model,
        "mlm_model": config.mlm_model,
        "llm_model": config.llm_model,
        "cli_provider": config.cli_provider,
        "cli_slm_model": config.cli_slm_model,
        "cli_mlm_model": config.cli_mlm_model,
        "cli_llm_model": config.cli_llm_model,
        "cli_api_base": config.cli_api_base,
        "enable_slm_fallback": config.enable_slm_fallback,
        "enable_bm25_fallback": config.enable_bm25_fallback,
        "enable_keyword_fallback": config.enable_keyword_fallback,
        "enable_dynamic_model_routing": config.enable_dynamic_model_routing,
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
        "auto_summarize_on_update": config.auto_summarize_on_update,
        "summary_use_docstrings": config.summary_use_docstrings,
        "summary_use_local_heuristics": config.summary_use_local_heuristics,
        "summary_min_words": config.summary_min_words,
        "auto_rebuild_on_completion": config.auto_rebuild_on_completion,
        "enable_embeddings": config.enable_embeddings,
        # Ablation toggles (eval only)
        "enable_graph_expansion": config.enable_graph_expansion,
        "graph_expansion_policy": config.graph_expansion_policy,
        "enable_centrality_rerank": config.enable_centrality_rerank,
        "enable_summaries": config.enable_summaries,
        # Tier-0.5 / queue
        "ollama_base_url": config.ollama_base_url,
        "ollama_summary_model": config.ollama_summary_model,
        "enable_local_summary": config.enable_local_summary,
        "ollama_timeout": config.ollama_timeout,
        "summary_queue_enabled": config.summary_queue_enabled,
        "summary_queue_max_batch": config.summary_queue_max_batch,
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
