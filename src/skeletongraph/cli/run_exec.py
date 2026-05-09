"""Helpers for SG CLI execution planning and provider-backed runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


RUN_SYSTEM_PROMPT = """You are an SG CLI coding agent.

Use only the provided SkeletonGraph context unless the task is impossible.
Return a unified diff patch when code changes are needed.
If no code change is needed, return a concise explanation instead.
Do not wrap the patch in Markdown fences.
Do not invent files, tests, or APIs not present in the context.
"""


@dataclass
class RunPlan:
    """Structured plan for an SG CLI run."""

    prompt: str
    mode: str
    routed_tier: str
    selected_tier: str
    selected_model: str
    cli_provider: str
    api_key_env: List[str]
    api_key_configured: bool
    api_base: Optional[str]
    context_tokens: int
    confidence: str
    complexity_score: float
    routing_reason: str
    targets: List[str]
    packet_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "routed_tier": self.routed_tier,
            "selected_tier": self.selected_tier,
            "selected_model": self.selected_model,
            "cli_provider": self.cli_provider,
            "api_key_env": self.api_key_env,
            "api_key_configured": self.api_key_configured,
            "api_base": self.api_base,
            "context_tokens": self.context_tokens,
            "confidence": self.confidence,
            "complexity_score": self.complexity_score,
            "routing_reason": self.routing_reason,
            "targets": self.targets,
            "packet_path": self.packet_path,
        }


def build_execution_prompt(user_prompt: str, context_text: str) -> str:
    """Build the provider prompt for SG CLI execution."""
    return (
        "## User Task\n"
        f"{user_prompt}\n\n"
        "## SkeletonGraph Context Packet\n"
        f"{context_text}\n\n"
        "## Output Contract\n"
        "Return a unified diff patch if files should change. "
        "Return a concise explanation only if no edit is needed."
    )


def default_run_paths(project_root: Path) -> tuple[Path, Path]:
    """Return output and log paths for an execution attempt."""
    run_dir = project_root / ".skeletongraph" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return run_dir / f"{stamp}.patch", run_dir / "run_log.jsonl"


def write_run_log(log_path: Path, entry: Dict[str, Any]) -> None:
    """Append one JSONL run record."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
