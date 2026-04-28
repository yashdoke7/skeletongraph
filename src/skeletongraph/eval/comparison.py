"""
Side-by-side comparison engine for SkeletonGraph evaluation.

Compares two AgentTrace objects (SG vs Native) and produces
structured metrics at all measurable tiers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schema import AgentTrace


@dataclass
class ComparisonResult:
    """Side-by-side comparison of SG vs Native agent traces."""

    sg_trace: AgentTrace
    native_trace: AgentTrace

    # ── Tier A: Retrieval Efficiency ───────────────────────────────
    @property
    def sg_retrieval_tokens(self) -> int:
        return self.sg_trace.total_tool_output_tokens

    @property
    def native_retrieval_tokens(self) -> int:
        return self.native_trace.total_tool_output_tokens

    @property
    def retrieval_reduction_ratio(self) -> float:
        if self.sg_retrieval_tokens == 0:
            return 0.0
        return self.native_retrieval_tokens / self.sg_retrieval_tokens

    @property
    def retrieval_tokens_saved(self) -> int:
        return max(0, self.native_retrieval_tokens - self.sg_retrieval_tokens)

    # ── Tier B: Full Conversation Cost ────────────────────────────
    @property
    def sg_conversation_tokens(self) -> int:
        return self.sg_trace.total_conversation_tokens

    @property
    def native_conversation_tokens(self) -> int:
        return self.native_trace.total_conversation_tokens

    @property
    def conversation_reduction_ratio(self) -> float:
        if self.sg_conversation_tokens == 0:
            return 0.0
        return self.native_conversation_tokens / self.sg_conversation_tokens

    # ── Per-layer breakdown for SG side ─────────────────────────
    @property
    def sg_layer1(self) -> int:
        return self.sg_trace.total_tool_output_tokens

    @property
    def sg_layer2(self) -> int:
        return self.sg_trace.total_response_tokens

    @property
    def sg_layer3(self) -> int:
        return self.sg_trace.estimated_history_tokens

    @property
    def sg_layer4(self) -> Optional[int]:
        return self.sg_trace.reasoning_tokens

    @property
    def sg_layer5(self) -> int:
        return self.sg_trace.mcp_schema_overhead_tokens

    # ── Per-layer breakdown for native side ─────────────────────
    @property
    def native_layer1(self) -> int:
        return self.native_trace.total_tool_output_tokens

    @property
    def native_layer2(self) -> int:
        return self.native_trace.total_response_tokens

    @property
    def native_layer3(self) -> int:
        return self.native_trace.estimated_history_tokens

    @property
    def native_layer4(self) -> Optional[int]:
        return self.native_trace.reasoning_tokens

    @property
    def native_layer5(self) -> int:
        # Native side has no MCP schema overhead (it doesn't use SG)
        return 0

    # ── Tier C: Turn Efficiency ───────────────────────────────────
    @property
    def sg_turns(self) -> int:
        return self.sg_trace.total_turns

    @property
    def native_turns(self) -> int:
        return self.native_trace.total_turns

    @property
    def sg_tool_calls(self) -> int:
        return self.sg_trace.tool_call_count

    @property
    def native_tool_calls(self) -> int:
        return self.native_trace.tool_call_count

    @property
    def native_repeated_views(self) -> int:
        return self.native_trace.repeated_file_views

    # ── Tier D: Quality ───────────────────────────────────────────
    @property
    def both_completed(self) -> bool:
        return bool(self.sg_trace.task_completed and self.native_trace.task_completed)

    # ── Serialization ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "comparison": {
                "project": self.sg_trace.project or self.native_trace.project,
                "sg_agent": self.sg_trace.agent,
                "native_agent": self.native_trace.agent,
                "task": self.sg_trace.task_prompt,
            },
            "tier_a_retrieval": {
                "sg_tokens": self.sg_retrieval_tokens,
                "native_tokens": self.native_retrieval_tokens,
                "reduction_ratio": round(self.retrieval_reduction_ratio, 1),
                "tokens_saved": self.retrieval_tokens_saved,
            },
            "tier_b_conversation": {
                "sg_tokens": self.sg_conversation_tokens,
                "native_tokens": self.native_conversation_tokens,
                "reduction_ratio": round(self.conversation_reduction_ratio, 1),
            },
            "tier_b_breakdown": {
                "sg": {
                    "layer1_tool_output": self.sg_layer1,
                    "layer2_responses":   self.sg_layer2,
                    "layer3_history":     self.sg_layer3,
                    "layer4_reasoning":   self.sg_layer4,
                    "layer5_mcp_schema":  self.sg_layer5,
                    "total":              self.sg_conversation_tokens,
                },
                "native": {
                    "layer1_tool_output": self.native_layer1,
                    "layer2_responses":   self.native_layer2,
                    "layer3_history":     self.native_layer3,
                    "layer4_reasoning":   self.native_layer4,
                    "layer5_mcp_schema":  0,
                    "total":              self.native_conversation_tokens,
                },
            },
            "tier_c_efficiency": {
                "sg_turns": self.sg_turns,
                "native_turns": self.native_turns,
                "sg_tool_calls": self.sg_tool_calls,
                "native_tool_calls": self.native_tool_calls,
                "native_repeated_views": self.native_repeated_views,
            },
            "tier_d_quality": {
                "both_completed": self.both_completed,
                "sg_completed": self.sg_trace.task_completed,
                "native_completed": self.native_trace.task_completed,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compare_traces(sg_trace: AgentTrace, native_trace: AgentTrace) -> ComparisonResult:
    """Compare an SG trace against a native trace."""
    return ComparisonResult(sg_trace=sg_trace, native_trace=native_trace)


def save_comparison(
    result: ComparisonResult,
    output_dir: Path,
    filename: str = "comparison.json",
) -> Path:
    """Save comparison result to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(result.to_json(), encoding="utf-8")
    return output_path
