"""
Agent-agnostic metrics logger for SkeletonGraph evaluation.

Appends one structured JSON line per query to:
    .skeletongraph/metrics/query_log.jsonl

Each line records mode ("skeleton" or "baseline"), token counts,
reduction ratio, confidence, and zone breakdown. This file is the
single source of truth for your evaluation benchmark — works with
Antigravity, Roo Code, Copilot, Claude Code, or any MCP client.

Usage:
    from skeletongraph.metrics.metrics_logger import MetricsLogger
    logger = MetricsLogger(project_root)
    logger.log_skeleton_query(prompt, assembled)
    logger.log_baseline_estimate(prompt, total_tokens, files_read)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryMetric:
    """A single query evaluation record."""
    timestamp: str
    mode: str                           # "skeleton" or "baseline"
    prompt: str
    sg_tokens: int = 0                  # Tokens in SG assembled context
    native_tokens_estimated: int = 0    # Estimated baseline tokens
    reduction_ratio: float = 0.0
    confidence: str = ""
    entities_matched: List[str] = field(default_factory=list)
    zone_breakdown: Dict[str, int] = field(default_factory=dict)
    session_dedup_count: int = 0
    session_tokens_saved: int = 0
    files_involved: List[str] = field(default_factory=list)
    duration_ms: int = 0
    # Information Retrieval (IR) Metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    mrr: Optional[float] = None


class MetricsLogger:
    """Append-only JSONL logger for evaluation metrics."""

    def __init__(self, project_root: Path) -> None:
        self._log_dir = project_root / ".skeletongraph" / "metrics"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "query_log.jsonl"

    def log_skeleton_query(
        self,
        prompt: str,
        sg_tokens: int,
        native_tokens_estimated: int,
        reduction_ratio: float,
        confidence: str,
        entities_matched: List[str],
        zone_breakdown: Dict[str, int],
        session_dedup_count: int = 0,
        session_tokens_saved: int = 0,
        files_involved: Optional[List[str]] = None,
        duration_ms: int = 0,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        mrr: Optional[float] = None,
    ) -> None:
        """Log a SkeletonGraph query result."""
        metric = QueryMetric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="skeleton",
            prompt=prompt,
            sg_tokens=sg_tokens,
            native_tokens_estimated=native_tokens_estimated,
            reduction_ratio=reduction_ratio,
            confidence=confidence,
            entities_matched=entities_matched,
            zone_breakdown=zone_breakdown,
            session_dedup_count=session_dedup_count,
            session_tokens_saved=session_tokens_saved,
            files_involved=files_involved or [],
            duration_ms=duration_ms,
            precision=precision,
            recall=recall,
            mrr=mrr,
        )
        self._append(metric)

    def log_baseline_estimate(
        self,
        prompt: str,
        total_tokens: int,
        files_read: List[str],
        files_grepped: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """Log a simulated baseline estimation."""
        metric = QueryMetric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="baseline",
            prompt=prompt,
            native_tokens_estimated=total_tokens,
            files_involved=files_read,
            duration_ms=duration_ms,
        )
        self._append(metric)

    def log_tool_usage(
        self,
        tool_name: str,
        sg_tokens: int,
        files_involved: Optional[List[str]] = None,
        duration_ms: int = 0,
    ) -> None:
        """Log token usage for secondary MCP tools (expand_context, view_file, etc)."""
        metric = QueryMetric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="tool_usage",
            prompt=f"[{tool_name}]",
            sg_tokens=sg_tokens,
            files_involved=files_involved or [],
            duration_ms=duration_ms,
        )
        self._append(metric)

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Read the log and produce a comparison summary."""
        if not self._log_file.exists():
            return {"error": "No metrics logged yet."}

        skeleton_entries = []
        baseline_entries = []

        for line in self._log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("mode") == "skeleton":
                    skeleton_entries.append(entry)
                elif entry.get("mode") == "baseline":
                    baseline_entries.append(entry)
            except json.JSONDecodeError:
                continue

        summary: Dict[str, Any] = {
            "total_skeleton_queries": len(skeleton_entries),
            "total_baseline_queries": len(baseline_entries),
        }

        if skeleton_entries:
            avg_sg = sum(e.get("sg_tokens", 0) for e in skeleton_entries) / len(skeleton_entries)
            avg_native = sum(e.get("native_tokens_estimated", 0) for e in skeleton_entries) / len(skeleton_entries)
            avg_ratio = sum(e.get("reduction_ratio", 0) for e in skeleton_entries) / len(skeleton_entries)
            summary["skeleton"] = {
                "avg_sg_tokens": round(avg_sg),
                "avg_native_estimated": round(avg_native),
                "avg_reduction_ratio": round(avg_ratio, 1),
            }

            # Calculate IR Metrics if logged (e.g. from eval runs)
            ir_entries = [e for e in skeleton_entries if e.get("precision") is not None]
            if ir_entries:
                avg_prec = sum(e["precision"] for e in ir_entries) / len(ir_entries)
                avg_rec = sum(e["recall"] for e in ir_entries) / len(ir_entries)
                avg_mrr = sum(e["mrr"] for e in ir_entries) / len(ir_entries)
                summary["skeleton"]["ir_metrics"] = {
                    "avg_precision": round(avg_prec, 2),
                    "avg_recall": round(avg_rec, 2),
                    "avg_mrr": round(avg_mrr, 2),
                }

        if baseline_entries:
            avg_baseline = sum(e.get("native_tokens_estimated", 0) for e in baseline_entries) / len(baseline_entries)
            summary["baseline"] = {
                "avg_tokens": round(avg_baseline),
            }

        # Cross comparison
        if skeleton_entries and baseline_entries:
            sg_avg = summary["skeleton"]["avg_sg_tokens"]
            bl_avg = summary["baseline"]["avg_tokens"]
            if sg_avg > 0:
                summary["cross_comparison"] = {
                    "actual_reduction_ratio": round(bl_avg / sg_avg, 1),
                    "tokens_saved_per_query": round(bl_avg - sg_avg),
                }

        return summary

    def _append(self, metric: QueryMetric) -> None:
        """Append a metric record to the JSONL file."""
        data = asdict(metric)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
