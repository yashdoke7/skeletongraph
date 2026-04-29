"""
Benchmark runner: orchestrates evaluation across tasks and benchmarks.

This is the main entry point for running evaluations. It:
  1. Loads tasks from the specified dataset
  2. Loads real agent session traces (SG + native)
  3. Runs all requested benchmarks
  4. Aggregates scores and generates reports
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from ..datasets.base import EvalTask, EvalResult, BenchmarkSummary
from ..schema import AgentTrace
from ..scorer import (
    TokenEfficiencyScore,
    RetrievalQualityScore,
    aggregate_token_scores,
    aggregate_retrieval_scores,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark evaluation across tasks.
    
    Usage:
        runner = BenchmarkRunner(tasks, traces_dir)
        results = runner.run_all()
        runner.save_results(output_dir)
    """
    
    def __init__(
        self,
        tasks: List[EvalTask],
        traces_dir: Path,
        repos_dir: Optional[Path] = None,
    ):
        self.tasks = tasks
        self.traces_dir = traces_dir
        self.repos_dir = repos_dir
        
        # Load traces
        self.sg_traces: Dict[str, AgentTrace] = {}
        self.native_traces: Dict[str, AgentTrace] = {}
        self._load_traces()
        self.validation_warnings: List[str] = self._build_validation_warnings()
        
        # Results
        self.token_scores: List[TokenEfficiencyScore] = []
        self.sg_retrieval_scores: List[RetrievalQualityScore] = []
        self.native_retrieval_scores: List[RetrievalQualityScore] = []
    
    def _load_traces(self):
        """Load agent traces from the traces directory.
        
        Expected structure:
          traces_dir/
            {task_id}/
              sg_trace.json
              native_trace.json
        
        Or flat structure:
          traces_dir/
            {task_id}_sg.json
            {task_id}_native.json
        """
        if not self.traces_dir.exists():
            logger.warning("Traces directory does not exist: %s", self.traces_dir)
            return
        
        for task in self.tasks:
            tid = task.task_id
            
            # Try nested structure first
            nested_dir = self.traces_dir / tid
            if nested_dir.is_dir():
                sg_path = nested_dir / "sg_trace.json"
                native_path = nested_dir / "native_trace.json"
            else:
                # Try flat structure
                sg_path = self.traces_dir / f"{tid}_sg.json"
                native_path = self.traces_dir / f"{tid}_native.json"
            
            if sg_path.exists():
                try:
                    self.sg_traces[tid] = AgentTrace.from_json(
                        sg_path.read_text(encoding="utf-8")
                    )
                    logger.debug("Loaded SG trace for %s", tid)
                except Exception as e:
                    logger.warning("Failed to load SG trace for %s: %s", tid, e)
            
            if native_path.exists():
                try:
                    self.native_traces[tid] = AgentTrace.from_json(
                        native_path.read_text(encoding="utf-8")
                    )
                    logger.debug("Loaded native trace for %s", tid)
                except Exception as e:
                    logger.warning("Failed to load native trace for %s: %s", tid, e)
        
        logger.info(
            "Loaded traces: %d SG, %d native (out of %d tasks)",
            len(self.sg_traces),
            len(self.native_traces),
            len(self.tasks),
        )

    def _build_validation_warnings(self) -> List[str]:
        """Flag evidence-quality issues without blocking exploratory runs."""
        warnings: List[str] = []
        paired_ids = set(self.sg_traces) & set(self.native_traces)

        if len(self.tasks) < 10:
            warnings.append(
                f"Only {len(self.tasks)} task(s) loaded. Treat aggregates as smoke-test results."
            )
        if len(paired_ids) < len(self.tasks):
            warnings.append(
                f"{len(self.tasks) - len(paired_ids)} task(s) are missing a paired SG/native trace."
            )

        for label, traces in (("SG", self.sg_traces), ("native", self.native_traces)):
            for task_id, trace in traces.items():
                prompt = trace.task_prompt.strip().lower()
                if not prompt or prompt in {"dummy", "unknown"}:
                    warnings.append(f"{label} trace for {task_id} has a placeholder prompt.")
                if not trace.tool_calls:
                    warnings.append(f"{label} trace for {task_id} has no tool calls.")

        return warnings
    
    def run_token_efficiency(self) -> List[TokenEfficiencyScore]:
        """Run token efficiency benchmark across all tasks with available traces."""
        from .token_efficiency import run_token_efficiency_batch
        
        self.token_scores = run_token_efficiency_batch(
            self.tasks,
            self.sg_traces,
            self.native_traces,
            self.repos_dir,
        )
        return self.token_scores
    
    def run_retrieval_quality(self) -> dict:
        """Run retrieval quality benchmark across all tasks."""
        from .retrieval_quality import run_retrieval_quality_batch
        
        results = run_retrieval_quality_batch(
            self.tasks,
            self.sg_traces,
            self.native_traces,
        )
        self.sg_retrieval_scores = results.get("sg", [])
        self.native_retrieval_scores = results.get("native", [])
        return results
    
    def run_all(self) -> dict:
        """Run all benchmarks and return aggregated results."""
        logger.info("Running all benchmarks on %d tasks...", len(self.tasks))
        
        # Token efficiency
        logger.info("=== Token Efficiency ===")
        self.run_token_efficiency()
        
        # Retrieval quality
        logger.info("=== Retrieval Quality ===")
        self.run_retrieval_quality()
        
        # Aggregate
        return self.get_summary()
    
    def get_summary(self) -> dict:
        """Get aggregated summary of all benchmark results."""
        summary = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "task_count": len(self.tasks),
                "sg_trace_count": len(self.sg_traces),
                "native_trace_count": len(self.native_traces),
                "paired_trace_count": len(set(self.sg_traces) & set(self.native_traces)),
                "dataset_source": self.tasks[0].source if self.tasks else "unknown",
                "warnings": self.validation_warnings,
            },
            "token_efficiency": aggregate_token_scores(self.token_scores),
            "retrieval_quality": {
                "sg": aggregate_retrieval_scores(self.sg_retrieval_scores),
                "native": aggregate_retrieval_scores(self.native_retrieval_scores),
            },
            "per_task": [],
        }
        
        # Add per-task detail
        for task in self.tasks:
            tid = task.task_id
            task_detail = {"task_id": tid, "repo": task.repo}
            
            # Find token score
            for ts in self.token_scores:
                if ts.task_id == tid:
                    task_detail["token_efficiency"] = {
                        "native_retrieval": ts.native_retrieval_tokens,
                        "sg_retrieval": ts.sg_retrieval_tokens,
                        "reduction_ratio": round(ts.retrieval_reduction_ratio, 2),
                        "native_cost": round(ts.native_cost_usd, 4),
                        "sg_cost": round(ts.sg_cost_usd, 4),
                    }
                    break
            
            # Find retrieval quality scores
            for rs in self.sg_retrieval_scores:
                if rs.task_id == tid:
                    task_detail["sg_retrieval_quality"] = rs.to_dict()
                    break
            
            for rs in self.native_retrieval_scores:
                if rs.task_id == tid:
                    task_detail["native_retrieval_quality"] = rs.to_dict()
                    break
            
            summary["per_task"].append(task_detail)
        
        return summary
    
    def save_results(self, output_dir: Path) -> None:
        """Save all results to JSON and generate markdown report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON results
        summary = self.get_summary()
        results_path = output_dir / "benchmark_results.json"
        results_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved benchmark results: %s", results_path)
        
        # Generate markdown report
        report = generate_benchmark_report(summary)
        report_path = output_dir / "benchmark_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Saved benchmark report: %s", report_path)


def generate_benchmark_report(summary: dict) -> str:
    """Generate a publishable markdown benchmark report."""
    meta = summary.get("meta", {})
    te = summary.get("token_efficiency", {})
    rq = summary.get("retrieval_quality", {})
    
    sg_rq = rq.get("sg", {})
    native_rq = rq.get("native", {})
    
    # Helper to format aggregate value
    def fmt_agg(agg: dict, key: str = "mean", decimals: int = 2) -> str:
        if not agg:
            return "N/A"
        val = agg.get(key, 0)
        return f"{val:.{decimals}f}"
    
    def fmt_agg_ci(agg: dict) -> str:
        if not agg:
            return "N/A"
        return f"{agg.get('mean', 0):.2f} ± {agg.get('std', 0):.2f}"

    def fmt_int(value) -> str:
        return f"{value:,}" if isinstance(value, (int, float)) else str(value)
    
    report = f"""# SkeletonGraph Benchmark Report
*Generated: {meta.get('timestamp', 'unknown')}*

## Summary
- **Tasks Evaluated:** {meta.get('task_count', 0)}
- **SG Traces Available:** {meta.get('sg_trace_count', 0)}
- **Native Traces Available:** {meta.get('native_trace_count', 0)}
- **Paired Traces Evaluated:** {meta.get('paired_trace_count', 0)}
- **Dataset:** {meta.get('dataset_source', 'unknown')}
- **Token Counter:** tiktoken cl100k_base (BPE-exact)

---

## Evidence Quality
"""

    warnings = meta.get("warnings", [])
    if warnings:
        for warning in warnings:
            report += f"- WARNING: {warning}\n"
    else:
        report += "- No validation warnings detected in loaded traces.\n"

    report += f"""

## Token Efficiency

| Metric | Native Agent | SkeletonGraph | Improvement |
|:---|---:|---:|:---|
| Avg Retrieval Tokens | {fmt_agg(te.get('native_retrieval_tokens', {}), decimals=0)} | {fmt_agg(te.get('sg_retrieval_tokens', {}), decimals=0)} | **{fmt_agg(te.get('retrieval_reduction_ratio', {}), decimals=1)}x ↓** |
| Avg Total Tokens | -- | -- | **{fmt_agg(te.get('total_reduction_ratio', {}), decimals=1)}x ↓** |
| Avg Cost (USD) | ${fmt_agg(te.get('native_cost_usd', {}), decimals=4)} | ${fmt_agg(te.get('sg_cost_usd', {}), decimals=4)} | **{fmt_agg(te.get('cost_savings_pct', {}), decimals=1)}% saved** |

## Retrieval Quality (File Localization)

| Metric | Native Agent | SkeletonGraph |
|:---|---:|---:|
| **Precision** | {fmt_agg_ci(native_rq.get('precision', {}))} | {fmt_agg_ci(sg_rq.get('precision', {}))} |
| **Recall** | {fmt_agg_ci(native_rq.get('recall', {}))} | {fmt_agg_ci(sg_rq.get('recall', {}))} |
| **F1** | {fmt_agg_ci(native_rq.get('f1', {}))} | {fmt_agg_ci(sg_rq.get('f1', {}))} |
| **MRR** | {fmt_agg_ci(native_rq.get('mrr', {}))} | {fmt_agg_ci(sg_rq.get('mrr', {}))} |
| **Hit@3 Rate** | {native_rq.get('hit_at_3_rate', 'N/A')} | {sg_rq.get('hit_at_3_rate', 'N/A')} |
| **Hit@5 Rate** | {native_rq.get('hit_at_5_rate', 'N/A')} | {sg_rq.get('hit_at_5_rate', 'N/A')} |

---

## Per-Task Results
"""
    
    # Add per-task table
    per_task = summary.get("per_task", [])
    if per_task:
        report += "\n| Task | Repo | Native Tokens | SG Tokens | Reduction | SG F1 | Native F1 |\n"
        report += "|:---|:---|---:|---:|:---|---:|---:|\n"
        
        for pt in per_task:
            te_data = pt.get("token_efficiency", {})
            sg_rq_data = pt.get("sg_retrieval_quality", {})
            n_rq_data = pt.get("native_retrieval_quality", {})
            
            report += (
                f"| {pt.get('task_id', '?')} "
                f"| {pt.get('repo', '?')} "
                f"| {fmt_int(te_data.get('native_retrieval', 'N/A'))} "
                f"| {fmt_int(te_data.get('sg_retrieval', 'N/A'))} "
                f"| {te_data.get('reduction_ratio', 'N/A')}x "
                f"| {sg_rq_data.get('f1', 'N/A')} "
                f"| {n_rq_data.get('f1', 'N/A')} |\n"
            )
    
    report += """
---

## Methodology
- **Token Counter:** tiktoken cl100k_base (BPE-exact, NOT character-based estimation)
- **Baseline:** Native agent's actual retrieval from real session logs (NOT simulated file reads)
- **Ground Truth:** Human-authored PR patches from SWE-bench Verified (NOT self-referential graph edges)
- **Confidence:** All aggregate metrics include standard deviation (mean ± std)
- **Reproducibility:** All traces and task definitions are stored alongside results
"""
    
    return report
