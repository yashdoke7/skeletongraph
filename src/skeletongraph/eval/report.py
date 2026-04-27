"""
Markdown report generator for SkeletonGraph evaluation results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .comparison import ComparisonResult
from .schema import AgentTrace


def generate_report(result: ComparisonResult) -> str:
    """Generate a full markdown comparison report."""
    d = result.to_dict()
    comp = d["comparison"]
    ta = d["tier_a_retrieval"]
    tb = d["tier_b_conversation"]
    tc = d["tier_c_efficiency"]
    td = d["tier_d_quality"]

    report = f"""# SkeletonGraph Evaluation Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Task
**Project:** {comp['project']}
**Prompt:** {comp['task'][:100]}...
**SG Agent:** {comp['sg_agent']}  |  **Baseline Agent:** {comp['native_agent']}

---

## Tier A: Retrieval Efficiency
*Measures: Token cost of tool outputs only (view_file, grep, query_context)*

| Metric | SkeletonGraph | Native Agent | Δ |
|:---|---:|---:|:---|
| **Tool Output Tokens** | {ta['sg_tokens']:,} | {ta['native_tokens']:,} | **{ta['reduction_ratio']}x reduction** |
| **Tokens Saved** | — | — | **{ta['tokens_saved']:,}** |

## Tier B: Full Conversation Cost
*Measures: Tool outputs + agent responses + cumulative history re-submission*

| Metric | SkeletonGraph | Native Agent | Δ |
|:---|---:|---:|:---|
| **Total Conversation Tokens** | {tb['sg_tokens']:,} | {tb['native_tokens']:,} | **{tb['reduction_ratio']}x reduction** |

### Breakdown

| Layer | SkeletonGraph | Native Agent |
|:---|---:|---:|
| Layer 1: Tool Output | {result.sg_trace.total_tool_output_tokens:,} | {result.native_trace.total_tool_output_tokens:,} |
| Layer 2: Agent Responses | {result.sg_trace.total_response_tokens:,} | {result.native_trace.total_response_tokens:,} |
| Layer 3: History Re-submission | {result.sg_trace.estimated_history_tokens:,} | {result.native_trace.estimated_history_tokens:,} |
| Layer 4: Reasoning | {result.sg_trace.reasoning_tokens or 'N/A'} | {result.native_trace.reasoning_tokens or 'N/A'} |

## Tier C: Turn Efficiency
*Measures: Agent actions and round-trip overhead*

| Metric | SkeletonGraph | Native Agent |
|:---|---:|---:|
| Total Turns | {tc['sg_turns']} | {tc['native_turns']} |
| Total Tool Calls | {tc['sg_tool_calls']} | {tc['native_tool_calls']} |
| File Views | {result.sg_trace.view_file_count} | {result.native_trace.view_file_count} |
| Grep Searches | {result.sg_trace.grep_count} | {result.native_trace.grep_count} |
| SG Tool Calls | {result.sg_trace.sg_tool_count} | — |
| Repeated File Views | — | {tc['native_repeated_views']} |

## Tier D: Task Completion Quality

| Metric | SkeletonGraph | Native Agent |
|:---|:---|:---|
| Task Completed | {'✅' if td['sg_completed'] else '❌'} | {'✅' if td['native_completed'] else '❌'} |
| Files Modified | {len(result.sg_trace.files_modified)} | {len(result.native_trace.files_modified)} |
| Tests Passed | {'✅' if result.sg_trace.test_passed else 'N/A'} | {'✅' if result.native_trace.test_passed else 'N/A'} |

---

## Methodology
- **Layer 1 (Tool Output):** Measured by reading actual file sizes from disk (800-line cap per view), counting repeated views.
- **Layer 2 (Agent Responses):** Measured from exported chat text (`len(text) // 4`).
- **Layer 3 (History):** Computed as cumulative sum of prior turns' content (Layers 1+2) re-submitted per turn.
- **Layer 4 (Reasoning):** Available only for agents that expose API-level token breakdowns (Codex, Copilot w/ debug mode). Reported as N/A otherwise.
"""
    return report


def save_report(result: ComparisonResult, output_path: Path) -> None:
    """Generate and save the markdown report."""
    report = generate_report(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
