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

    # Format optional Layer 4 values
    sg_l4 = f"{result.sg_layer4:,}" if result.sg_layer4 is not None else "N/A"
    native_l4 = f"{result.native_layer4:,}" if result.native_layer4 is not None else "N/A"

    report = f"""# SkeletonGraph Evaluation Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Task
**Project:** {comp['project']}
**Prompt:** {comp['task'][:100]}...
**SG Agent:** {comp['sg_agent']}  |  **Baseline Agent:** {comp['native_agent']}

---

## Tier A: Retrieval Efficiency
*Measures: Token cost of tool outputs only (view_file, grep, query_context)*

| Metric | SkeletonGraph | Native Agent | Delta |
|:---|---:|---:|:---|
| **Tool Output Tokens** | {ta['sg_tokens']:,} | {ta['native_tokens']:,} | **{ta['reduction_ratio']}x reduction** |
| **Tokens Saved** | -- | -- | **{ta['tokens_saved']:,}** |

## Tier B: Full Conversation Cost
*Measures: Tool outputs + agent responses + history compounding + MCP schema overhead*

| Metric | SkeletonGraph | Native Agent | Delta |
|:---|---:|---:|:---|
| **Total Conversation Tokens** | {tb['sg_tokens']:,} | {tb['native_tokens']:,} | **{tb['reduction_ratio']}x reduction** |

### Per-Layer Breakdown

| Layer | What it measures | SG | Native |
|:---|:---|---:|---:|
| L1: Tool output | File reads, grep, query_context | {result.sg_layer1:,} | {result.native_layer1:,} |
| L2: Agent responses | Text the agent sent to user | {result.sg_layer2:,} | {result.native_layer2:,} |
| L3: History compounding | Prior turns re-sent each call | {result.sg_layer3:,} | {result.native_layer3:,} |
| L4: Reasoning | Internal chain-of-thought | {sg_l4} | {native_l4} |
| L5: MCP schema | Tool schema load per turn | {result.sg_layer5:,} | 0 |
| **Total** | | **{result.sg_conversation_tokens:,}** | **{result.native_conversation_tokens:,}** |

## Tier C: Turn Efficiency
*Measures: Agent actions and round-trip overhead*

| Metric | SkeletonGraph | Native Agent |
|:---|---:|---:|
| Total Turns | {tc['sg_turns']} | {tc['native_turns']} |
| Total Tool Calls | {tc['sg_tool_calls']} | {tc['native_tool_calls']} |
| File Views | {result.sg_trace.view_file_count} | {result.native_trace.view_file_count} |
| Grep Searches | {result.sg_trace.grep_count} | {result.native_trace.grep_count} |
| SG Tool Calls | {result.sg_trace.sg_tool_count} | -- |
| Repeated File Views | -- | {tc['native_repeated_views']} |

## Tier D: Task Completion Quality

| Metric | SkeletonGraph | Native Agent |
|:---|:---|:---|
| Task Completed | {'Yes' if td['sg_completed'] else 'No'} | {'Yes' if td['native_completed'] else 'No'} |
| Files Modified | {len(result.sg_trace.files_modified)} | {len(result.native_trace.files_modified)} |
| Tests Passed | {'Yes' if result.sg_trace.test_passed else 'N/A'} | {'Yes' if result.native_trace.test_passed else 'N/A'} |

---

## Methodology
- **L1 (Tool Output):** Actual file sizes from disk (tiktoken BPE, 800-line cap per view). Repeated views counted separately.
- **L2 (Agent Responses):** Measured from exported chat text (tiktoken BPE).
- **L3 (History):** Cumulative sum -- at each turn, all prior turns' (L1+L2) content re-submitted.
- **L4 (Reasoning):** Available for Claude Code (JSONL usage field) and Codex. N/A for Antigravity (Gemini internal), Cursor (not exposed).
- **L5 (MCP Schema):** SG tool schemas loaded per turn. Measured from actual JSON schema encoding. Native side = 0.
- **Token counter:** tiktoken cl100k_base (fallback: len//4 if tiktoken unavailable).
"""
    return report


def save_report(result: ComparisonResult, output_path: Path) -> None:
    """Generate and save the markdown report."""
    report = generate_report(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
