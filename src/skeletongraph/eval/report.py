"""
Markdown report generator for SkeletonGraph evaluation results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .comparison import ComparisonResult


def generate_report(result: ComparisonResult) -> str:
    """Generate a full markdown comparison report."""
    d = result.to_dict()
    comp = d["comparison"]
    tb = d["tier_b_conversation"]
    tc = d["tier_c_efficiency"]
    td = d["tier_d_quality"]

    # Calculate costs ($3.00 per 1M input tokens, assume generation is separate but we just use flat for context scaling demo)
    PRICE_PER_M = 3.00
    whole_cost = (tb['whole_codebase_tokens'] / 1_000_000) * PRICE_PER_M
    native_cost = (tb['native_tokens'] / 1_000_000) * PRICE_PER_M
    sg_cost = (tb['sg_tokens'] / 1_000_000) * PRICE_PER_M

    # Format tokens with N/A handler
    sg_l4 = f"{result.sg_layer4:,}" if result.sg_layer4 is not None else "Hidden"
    native_l4 = f"{result.native_layer4:,}" if result.native_layer4 is not None else "Hidden"
    
    whole_codebase_str = f"{tb['whole_codebase_tokens']:,}" if tb['whole_codebase_tokens'] else "N/A"

    report = f"""# SkeletonGraph Evaluation Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Task
**Project:** {comp['project']}
**Prompt:** {comp['task'][:100]}...
**SG Agent:** {comp['sg_agent']}  |  **Baseline Agent:** {comp['native_agent']}

---

## Token Efficiency matrix (The 3 Planes)
*Measures Input Context Window Load: Tool outputs + agent responses + history + schema overhead.*

| Metric | 1. Whole Codebase (Static) | 2. Native Agent (Dynamic) | 3. SkeletonGraph (Optimized) |
|:---|---:|---:|---:|
| **Total Context Tokens** | {whole_codebase_str} | {tb['native_tokens']:,} | {tb['sg_tokens']:,} |
| **Reduction vs Codebase** | -- | **{tb.get('static_to_native_reduction_ratio', 0)}x** | **{tb.get('static_to_sg_reduction_ratio', 0)}x** |
| **Reduction vs Native** | -- | -- | **{tb['native_to_sg_reduction_ratio']}x** |
| **Estimated API Cost** | ${whole_cost:.4f} | ${native_cost:.4f} | ${sg_cost:.4f} |

### Token Breakdown by Pricing Category

| Component | What it measures | Native Agent | SkeletonGraph |
|:---|:---|---:|---:|
| **Context Assembly** | All text retrieved via tools (reads, grep, SG) | {result.native_layer1:,} | {result.sg_layer1:,} |
| **History Compounding** | Cumulative cost of chatting over multiple turns | {result.native_layer3:,} | {result.sg_layer3:,} |
| **Schema Overhead** | Token cost to load MCP protocol definitions | 0 | {result.sg_layer5:,} |
| **Output Text** | The actual text the agent wrote to the user | {result.native_layer2:,} | {result.sg_layer2:,} |
| **Total Measurable** | Sum of above rows | **{tb['native_tokens']:,}** | **{tb['sg_tokens']:,}** |
| **Reasoning Tax\*** | *Hidden <thinking> steps agent takes internally* | *{native_l4}* | *{sg_l4}* |

*\*Reasoning tokens are excluded from the main summation because closed-models (like Gemini/Cursor) do not expose them, preventing apples-to-apples contextual load comparisons.*

---

## Behavioral Efficiency
*Measures agent actions and round-trip overhead*

| Metric | Native Agent | SkeletonGraph |
|:---|---:|---:|
| Total Agent Turns | {tc['native_turns']} | {tc['sg_turns']} |
| Total Tool Calls | {tc['native_tool_calls']} | {tc['sg_tool_calls']} |
| SG Context Queries | -- | {result.sg_trace.sg_tool_count} |
| File Views | {result.native_trace.view_file_count} | {result.sg_trace.view_file_count} |
| Grep Searches | {result.native_trace.grep_count} | {result.sg_trace.grep_count} |

---

## Execution Quality
*Measures accuracy against historical ground truth*

| Metric | Native Agent | SkeletonGraph |
|:---|:---|:---|
| **Task Completed** | {'Yes' if td['native_completed'] else 'No'} | {'Yes' if td['sg_completed'] else 'No'} |
| **Files Modified** | {len(result.native_trace.files_modified)} | {len(result.sg_trace.files_modified)} |
| **F1 / Precision / Recall** | N/A *(Coming Soon)* | N/A *(Coming Soon)* |

---

## Methodology
- **Whole Codebase:** Literal BPE tokenization of all readable, non-ignored files in the repository.
- **Context Assembly:** True byte-size of the exact strings returned by VS Code / IDE to the LLM (tiktoken cl100k_base).
- **History Compounding:** Cumulative sum. At each turn > 1, all prior turns' context is appended again as input.
- **Token Pricing:** Modeled at a flat $3.00 / 1M tokens.
"""
    return report


def save_report(result: ComparisonResult, output_path: Path) -> None:
    """Generate and save the markdown report."""
    report = generate_report(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
