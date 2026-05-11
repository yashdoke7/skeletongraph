#!/usr/bin/env python3
"""
parse_claude_logs.py — Extract token metrics from Claude Code conversation JSON.

Usage:
    python eval/scripts/parse_claude_logs.py \
        --task-id requests-1142 \
        --run-dir eval/runs/claude_code/requests-1142 \
        --output eval/runs/claude_code/requests-1142/metrics.json
"""

import argparse
import json
import re
from pathlib import Path


def parse_claude_output(json_path: Path) -> dict:
    """Parse Claude Code --output-format json output."""
    if not json_path.exists():
        return {"error": f"File not found: {json_path}"}

    try:
        raw = json_path.read_text()
        # Claude Code may output multiple JSON objects (one per turn) or a single array
        # Handle both formats
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try parsing as newline-delimited JSON
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            data = [json.loads(l) for l in lines if l.startswith("{")]

    except Exception as e:
        return {"error": str(e)}

    if not data:
        return {"error": "No data in output"}

    # Normalize to list of messages
    if isinstance(data, dict):
        messages = data.get("messages", [data])
    elif isinstance(data, list):
        # Could be list of turn objects or flat message list
        messages = []
        for item in data:
            if isinstance(item, dict):
                if "messages" in item:
                    messages.extend(item["messages"])
                else:
                    messages.append(item)

    total_input = 0
    total_output = 0
    tool_calls = []
    turns = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role != "assistant":
            continue

        usage = msg.get("usage", {})
        input_toks = usage.get("input_tokens", 0)
        output_toks = usage.get("output_tokens", 0)
        total_input += input_toks
        total_output += output_toks

        turn_tools = []
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append({
                    "name": block.get("name", "unknown"),
                    "turn": len(turns),
                    "input_size": len(str(block.get("input", "")))
                })
                turn_tools.append(block.get("name", "unknown"))

        turns.append({
            "turn": len(turns),
            "input_tokens": input_toks,
            "output_tokens": output_toks,
            "tool_calls": turn_tools,
        })

    # File exploration tools (the ones we want to count as "exploration cost")
    exploration_tools = {"Read", "View", "Glob", "Grep", "LS", "find", "cat", "grep",
                         "view_file", "read_file", "list_directory"}
    exploration_calls = [t for t in tool_calls if t["name"] in exploration_tools]

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_turns": len(turns),
        "total_tool_calls": len(tool_calls),
        "exploration_tool_calls": len(exploration_calls),
        "tool_call_breakdown": {
            name: sum(1 for t in tool_calls if t["name"] == name)
            for name in set(t["name"] for t in tool_calls)
        },
        "turns": turns,
    }


def parse_hitlog(hitlog_path: Path) -> dict:
    """Parse SG hit log."""
    if not hitlog_path.exists():
        return {}

    entries = []
    for line in hitlog_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return {}

    last = entries[-1]
    return {
        "sg_mode": last.get("mode"),
        "sg_query_type": last.get("query_type"),
        "sg_confidence_level": last.get("confidence", {}).get("level") if isinstance(last.get("confidence"), dict) else last.get("confidence"),
        "sg_tokens_delivered": last.get("tokens_delivered"),
        "sg_layers_loaded": last.get("layers_loaded", []),
        "sg_modifiers": last.get("modifiers", []),
        "sg_tool_calls_after": last.get("tool_calls_after"),
        "sg_hit": last.get("tool_calls_after", 1) == 0,
        "all_entries": entries,
    }


def compute_reduction(native: dict, sg: dict) -> dict:
    native_total = native.get("total_tokens", 0)
    sg_total = sg.get("total_tokens", 0)

    if native_total == 0 or sg_total == 0:
        return {"reduction_ratio": None, "token_savings": None, "cost_savings_usd": None}

    # Estimate cost at Claude Sonnet 4 pricing (~$3/M input, $15/M output)
    def estimate_cost(metrics):
        input_cost = metrics.get("total_input_tokens", 0) * 3 / 1_000_000
        output_cost = metrics.get("total_output_tokens", 0) * 15 / 1_000_000
        return input_cost + output_cost

    native_cost = estimate_cost(native)
    sg_cost = estimate_cost(sg)

    return {
        "reduction_ratio": native_total / sg_total,
        "token_savings": native_total - sg_total,
        "token_savings_pct": (native_total - sg_total) / native_total * 100,
        "native_cost_usd": round(native_cost, 6),
        "sg_cost_usd": round(sg_cost, 6),
        "cost_savings_usd": round(native_cost - sg_cost, 6),
        "cost_reduction_ratio": native_cost / sg_cost if sg_cost > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir

    native_metrics = parse_claude_output(run_dir / "native_output.json")
    sg_metrics = parse_claude_output(run_dir / "sg_output.json")
    hitlog_metrics = parse_hitlog(run_dir / "sg_hitlog.jsonl")

    # Test results
    native_test = None
    sg_test = None
    if (run_dir / "native_test_result.txt").exists():
        native_test = (run_dir / "native_test_result.txt").read_text().strip() == "PASS"
    if (run_dir / "sg_test_result.txt").exists():
        sg_test = (run_dir / "sg_test_result.txt").read_text().strip() == "PASS"

    reduction = compute_reduction(native_metrics, sg_metrics)

    result = {
        "task_id": args.task_id,
        "agent": "claude_code",
        "tier": "A",
        "native": {**native_metrics, "test_passed": native_test},
        "sg": {**sg_metrics, **hitlog_metrics, "test_passed": sg_test},
        **reduction,
        "success": (
            reduction.get("reduction_ratio", 0) >= 5
            and (sg_test is None or sg_test)
            and (native_test is None or sg_test == native_test or sg_test)
            and hitlog_metrics.get("sg_hit", False)
        ),
        "flags": [],
    }

    # Flag issues
    if reduction.get("reduction_ratio", 0) < 5:
        result["flags"].append(f"LOW_REDUCTION: {reduction.get('reduction_ratio', 0):.1f}x (target: 5x)")
    if sg_test is False and native_test is not False:
        result["flags"].append("REGRESSION: SG failed test, native passed")
    if not hitlog_metrics.get("sg_hit"):
        after = hitlog_metrics.get("sg_tool_calls_after", "?")
        result["flags"].append(f"MISS: agent made {after} tool calls after SG context")
    if not hitlog_metrics:
        result["flags"].append("NO_HITLOG: SG may not have fired")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    ratio = reduction.get("reduction_ratio")
    status = "✓" if result["success"] else "✗"
    print(f"{status} {args.task_id}")
    print(f"  Tokens: {native_metrics.get('total_tokens', '?'):,} → {sg_metrics.get('total_tokens', '?'):,} ({ratio:.1f}x)" if ratio else "  Tokens: N/A")
    print(f"  Cost:   ${reduction.get('native_cost_usd', '?')} → ${reduction.get('sg_cost_usd', '?')}")
    print(f"  Mode:   {hitlog_metrics.get('sg_mode', 'N/A')} | Hit: {'✓' if hitlog_metrics.get('sg_hit') else '✗'}")
    if result["flags"]:
        for flag in result["flags"]:
            print(f"  ⚠ {flag}")


if __name__ == "__main__":
    main()
