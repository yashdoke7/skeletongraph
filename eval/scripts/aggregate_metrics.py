#!/usr/bin/env python3
"""
aggregate_metrics.py — Combine all per-task results into final benchmark report.

Usage:
    python eval/scripts/aggregate_metrics.py
    python eval/scripts/aggregate_metrics.py --eval-dir eval/
"""

import argparse
import json
from pathlib import Path
from statistics import mean, median


TIER_MAP = {
    "claude_code": "A",
    "cursor": "B",
    "codex": "B",
    "antigravity": "C",
    "copilot": "D",
}

TIER_LABELS = {
    "A": "Full (per-turn breakdown)",
    "B": "Session totals",
    "C": "Qualitative + partial",
    "D": "Correctness only",
}

MODE_EXPECTED_TOKENS = {
    "FAST": 950,
    "STANDARD": 3500,
    "DEEP": 6500,
    "PLANNING": 1980,
    "REVIEW": 1200,
}


def load_all_results(eval_dir: Path) -> list:
    results = []
    runs_dir = eval_dir / "runs"

    if not runs_dir.exists():
        return []

    for agent_dir in sorted(runs_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name

        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name

            metrics_path = task_dir / "metrics.json"
            patch_path = task_dir / "patch_comparison.json"

            if not metrics_path.exists():
                continue

            try:
                m = json.loads(metrics_path.read_text())
                p = json.loads(patch_path.read_text()) if patch_path.exists() else {}
                entry = {
                    "agent": agent,
                    "task_id": task_id,
                    "tier": TIER_MAP.get(agent, "?"),
                    **m,
                }
                # Merge patch comparison fields
                for key in ["native_patch_score", "sg_patch_score", "sg_regression",
                            "native_files_changed", "sg_files_changed"]:
                    if key in p:
                        entry[key] = p[key]
                results.append(entry)
            except Exception as e:
                print(f"  Warning: could not load {metrics_path}: {e}")

    return results


def fmt_tokens(n) -> str:
    if n is None:
        return "N/A"
    return f"{n:,.0f}"


def fmt_ratio(r) -> str:
    if r is None:
        return "N/A"
    flag = "✓" if r >= 5 else "⚠"
    return f"{r:.1f}x {flag}"


def fmt_pct(p) -> str:
    if p is None:
        return "N/A"
    return f"{p:.0%}"


def generate_report(results: list) -> str:
    lines = []

    lines.append("# SkeletonGraph v3 — Benchmark Report\n")
    lines.append(f"Total runs analyzed: **{len(results)}**\n")

    # ── Tier explanation ──
    lines.append("## Measurement Tiers\n")
    lines.append("| Tier | IDEs | What's Measured |")
    lines.append("|---|---|---|")
    for tier, label in TIER_LABELS.items():
        ides = [k for k, v in TIER_MAP.items() if v == tier]
        lines.append(f"| {tier} | {', '.join(ides)} | {label} |")

    # ── Overall by agent ──
    lines.append("\n## Token Reduction by IDE\n")
    lines.append("| IDE | Tier | Tasks | Native Tokens | SG Tokens | Reduction | Correctness |")
    lines.append("|---|---|---|---|---|---|---|")

    for agent in ["claude_code", "cursor", "codex", "antigravity", "copilot"]:
        ar = [r for r in results if r["agent"] == agent]
        if not ar:
            tier = TIER_MAP.get(agent, "?")
            lines.append(f"| {agent} | {tier} | 0 | — | — | — | — |")
            continue

        tier = TIER_MAP.get(agent, "?")
        native_toks = [r.get("native", {}).get("total_tokens") for r in ar
                       if isinstance(r.get("native"), dict) and r["native"].get("total_tokens")]
        sg_toks = [r.get("sg", {}).get("total_tokens") for r in ar
                   if isinstance(r.get("sg"), dict) and r["sg"].get("total_tokens")]
        ratios = [r.get("reduction_ratio") for r in ar if r.get("reduction_ratio")]
        correct = [r.get("sg", {}).get("test_passed") for r in ar
                   if isinstance(r.get("sg"), dict) and r["sg"].get("test_passed") is not None]

        avg_native = fmt_tokens(mean(native_toks)) if native_toks else "N/A"
        avg_sg = fmt_tokens(mean(sg_toks)) if sg_toks else "N/A"
        avg_ratio = fmt_ratio(mean(ratios)) if ratios else "N/A"
        correctness = fmt_pct(mean(correct)) if correct else "N/A"

        lines.append(f"| {agent} | {tier} | {len(ar)} | {avg_native} | {avg_sg} | {avg_ratio} | {correctness} |")

    # ── By mode ──
    lines.append("\n## Token Savings by Mode\n")
    lines.append("| Mode | Tasks | Avg SG Tokens | Expected | vs Expected | Hit Rate |")
    lines.append("|---|---|---|---|---|---|")

    for mode in ["FAST", "STANDARD", "DEEP", "PLANNING", "REVIEW"]:
        mr = [r for r in results
              if isinstance(r.get("sg"), dict) and r["sg"].get("sg_mode") == mode]
        if not mr:
            continue

        sg_toks = [r["sg"].get("tokens_delivered") for r in mr if r["sg"].get("tokens_delivered")]
        hits = [r["sg"].get("sg_hit") for r in mr if r["sg"].get("sg_hit") is not None]

        avg_sg = mean(sg_toks) if sg_toks else None
        expected = MODE_EXPECTED_TOKENS.get(mode)
        vs_expected = ""
        if avg_sg and expected:
            diff = avg_sg - expected
            vs_expected = f"+{diff:,.0f}" if diff > 0 else f"{diff:,.0f}"
        hit_rate = fmt_pct(mean(hits)) if hits else "N/A"

        lines.append(f"| {mode} | {len(mr)} | {fmt_tokens(avg_sg)} | ~{expected:,} | {vs_expected} | {hit_rate} |")

    # ── Mode routing accuracy ──
    lines.append("\n## Mode Routing Accuracy\n")
    lines.append("Did the classifier route to the expected mode?\n")
    lines.append("| Task | Expected Mode | Actual Mode | Match |")
    lines.append("|---|---|---|---|")

    from setup_workspaces import TASKS
    for r in results:
        task_cfg = TASKS.get(r["task_id"])
        if not task_cfg:
            continue
        expected_mode = task_cfg.get("expected_mode")
        actual_mode = r.get("sg", {}).get("sg_mode") if isinstance(r.get("sg"), dict) else None
        if expected_mode and actual_mode:
            match = "✓" if expected_mode == actual_mode else "✗"
            lines.append(f"| {r['task_id']} | {expected_mode} | {actual_mode} | {match} |")

    # ── Modifier accuracy ──
    modifier_tasks = [r for r in results
                      if TASKS.get(r["task_id"], {}).get("expected_modifier")]
    if modifier_tasks:
        lines.append("\n## Modifier Firing Accuracy\n")
        lines.append("| Task | Expected Modifier | Fired | Match |")
        lines.append("|---|---|---|---|")
        for r in modifier_tasks:
            expected_mod = TASKS[r["task_id"]]["expected_modifier"]
            actual_mods = r.get("sg", {}).get("sg_modifiers", []) if isinstance(r.get("sg"), dict) else []
            fired = expected_mod in actual_mods
            match = "✓" if fired else "✗"
            lines.append(f"| {r['task_id']} | {expected_mod} | {'yes' if fired else 'no'} | {match} |")

    # ── Regressions and failures ──
    failures = [r for r in results if r.get("flags") or r.get("sg_regression")]
    if failures:
        lines.append("\n## ⚠ Tasks Requiring Investigation\n")

        lines.append("### Diagnosis Checklist\n")
        lines.append("For each flagged task, check in this order:\n")
        lines.append("1. **Agent not reading context.md?** → Check `tool_calls_after` in sg_hitlog.jsonl")
        lines.append("2. **Shadow files ignored?** → Check if agent called `Read` on indexed files directly")
        lines.append("3. **Confidence routing wrong?** → Check `sg_mode` — was FAST used when STANDARD needed?")
        lines.append("4. **Context.md incomplete?** → Check if agent needed info not in delivered context")
        lines.append("5. **Mode selection wrong?** → Compare expected_mode vs actual sg_mode\n")

        for r in failures:
            flags = r.get("flags", [])
            regression = ["REGRESSION: SG failed, native passed"] if r.get("sg_regression") else []
            all_flags = " | ".join(flags + regression)
            lines.append(f"- **{r['agent']}/{r['task_id']}**: {all_flags}")

    # ── Summary ──
    successes = [r for r in results if r.get("success")]
    all_ratios = [r.get("reduction_ratio") for r in results if r.get("reduction_ratio")]
    all_correct = [r.get("sg", {}).get("test_passed") for r in results
                   if isinstance(r.get("sg"), dict) and r["sg"].get("test_passed") is not None]

    lines.append("\n## Final Summary\n")
    lines.append(f"| Metric | Result | Target | Status |")
    lines.append(f"|---|---|---|---|")

    ratio_result = median(all_ratios) if all_ratios else 0
    correct_result = mean(all_correct) if all_correct else 0
    pass_result = len(successes) / len(results) if results else 0

    lines.append(f"| Median token reduction | {fmt_ratio(ratio_result)} | >5x | {'✓' if ratio_result >= 5 else '✗'} |")
    lines.append(f"| Correctness maintained | {fmt_pct(correct_result)} | ≥ native | {'✓' if correct_result >= 0.8 else '✗'} |")
    lines.append(f"| Tasks passing all criteria | {len(successes)}/{len(results)} | >80% | {'✓' if pass_result >= 0.8 else '✗'} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="eval", type=Path)
    args = parser.parse_args()

    print(f"Loading results from {args.eval_dir}/runs/...")
    results = load_all_results(args.eval_dir)

    if not results:
        print("No results found. Run evaluations first.")
        print("For Claude Code: bash eval/scripts/run_claude_code.sh all")
        return

    print(f"Found {len(results)} completed runs")
    report = generate_report(results)

    out = args.eval_dir / "results" / "benchmark_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nReport: {out}")
    print("\n" + "="*50)
    print(report[:2000] + ("..." if len(report) > 2000 else ""))


if __name__ == "__main__":
    main()
