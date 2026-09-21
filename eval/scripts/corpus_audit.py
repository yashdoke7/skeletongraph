"""Independent, pairing-aware audit of the evaluation result corpus.

This script intentionally reads raw per-run records rather than summary.json.
It emits machine-readable tables used for the independent findings audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(values) if values else None


def median(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(values) if values else None


def paired_mean_ci(values):
    """Normal-approximation 95% CI for a paired mean difference."""
    if not values:
        return [None, None]
    n = len(values)
    center = statistics.fmean(values)
    if n == 1:
        return [center, center]
    half = 1.96 * statistics.stdev(values) / math.sqrt(n)
    return [center - half, center + half]


def exact_mcnemar_p(b, c):
    """Two-sided exact binomial test for discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def odds_ratio(a, b, c, d):
    # Haldane-Anscombe correction makes empty cells reportable.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def load_records(root: Path):
    records, bad = [], []
    for path in sorted(root.glob("**/*.json")):
        if path.name == "summary.json" or any(p.startswith("_quarantine") for p in path.parts):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # retain corrupt-input evidence
            bad.append({"path": str(path), "error": repr(exc)})
            continue
        if not isinstance(row, dict) or "task_id" not in row or "arm" not in row:
            continue
        row = dict(row)
        rel = path.relative_to(root)
        row["_path"] = str(rel)
        row["_tag"] = rel.parts[0]
        row["_archived"] = "_archive" in rel.parts
        row["_graded"] = isinstance(row.get("resolved"), bool)
        records.append(row)
    return records, bad


def completed(row):
    return row.get("stopped") in {"submit", "max_turns"} and not row.get("error")


def summarize_group(rows):
    graded = [r for r in rows if r["_graded"]]
    complete = [r for r in rows if completed(r)]
    return {
        "n": len(rows),
        "tasks": len({r.get("task_id") for r in rows}),
        "graded_n": len(graded),
        "resolved": sum(r.get("resolved") is True for r in graded),
        "pass_rate": mean([int(r["resolved"]) for r in graded]),
        "complete_n": len(complete),
        "error_n": sum(bool(r.get("error")) for r in rows),
        "stopped": dict(Counter(str(r.get("stopped")) for r in rows)),
        "retrieval_hit": mean([int(bool(r.get("retrieval_hit"))) for r in complete]),
        "retrieval_precision": mean([r.get("retrieval_precision") for r in complete]),
        "edited_gold": mean([int(bool(r.get("edited_gold_file"))) for r in complete]),
        "turns_mean": mean([r.get("n_turns") for r in complete]),
        "turns_median": median([r.get("n_turns") for r in complete]),
        "billed_input_mean": mean([r.get("billed_input") for r in complete]),
        "total_input_mean": mean([r.get("total_input_tokens") for r in complete]),
        "output_mean": mean([r.get("billed_output") for r in complete]),
        "cost_mean": mean([r.get("imputed_cost") for r in complete]),
        "wall_mean": mean([r.get("wall_s") for r in complete]),
        "search_calls_mean": mean([r.get("n_search_calls") for r in complete]),
        "tool_calls_mean": mean([r.get("n_tool_calls") for r in complete]),
        "patch_lines_mean": mean([(r.get("patch_lines_added") or 0) + (r.get("patch_lines_removed") or 0) for r in complete]),
        "models": dict(Counter(str(r.get("model_full") or r.get("model")) for r in rows)),
        "harnesses": dict(Counter(str(r.get("harness")) for r in rows)),
        "harness_versions": dict(Counter(str(r.get("harness_version")) for r in rows)),
    }


def unique_by_task(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.get("task_id"), r.get("repeat", 0))].append(r)
    # Duplicates are surfaced separately; deterministic final-path choice only
    # prevents them from multiplying observations in pair tables.
    return {k: sorted(v, key=lambda r: r["_path"])[-1] for k, v in grouped.items()}


def paired_comparison(left_rows, right_rows, left, right, tag):
    lm, rm = unique_by_task(left_rows), unique_by_task(right_rows)
    keys = sorted(set(lm) & set(rm))
    graded = [k for k in keys if lm[k]["_graded"] and rm[k]["_graded"]]
    deltas = [int(lm[k]["resolved"]) - int(rm[k]["resolved"]) for k in graded]
    left_only = sum(lm[k]["resolved"] and not rm[k]["resolved"] for k in graded)
    right_only = sum(rm[k]["resolved"] and not lm[k]["resolved"] for k in graded)
    out = {
        "tag": tag,
        "left": left,
        "right": right,
        "common_n": len(keys),
        "graded_pairs": len(graded),
        "left_pass": mean([int(lm[k]["resolved"]) for k in graded]),
        "right_pass": mean([int(rm[k]["resolved"]) for k in graded]),
        "pass_diff": mean(deltas),
        "pass_diff_ci": paired_mean_ci(deltas) if deltas else [None, None],
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_p": exact_mcnemar_p(left_only, right_only),
    }
    metrics = {
        "retrieval_hit": lambda r: int(bool(r.get("retrieval_hit"))),
        "retrieval_precision": lambda r: r.get("retrieval_precision"),
        "edited_gold": lambda r: int(bool(r.get("edited_gold_file"))),
        "turns": lambda r: r.get("n_turns"),
        "billed_input": lambda r: r.get("billed_input"),
        "total_input": lambda r: r.get("total_input_tokens"),
        "output": lambda r: r.get("billed_output"),
        "cost": lambda r: r.get("imputed_cost"),
        "wall": lambda r: r.get("wall_s"),
        "search_calls": lambda r: r.get("n_search_calls"),
        "tool_calls": lambda r: r.get("n_tool_calls"),
    }
    for name, fn in metrics.items():
        vals = [(fn(lm[k]), fn(rm[k])) for k in keys]
        vals = [(a, b) for a, b in vals if isinstance(a, (int, float)) and isinstance(b, (int, float))]
        diffs = [a - b for a, b in vals]
        out[name] = {
            "n": len(vals),
            "left": mean([a for a, _ in vals]),
            "right": mean([b for _, b in vals]),
            "diff": mean(diffs),
            "diff_ci": paired_mean_ci(diffs) if vals else [None, None],
        }
    return out


def retrieval_outcome(rows):
    graded = [r for r in rows if r["_graded"] and completed(r)]
    hit_pass = sum(bool(r.get("retrieval_hit")) and r["resolved"] for r in graded)
    hit_fail = sum(bool(r.get("retrieval_hit")) and not r["resolved"] for r in graded)
    miss_pass = sum(not bool(r.get("retrieval_hit")) and r["resolved"] for r in graded)
    miss_fail = sum(not bool(r.get("retrieval_hit")) and not r["resolved"] for r in graded)
    return {
        "n": len(graded), "hit_pass": hit_pass, "hit_fail": hit_fail,
        "miss_pass": miss_pass, "miss_fail": miss_fail,
        "pass_given_hit": hit_pass / (hit_pass + hit_fail) if hit_pass + hit_fail else None,
        "pass_given_miss": miss_pass / (miss_pass + miss_fail) if miss_pass + miss_fail else None,
        "odds_ratio": odds_ratio(hit_pass, hit_fail, miss_pass, miss_fail),
    }


def invariant_issues(records):
    by_task = defaultdict(list)
    for r in records:
        if not r["_archived"]:
            by_task[r.get("task_id")].append(r)
    issues = []
    for task, rows in by_task.items():
        for field in ("base_commit", "gold_files", "repo"):
            vals = {json.dumps(r.get(field), sort_keys=True) for r in rows if r.get(field) is not None}
            if len(vals) > 1:
                issues.append({"task": task, "field": field, "values": sorted(vals), "n_rows": len(rows)})
    return issues


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def cross_panel_stability(by_tag_arm):
    specs = {
        "react_v2": ("nemotron_v2", "none", "sg-rerank"),
        "react_v4": ("nemotron_v4", "none", "fusion"),
        "claude_v7": ("claude_v7", "native", "sg-fusion"),
        "claude_rep2": ("claude_v7_rep2", "native", "sg-fusion"),
        "codex": ("codex_v1", "codex-native", "codex-sg-fusion"),
    }
    effects = {}
    for name, (tag, base, treatment) in specs.items():
        bm = unique_by_task(by_tag_arm.get((tag, base), []))
        tm = unique_by_task(by_tag_arm.get((tag, treatment), []))
        effects[name] = {
            k[0]: int(tm[k]["resolved"]) - int(bm[k]["resolved"])
            for k in set(bm) & set(tm)
            if bm[k]["_graded"] and tm[k]["_graded"]
        }
    comparisons = []
    for left, right in combinations(effects, 2):
        keys = sorted(set(effects[left]) & set(effects[right]))
        lv, rv = [effects[left][k] for k in keys], [effects[right][k] for k in keys]
        left_wins = {k for k in keys if effects[left][k] > 0}
        right_wins = {k for k in keys if effects[right][k] > 0}
        union = left_wins | right_wins
        comparisons.append({
            "left": left, "right": right, "n": len(keys),
            "effect_correlation": pearson(lv, rv),
            "both_benefit": sum(a > 0 and b > 0 for a, b in zip(lv, rv)),
            "left_benefit_right_harm": sum(a > 0 and b < 0 for a, b in zip(lv, rv)),
            "left_harm_right_benefit": sum(a < 0 and b > 0 for a, b in zip(lv, rv)),
            "both_harm": sum(a < 0 and b < 0 for a, b in zip(lv, rv)),
            "benefit_jaccard": len(left_wins & right_wins) / len(union) if union else None,
        })
    return {"effects": effects, "comparisons": comparisons}


def standalone_retrieval(results_root: Path):
    datasets = []
    for path in sorted(results_root.glob("paper_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("per_task"), list):
            continue
        datasets.append({
            "file": path.name,
            "backend": data.get("backend"),
            "dataset": data.get("dataset"),
            "granularity": data.get("granularity"),
            "n_tasks": data.get("n_tasks"),
            "n_skipped": data.get("n_skipped"),
            "aggregate": data.get("aggregate"),
            "tasks": {r.get("task_id"): r for r in data["per_task"]},
        })
    paired = []
    groups = defaultdict(list)
    for d in datasets:
        key = (d["dataset"], d["granularity"])
        groups[key].append(d)
    for (dataset, granularity), group in groups.items():
        for a, b in combinations(group, 2):
            keys = sorted(set(a["tasks"]) & set(b["tasks"]))
            metrics = sorted(set(a["aggregate"] or {}) & set(b["aggregate"] or {}))
            deltas = {}
            for metric in metrics:
                vals = []
                for k in keys:
                    av, bv = a["tasks"][k].get(metric), b["tasks"][k].get(metric)
                    if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                        vals.append(av - bv)
                if vals:
                    deltas[metric] = {"n": len(vals), "diff": mean(vals), "ci": paired_mean_ci(vals)}
            paired.append({"dataset": dataset, "granularity": granularity, "left": a["file"], "right": b["file"], "common_n": len(keys), "metrics": deltas})
    return datasets, paired


def transcript_inventory(agent_root: Path):
    rows = []
    for directory in sorted(agent_root.glob("*/_*_transcripts")):
        tag = directory.parent.name
        kind = "claude" if "claude" in directory.name else "codex"
        for path in sorted(directory.glob("*.jsonl")):
            first = None
            init = None
            earliest = None
            tool_names = Counter()
            line_count = 0
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        line_count += 1
                        try:
                            event = json.loads(line)
                        except Exception:
                            continue
                        if first is None:
                            first = event
                        if event.get("type") == "system" and event.get("subtype") == "init":
                            init = event
                        timestamp = event.get("timestamp")
                        if timestamp and (earliest is None or timestamp < earliest):
                            earliest = timestamp
                        if event.get("type") == "assistant":
                            for item in (event.get("message") or {}).get("content") or []:
                                if item.get("type") == "tool_use":
                                    tool_names[item.get("name")] += 1
                        elif event.get("type") in {"item.started", "item.completed"}:
                            item = event.get("item") or {}
                            if item.get("type"):
                                tool_names[item.get("type")] += 1
            except Exception:
                continue
            parts = path.stem.split("__")
            arm = parts[-3] if len(parts) >= 3 else "?"
            task_id = "__".join(parts[:-3]) if len(parts) >= 4 else "?"
            rows.append({
                "tag": tag, "kind": kind, "file": path.name, "task_id": task_id, "arm": arm,
                "bytes": path.stat().st_size, "lines": line_count, "earliest": earliest,
                "version": (init or first or {}).get("claude_code_version"),
                "model": (init or first or {}).get("model"),
                "permission_mode": (init or first or {}).get("permissionMode"),
                "advertised_tools": (init or first or {}).get("tools") or [],
                "advertised_skills": (init or first or {}).get("skills") or [],
                "used_tools": dict(tool_names),
            })
    grouped = {}
    for key in sorted({(r["tag"], r["kind"], r["arm"]) for r in rows}):
        subset = [r for r in rows if (r["tag"], r["kind"], r["arm"]) == key]
        grouped["::".join(key)] = {
            "n": len(subset), "bytes": sum(r["bytes"] for r in subset),
            "versions": dict(Counter(str(r["version"]) for r in subset)),
            "models": dict(Counter(str(r["model"]) for r in subset)),
            "permission_modes": dict(Counter(str(r["permission_mode"]) for r in subset)),
            "advertised_toolset_hashes": dict(Counter(hashlib.md5(json.dumps(r["advertised_tools"]).encode()).hexdigest()[:10] for r in subset)),
            "advertised_skill_counts": dict(Counter(len(r["advertised_skills"]) for r in subset)),
            "earliest": min((r["earliest"] for r in subset if r["earliest"]), default=None),
        }
    return rows, grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("eval/results"))
    parser.add_argument("--out", type=Path, default=Path("tmp/corpus_audit.json"))
    args = parser.parse_args()
    records, bad = load_records(args.root / "agent")
    active = [r for r in records if not r["_archived"]]

    by_tag = defaultdict(list)
    by_tag_arm = defaultdict(list)
    for r in active:
        by_tag[r["_tag"]].append(r)
        by_tag_arm[(r["_tag"], r["arm"])].append(r)

    tags = {tag: summarize_group(rows) for tag, rows in sorted(by_tag.items())}
    arms = {f"{tag}::{arm}": summarize_group(rows) for (tag, arm), rows in sorted(by_tag_arm.items())}

    pairs = []
    for tag, rows in sorted(by_tag.items()):
        arm_names = sorted({r["arm"] for r in rows})
        for left, right in combinations(arm_names, 2):
            pairs.append(paired_comparison(
                [r for r in rows if r["arm"] == left],
                [r for r in rows if r["arm"] == right], left, right, tag,
            ))

    retrieval_by_group = {
        "all_active": retrieval_outcome(active),
        **{f"tag::{tag}": retrieval_outcome(rows) for tag, rows in sorted(by_tag.items())},
        **{f"arm::{tag}::{arm}": retrieval_outcome(rows) for (tag, arm), rows in sorted(by_tag_arm.items())},
    }

    duplicate_keys = []
    key_rows = defaultdict(list)
    for r in active:
        key_rows[(r["_tag"], r.get("task_id"), r.get("arm"), r.get("repeat", 0))].append(r["_path"])
    for key, paths in key_rows.items():
        if len(paths) > 1:
            duplicate_keys.append({"key": key, "paths": paths})

    field_presence = Counter()
    for r in active:
        field_presence.update(k for k, v in r.items() if v is not None and not k.startswith("_"))

    retr_sets, retr_pairs = standalone_retrieval(args.root)
    transcript_rows, transcript_groups = transcript_inventory(args.root / "agent")
    output = {
        "scope": {
            "records_total": len(records), "records_active": len(active),
            "records_archived": sum(r["_archived"] for r in records),
            "unique_active_tasks": len({r.get("task_id") for r in active}),
            "graded_active": sum(r["_graded"] for r in active),
            "bad_json": bad,
        },
        "tags": tags,
        "arms": arms,
        "paired": pairs,
        "retrieval_outcome": retrieval_by_group,
        "invariant_issues": invariant_issues(records),
        "duplicate_keys": duplicate_keys,
        "cross_panel_stability": cross_panel_stability(by_tag_arm),
        "field_presence": dict(field_presence.most_common()),
        "standalone_retrieval": [{k: v for k, v in d.items() if k != "tasks"} for d in retr_sets],
        "standalone_paired": retr_pairs,
        "transcript_inventory": transcript_groups,
        "transcript_rows": transcript_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["scope"], indent=2))
    print(f"tags={len(tags)} arms={len(arms)} pairs={len(pairs)} output={args.out}")


if __name__ == "__main__":
    main()
