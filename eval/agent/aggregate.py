"""Aggregate every run JSON into the Axis 1/2/3/5 tables + significance tests.

    python -m eval.agent.aggregate
    python -m eval.agent.aggregate --stage B

Reads eval/results/agent/*.json (written by run_agent, verdicts by verify) and
writes eval/results/agent/SUMMARY.md.

Significance: McNemar's exact test on paired pass/fail (SG vs each baseline),
bootstrap CI on retrieval-hit rate. Pure-stdlib — no scipy needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from . import config


def _load(stage: str | None) -> list:
    recs = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == "summary.json":
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if stage and stage in config.STAGES:
            st = config.STAGES[stage]
            if r.get("arm") not in st.arms or r.get("model") not in st.models:
                continue
        recs.append(r)
    return recs


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else 0.0


# ── significance ────────────────────────────────────────────────────────────


def mcnemar(pairs: list) -> tuple:
    """Exact McNemar on a list of (sg_pass, base_pass) bools. Returns (b, c, p)."""
    b = sum(1 for s, o in pairs if s and not o)   # SG wins
    c = sum(1 for s, o in pairs if o and not s)   # baseline wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial p
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def bootstrap_ci(values: list, iters: int = 2000) -> tuple:
    """95% bootstrap CI of the mean of a 0/1 list."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(config.SEED)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (round(means[int(0.025 * iters)], 4),
            round(means[int(0.975 * iters)], 4))


# ── main ────────────────────────────────────────────────────────────────────


def aggregate(stage: str | None) -> None:
    recs = _load(stage)
    if not recs:
        raise SystemExit("no runs found")

    # group by arm (collapsing repeats/models for the headline table)
    by_arm_all: dict = defaultdict(list)
    for r in recs:
        by_arm_all[r["arm"]].append(r)
    # Metric tables count only COMPLETED runs. A run that errored (endpoint down)
    # or never produced a tool call has no meaningful retrieval/edit data, and
    # including it as zeros silently drags every average down — and lets a stale
    # failed run from an earlier session pollute the summary. Failures are still
    # surfaced in the Stop-reason table below.
    _COMPLETE = ("submit", "max_turns")
    by_arm: dict = {a: [r for r in rs if r.get("stopped") in _COMPLETE]
                    for a, rs in by_arm_all.items()}
    n_excluded = len(recs) - sum(len(v) for v in by_arm.values())

    lines = ["# Agentic Evaluation — Summary", ""]
    if stage:
        lines.append(f"Stage: **{stage}**  ·  {config.STAGES[stage].note}")
    lines += [f"Runs: {len(recs)} ({n_excluded} incomplete excluded from metrics)"
              f"  ·  arms: {sorted(by_arm)}", "",
              "## Headline results",
              "",
              "**pass@1** (SWE-bench resolved %) is the primary metric. "
              "Retrieval recall/precision/rank measure retrieval quality "
              "independent of task success. Tokens / turns / cost measure "
              "efficiency (paper headline is cost-at-iso-accuracy).",
              "",
              "edited-gold is a localization proxy — kept for diagnosis but "
              "NOT a primary metric (model-dependent and noisy).",
              ""]

    # Column-aligned monospace table (easier to read than pipe-separated).
    # The two-line header makes copy-paste into a paper Markdown clean.
    arm_pass: dict = {}
    rows = []
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        resolved = [1 if r.get("resolved") else 0 for r in rs
                    if "resolved" in r]
        arm_pass[arm] = {r["task_id"]: bool(r.get("resolved")) for r in rs}
        p1 = _mean(resolved) if resolved else None
        ranks = [r.get("retrieval_rank") for r in rs
                 if r.get("retrieval_rank")]          # nonzero = found
        med_rank = round(statistics.median(ranks), 1) if ranks else None
        label = config.ARMS.get(arm, arm).label if arm in config.ARMS else arm
        rows.append({
            "arm": label,
            "n": len(rs),
            "pass1": p1,
            "recall": _mean([1 if r.get('retrieval_hit') else 0 for r in rs]),
            "prec": _mean([r.get('retrieval_precision') for r in rs]),
            "rank": med_rank,
            "egold": _mean([1 if r.get('edited_gold_file') else 0 for r in rs]),
            "turns": _mean([r.get('n_turns') for r in rs]),
            "intok": round(_mean([r.get('billed_input') for r in rs])),
            "outtok": round(_mean([r.get('billed_output') for r in rs])),
            "cost": _mean([r.get('imputed_cost') for r in rs]),
        })

    # Markdown table (pipe form) — for paste-into-paper compatibility
    lines += [
        "| arm | n | pass@1 | recall | prec | rank | tokens | turns | cost$ | (egold) |",
        "| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|",
    ]
    for r in rows:
        p1 = f"{r['pass1']*100:.1f}%" if r['pass1'] is not None else "n/a"
        rank = f"{r['rank']:.1f}" if r['rank'] is not None else "—"
        lines.append(
            f"| {r['arm']} | {r['n']} | {p1} | "
            f"{r['recall']:.3f} | {r['prec']:.3f} | {rank} | "
            f"{r['intok']:>6,} | {r['turns']:.1f} | "
            f"${r['cost']:.4f} | _{r['egold']:.2f}_ |"
        )

    # Console-friendly monospace echo (printed below the Markdown table).
    lines += ["", "```", "Compact view (sorted by pass@1):", ""]
    sorted_rows = sorted(rows, key=lambda r: -(r['pass1'] or 0))
    lines.append(
        f"{'arm':<28} {'n':>3} {'pass@1':>7} {'recall':>7} "
        f"{'prec':>6} {'rank':>5} {'tok':>7} {'turns':>5} {'$':>7}"
    )
    lines.append("-" * 90)
    for r in sorted_rows:
        p1 = f"{r['pass1']*100:5.1f}%" if r['pass1'] is not None else "  n/a "
        rank = f"{r['rank']:.1f}" if r['rank'] is not None else "  — "
        lines.append(
            f"{r['arm']:<28} {r['n']:>3} {p1:>7} {r['recall']:>7.3f} "
            f"{r['prec']:>6.3f} {rank:>5} {r['intok']:>7,} "
            f"{r['turns']:>5.1f} ${r['cost']:>6.4f}"
        )
    lines.append("```")

    # ── data integrity — SG runs that silently lost their embedding index ───
    lines += ["", "## Data integrity", ""]
    # sg-noembed runs WITHOUT embeddings on purpose — don't flag it as degraded.
    sg_runs = [r for a, rs in by_arm.items()
               if a.startswith("sg") and a != "sg-noembed" for r in rs]
    degraded = [r for r in sg_runs if r.get("embeddings_used") is False]
    if degraded:
        lines.append(f"**WARNING — {len(degraded)}/{len(sg_runs)} SG run(s) ran "
                      f"WITHOUT embeddings (BM25-only fallback). Their numbers "
                      f"are NOT real SG — exclude or re-run:**")
        for r in degraded:
            lines.append(f"- `{r.get('run_id')}`")
    elif sg_runs:
        lines.append(f"All {len(sg_runs)} SG run(s) used the embedding index. OK")
    else:
        lines.append("_No SG runs in this selection._")

    # ── trajectory dynamics ─────────────────────────────────────────────────
    # The interesting agent-behavior signals: when does the agent first edit?
    # how many edits does it attempt vs land? does the empty-submit guard fire?
    # These differentiate retrieval quality from model capability.
    lines += ["", "## Trajectory dynamics by arm", "",
              "edits-success-rate = successful / attempted edit_file calls · "
              "guard-fired% = empty-submit guard had to nudge the model · "
              "tte = mean turn index of first successful edit",
              "",
              "| Arm | n | edits attempted | edits successful | "
              "success-rate | guard-fired% | mean tte |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        att = [r.get("edits_attempted", 0) for r in rs]
        suc = [r.get("edits_successful", 0) for r in rs]
        rate = round(sum(suc) / sum(att), 4) if sum(att) else 0.0
        guard = _mean([1 if r.get("empty_submit_blocked") else 0 for r in rs])
        ttes = [r.get("time_to_first_edit_turn") for r in rs
                if r.get("time_to_first_edit_turn") is not None]
        mean_tte = round(sum(ttes) / len(ttes), 1) if ttes else "n/a"
        lines.append(
            f"| {arm} | {len(rs)} | {round(_mean(att), 2)} "
            f"| {round(_mean(suc), 2)} | {rate} | {guard} | {mean_tte} |"
        )

    # ── consolidation gap (the ContextBench-style headline figure) ──────────
    # Files retrieved but never appearing in the final patch. Lower is better;
    # SG with summaries should focus the model and shrink this gap vs naive
    # retrievers that surface noise.
    lines += ["", "## Consolidation gap by arm", "",
              "files-read = mean files opened with read_file · "
              "files-in-patch = mean files touched by patch · "
              "gap-files = 1 − (read ∩ patch) / read   (0 = perfect)",
              "",
              "| Arm | n | files-read | files-in-patch | gap-files |",
              "| --- | --- | --- | --- | --- |"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        gaps = [(r.get("consolidation") or {}).get("consolidation_gap_files")
                for r in rs]
        read_c = [(r.get("consolidation") or {}).get("files_read_count", 0)
                  for r in rs]
        patch_c = [(r.get("consolidation") or {}).get("files_in_patch_count", 0)
                   for r in rs]
        lines.append(
            f"| {arm} | {len(rs)} | {round(_mean(read_c), 2)} "
            f"| {round(_mean(patch_c), 2)} | {_mean(gaps)} |"
        )

    # ── search dynamics ─────────────────────────────────────────────────────
    # Did the agent thrash on retrieval? Multiple searches per task with high
    # search_calls but low retrieval_precision = thrashing. Useful for the
    # "smart retrieval reduces tool calls" claim.
    lines += ["", "## Search dynamics by arm", "",
              "| Arm | n | mean search-calls | mean unique-files-retrieved | search-errors% |",
              "| --- | --- | --- | --- | --- |"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        ncalls = [r.get("n_search_calls", 0) for r in rs]
        uniq = [r.get("unique_files_retrieved_total", 0) for r in rs]
        errs = []
        for r in rs:
            scs = r.get("search_calls") or []
            errs.append(sum(1 for sc in scs if sc.get("error")) / max(len(scs), 1))
        lines.append(
            f"| {arm} | {len(rs)} | {round(_mean(ncalls), 2)} "
            f"| {round(_mean(uniq), 2)} | {round(_mean(errs), 4)} |"
        )

    # significance: SG vs each baseline, paired on task_id
    lines += ["", "## Significance — SG vs each baseline (McNemar, paired)", ""]
    if "sg" in arm_pass and any("resolved" in r for r in recs):
        lines += ["| Baseline | SG-only wins | base-only wins | p-value | "
                  "verdict |", "| --- | --- | --- | --- | --- |"]
        sg = arm_pass["sg"]
        for arm in sorted(by_arm):
            if arm == "sg":
                continue
            base = arm_pass.get(arm, {})
            common = sorted(set(sg) & set(base))
            pairs = [(sg[t], base[t]) for t in common]
            if not pairs:
                continue
            b, c, p = mcnemar(pairs)
            verdict = ("SG better" if b > c and p < 0.05 else
                       "baseline better" if c > b and p < 0.05 else
                       "no sig. difference")
            lines.append(f"| {arm} | {b} | {c} | {p:.4f} | {verdict} |")
    else:
        lines.append("_Run verify.py first — no pass/fail verdicts yet._")

    # retrieval-hit CI per arm
    lines += ["", "## Retrieval-hit rate — 95% bootstrap CI", "",
              "| Arm | hit-rate | 95% CI |", "| --- | --- | --- |"]
    for arm in sorted(by_arm):
        vals = [1 if r.get("retrieval_hit") else 0 for r in by_arm[arm]]
        lo, hi = bootstrap_ci(vals)
        lines.append(f"| {arm} | {_mean(vals)} | [{lo}, {hi}] |")

    # failure taxonomy
    lines += ["", "## Stop reason (failure mode) by arm", "",
              "| Arm | submit | max_turns | error | no_tool |",
              "| --- | --- | --- | --- | --- |"]
    for arm in sorted(by_arm_all):
        rs = by_arm_all[arm]          # ALL runs — failures belong in this table
        cnt = defaultdict(int)
        for r in rs:
            cnt[r.get("stopped", "?")] += 1
        lines.append(f"| {arm} | {cnt['submit']} | {cnt['max_turns']} "
                     f"| {cnt['error']} | {cnt['no_tool']} |")

    out = config.RUNS_DIR / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines[:18]))
    print("...")
    print(f"({len(lines)} total lines in SUMMARY.md — open the file for the full breakdown)")

    # Machine-readable summary alongside the human-readable SUMMARY.md.
    # Keyed by arm — retrieval/efficiency numbers for downstream scripting.
    summary = {}
    for arm in sorted(by_arm):
        recs = by_arm[arm]
        complete = [r for r in by_arm_all[arm]
                    if r.get("stopped") in ("submit", "max_turns")]
        pass_vals = []
        for r in recs:
            if "resolved" in r:
                pass_vals.append(1 if r.get("resolved") else 0)
            elif "verdict" in r and r.get("verdict") is not None:
                pass_vals.append(1 if r.get("verdict") else 0)
        summary[arm] = {
            "n": len(recs),
            "n_complete": len(complete),
            "retrieval_hit": _mean([r["retrieval_hit"] for r in recs
                                    if "retrieval_hit" in r]),
            "precision": _mean([r["retrieval_precision"] for r in recs
                                if "retrieval_precision" in r]),
            "edited_gold": _mean([r["edited_gold_file"] for r in recs
                                  if "edited_gold_file" in r]),
            "avg_turns": _mean([r["n_turns"] for r in complete
                                if "n_turns" in r]),
            "avg_input_tok": _mean([r["billed_input"] for r in complete
                                    if "billed_input" in r]),
            "avg_output_tok": _mean([r["billed_output"] for r in complete
                                     if "billed_output" in r]),
            "total_cost_usd": round(sum(r.get("imputed_cost", 0)
                                        for r in complete), 4),
            "pass1": _mean(pass_vals) if pass_vals else None,
        }
    jsout = config.RUNS_DIR / "summary.json"
    jsout.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {jsout}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None,
                    help="Filter to a specific stage's arms (e.g. 0-full)")
    ap.add_argument("--run-dir", default=None, type=Path,
                    help="Override RUNS_DIR (e.g. eval/results/agent/qwen7b_swebench)")
    args = ap.parse_args()
    if args.run_dir:
        config.RUNS_DIR = Path(args.run_dir).expanduser().resolve()
    aggregate(args.stage)


if __name__ == "__main__":
    main()
