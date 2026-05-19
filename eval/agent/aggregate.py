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
        if p.name.startswith("_"):
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
    by_arm: dict = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]].append(r)

    lines = ["# Agentic Evaluation — Summary", ""]
    if stage:
        lines.append(f"Stage: **{stage}**  ·  {config.STAGES[stage].note}")
    lines += [f"Runs: {len(recs)}  ·  arms: {sorted(by_arm)}", "",
              "## Axes 1/2/3/5 by arm", "",
              "retrieval-hit = recall (gameable by noise dumps) · "
              "precision = gold/returned · rank = median rank of first gold file",
              "",
              "| Arm | n | pass@1 | retrieval-hit | precision | rank | "
              "edited-gold | avg turns | avg in-tok | avg out-tok | avg cost$ |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    arm_pass: dict = {}
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        resolved = [1 if r.get("resolved") else 0 for r in rs
                    if "resolved" in r]
        arm_pass[arm] = {r["task_id"]: bool(r.get("resolved")) for r in rs}
        p1 = _mean(resolved) if resolved else None
        ranks = [r.get("retrieval_rank") for r in rs
                 if r.get("retrieval_rank")]          # nonzero = found
        med_rank = round(statistics.median(ranks), 1) if ranks else "n/a"
        lines.append(
            f"| {config.ARMS.get(arm, arm).label if arm in config.ARMS else arm} "
            f"| {len(rs)} "
            f"| {p1 if p1 is not None else 'n/a (run verify.py)'} "
            f"| {_mean([1 if r.get('retrieval_hit') else 0 for r in rs])} "
            f"| {_mean([r.get('retrieval_precision') for r in rs])} "
            f"| {med_rank} "
            f"| {_mean([1 if r.get('edited_gold_file') else 0 for r in rs])} "
            f"| {_mean([r.get('n_turns') for r in rs])} "
            f"| {round(_mean([r.get('billed_input') for r in rs]))} "
            f"| {round(_mean([r.get('billed_output') for r in rs]))} "
            f"| {_mean([r.get('imputed_cost') for r in rs])} |"
        )

    # ── data integrity — SG runs that silently lost their embedding index ───
    lines += ["", "## Data integrity", ""]
    sg_runs = [r for r in recs if str(r.get("arm", "")).startswith("sg")]
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
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        cnt = defaultdict(int)
        for r in rs:
            cnt[r.get("stopped", "?")] += 1
        lines.append(f"| {arm} | {cnt['submit']} | {cnt['max_turns']} "
                     f"| {cnt['error']} | {cnt['no_tool']} |")

    out = config.RUNS_DIR / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines[:18]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    args = ap.parse_args()
    aggregate(args.stage)


if __name__ == "__main__":
    main()
