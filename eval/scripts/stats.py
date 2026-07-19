"""Statistical rigor pass over two arms' paired results — McNemar exact test on
pass@1, bootstrap CIs + effect size on cost/turns deltas. Companion to
compare_arms.py (which gives the descriptive tables); this gives the numbers a
paper reviewer or a skeptical reader will actually ask for.

Usage:
    python -m eval.scripts.stats --tag claude_v7 --arms native,sg-fusion
    python -m eval.scripts.stats --tag claude_v7 --arms native,sg-fusion --subset both_pass
    python -m eval.scripts.stats --tag claude_v7 --arms native,sg-fusion --subset named

--subset restricts the cost/turns bootstrap to a slice (default: all common
tasks). "both_pass" = matched-outcome tasks only (both arms resolved) — the
cleanest apples-to-apples cost comparison, no confound from task difficulty.
"named"/"unnamed" = does the issue text already name the gold file/function.

Prints:
  1. McNemar's EXACT test on the pass@1 discordant pairs (arm2 vs arm1). This
     is the right test for paired binary outcomes on the SAME tasks — a plain
     two-proportion z-test is wrong here because the tasks aren't independent
     draws, they're matched pairs.
  2. Paired bootstrap 95% CI on the mean cost delta and mean turns delta
     (resample task indices with replacement, 10000 iterations) — so "-14.6%"
     comes with an interval, not just a point estimate.
  3. Effect size (Cohen's d for paired differences) on cost.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import statistics
from pathlib import Path

from scipy.stats import binomtest

DEFAULT_DATASET = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl"
N_BOOT = 10000


def load_arm(tag: str, arm: str) -> dict:
    d = {}
    for f in glob.glob(f"eval/results/agent/{tag}/*__{arm}__*.json"):
        if "_INDEX" in f or "summary" in f:
            continue
        try:
            rec = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        d[rec["task_id"]] = rec
    return d


def load_dataset(path: str) -> dict:
    ds = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ds[rec["task_id"]] = rec
    return ds


def names_gold(query: str, gold_files: list, gold_fqns: list) -> bool:
    q = (query or "").lower()
    for gf in gold_files or []:
        base = Path(gf).stem.lower()
        fname = Path(gf).name.lower()
        if fname in q or (len(base) > 3 and re.search(r"\b" + re.escape(base) + r"\b", q)):
            return True
    for fq in gold_fqns or []:
        sym = fq.split("::")[-1].split(".")[-1].lower()
        if len(sym) > 3 and re.search(r"\b" + re.escape(sym) + r"\b", q):
            return True
    return False


def mcnemar_exact(a1_wins: int, a2_wins: int) -> float:
    """Exact binomial McNemar test on discordant pairs (two-sided)."""
    n = a1_wins + a2_wins
    if n == 0:
        return 1.0
    return binomtest(min(a1_wins, a2_wins), n, 0.5).pvalue


def bootstrap_aggregate_pct_ci(vals1: list, vals2: list, n_boot: int = N_BOOT, seed: int = 42) -> tuple:
    """CI on the AGGREGATE %% delta (sum(v2)/sum(v1) - 1) — matches the single
    number reported everywhere else (compare_arms.py, the "-14.6%" headline).
    NOT a bootstrap of the mean of per-task %% deltas: that metric is unstable
    when some tasks have near-zero cost (a $0.02 diff on a $0.05 task reads as
    +40%), and it does not equal the aggregate ratio — mean-of-ratios !=
    ratio-of-means. Resampling task PAIRS (not values independently) preserves
    the pairing, which is what makes this a valid paired bootstrap."""
    n = len(vals1)
    if n == 0 or sum(vals1) == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    idx = list(range(n))
    pcts = []
    for _ in range(n_boot):
        sample = [idx[rng.randrange(n)] for _ in range(n)]
        s1 = sum(vals1[i] for i in sample)
        s2 = sum(vals2[i] for i in sample)
        if s1 > 0:
            pcts.append((s2 - s1) / s1 * 100)
    pcts.sort()
    point = (sum(vals2) - sum(vals1)) / sum(vals1) * 100
    lo = pcts[int(0.025 * len(pcts))]
    hi = pcts[int(0.975 * len(pcts))]
    return (point, lo, hi)


def cohens_d_paired(diffs: list) -> float:
    if len(diffs) < 2:
        return 0.0
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    return mean / sd if sd > 0 else 0.0


def section(title: str) -> None:
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True, help="exactly 2 arms, baseline FIRST (e.g. native,sg-fusion)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--subset", choices=["all", "both_pass", "named", "unnamed"], default="all")
    args = ap.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if len(arm_names) != 2:
        raise SystemExit("need exactly 2 arms")
    a1, a2 = arm_names

    arms = {a: load_arm(args.tag, a) for a in arm_names}
    for a, d in arms.items():
        print(f"loaded {a}: {len(d)} runs")

    common = sorted(set(arms[a1]) & set(arms[a2]))
    ds = load_dataset(args.dataset)
    print(f"paired common task_ids: {len(common)}")
    if not common:
        raise SystemExit("no common tasks")

    # ---- 1. McNemar on pass@1 (always on the FULL common set — subset doesn't apply here) ----
    section(f"1. McNEMAR EXACT TEST — pass@1, {a1} vs {a2} (n={len(common)})")
    a1_resolved = sum(1 for t in common if arms[a1][t].get("resolved"))
    a2_resolved = sum(1 for t in common if arms[a2][t].get("resolved"))
    a1_only = [t for t in common if arms[a1][t].get("resolved") and not arms[a2][t].get("resolved")]
    a2_only = [t for t in common if not arms[a1][t].get("resolved") and arms[a2][t].get("resolved")]
    print(f"  {a1}: {a1_resolved}/{len(common)} ({a1_resolved/len(common)*100:.1f}%)")
    print(f"  {a2}: {a2_resolved}/{len(common)} ({a2_resolved/len(common)*100:.1f}%)")
    print(f"  discordant pairs: {a1}-only-win={len(a1_only)}  {a2}-only-win={len(a2_only)}")
    p = mcnemar_exact(len(a1_only), len(a2_only))
    verdict = "SIGNIFICANT" if p < 0.05 else "NOT significant"
    print(f"  McNemar exact p-value: {p:.4f}  ({verdict} at alpha=0.05)")
    if p >= 0.05:
        print(f"  => pass@1 difference between {a1} and {a2} is NOT distinguishable from chance at this n.")
        print(f"     Do not claim '{a2} solves more' as a headline off this number alone.")

    # ---- pick the subset for cost/turns analysis ----
    if args.subset == "both_pass":
        subset = [t for t in common if arms[a1][t].get("resolved") and arms[a2][t].get("resolved")]
        label = f"BOTH-PASS (matched outcome, n={len(subset)})"
    elif args.subset in ("named", "unnamed"):
        named_set = {t for t in common if names_gold(ds[t].get("query", ""), ds[t].get("gold_files", []), ds[t].get("gold_fqns", []))}
        subset = [t for t in common if (t in named_set) == (args.subset == "named")]
        label = f"{args.subset.upper()} (n={len(subset)})"
    else:
        subset = common
        label = f"ALL PAIRED TASKS (n={len(subset)})"

    # ---- 2. bootstrap CI on cost delta ----
    section(f"2. COST DELTA — {a2} vs {a1} — {label}")
    cost1 = [arms[a1][t].get("imputed_cost", 0) or 0 for t in subset]
    cost2 = [arms[a2][t].get("imputed_cost", 0) or 0 for t in subset]
    mean1, mean2 = statistics.mean(cost1), statistics.mean(cost2)
    point, lo, hi = bootstrap_aggregate_pct_ci(cost1, cost2)
    print(f"  {a1} mean cost/task: ${mean1:.3f}   {a2} mean cost/task: ${mean2:.3f}")
    print(f"  aggregate cost delta: {point:+.1f}%  [95% CI: {lo:+.1f}%, {hi:+.1f}%]  (paired bootstrap, n_boot={N_BOOT})")
    abs_diffs = [c2 - c1 for c1, c2 in zip(cost1, cost2)]
    d = cohens_d_paired(abs_diffs)
    print(f"  Cohen's d (paired, absolute $ delta): {d:.3f}  ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")

    # ---- 3. bootstrap CI on turns delta ----
    section(f"3. TURNS DELTA — {a2} vs {a1} — {label}")
    turns1 = [arms[a1][t].get("n_turns", 0) or 0 for t in subset]
    turns2 = [arms[a2][t].get("n_turns", 0) or 0 for t in subset]
    mean_t1, mean_t2 = statistics.mean(turns1), statistics.mean(turns2)
    tpoint, tlo, thi = bootstrap_aggregate_pct_ci(turns1, turns2)
    print(f"  {a1} mean turns/task: {mean_t1:.1f}   {a2} mean turns/task: {mean_t2:.1f}")
    print(f"  aggregate turns delta: {tpoint:+.1f}%  [95% CI: {tlo:+.1f}%, {thi:+.1f}%]  (paired bootstrap, n_boot={N_BOOT})")


if __name__ == "__main__":
    main()
