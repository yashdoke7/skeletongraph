"""Print every comparison table we've built ad-hoc in chat, from the command
line — so growing a Claude Code batch (native vs sg-fusion vs sg-fusion-v3 vs
serena, etc.) can be checked without spending a turn asking for it.

Usage:
    python -m eval.scripts.compare_arms --tag claude_v7 --arms native,sg-fusion
    python -m eval.scripts.compare_arms --tag claude_v7 --arms native,sg-fusion,sg-fusion-v3
    python -m eval.scripts.compare_arms --tag claude_v7 --arms native,sg-fusion --dataset <other.jsonl>

Prints, for the arms given (paired on their common task_ids):
  1. Overall pass@1 / cost / turns / tokens, with deltas vs the FIRST arm listed
     (treated as the baseline — put native first).
  2. Named vs unnamed split (does the issue text already name the gold file or
     function?) for both pass@1 and cost/turns.
  3. Quadrants for the LAST TWO arms given (both-pass / both-fail / each-only),
     with per-task detail on the disagreement cases.
  4. Retrieval quality: file-level hit rate vs FUNCTION-level hit rate (SG's
     real differentiator — native is 0% here structurally, it can't retrieve
     at function granularity).
  5. Tool-call mix per arm (where the turns/cost actually go).
  6. Outlier flag: any task whose cost is >2.5x that arm's own median — so a
     single volatile run doesn't get silently averaged into a headline number
     unnoticed (see project_fusion_cost_diagnosis memory for why this matters).

Reads run JSONs from eval/results/agent/<tag>/*.json. Needs the source dataset
(for `query`/`gold_fqns`, not stored in the run JSON) — defaults to the dataset
this project has been using for claude_v7; override with --dataset.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATASET = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl"


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


def is_sg_arm(arm: str) -> bool:
    """True for any SkeletonGraph-capable arm (sg, sg-fusion, sg-rerank, ...).

    Non-SG arms (native, cbmem, serena, ...) always carry sg_tool_calls==0 —
    they never had SG to call at all, so that 0 is legitimate, not a missed
    adoption event. Only SG arms should have the sg_tool_calls==0 exclusion
    applied to their retrieval metrics.
    """
    return arm == "sg" or arm.startswith("sg-")


def gold_func_hit(rec: dict, gold_fqns: list) -> bool:
    if not gold_fqns:
        return False
    gold_pairs = set()
    for fq in gold_fqns:
        parts = fq.split("::")
        f = parts[0].replace("\\", "/")
        sym = parts[-1].split(".")[-1].lower() if len(parts) > 1 else ""
        gold_pairs.add((f, sym))
    fqns = rec.get("all_search_fqns") or rec.get("first_search_fqns") or []
    for fq in fqns:
        fq = str(fq)
        f = fq.split("::")[0].replace("\\", "/")
        sym = fq.split("::")[-1].split(".")[-1].lower() if "::" in fq else ""
        if (f, sym) in gold_pairs:
            return True
    return False


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n/d*100:.0f}%)" if d else "n/a"


def delta(new: float, base: float) -> str:
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.1f}%"


def section(title: str) -> None:
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="SG_EVAL_RUN_TAG results dir under eval/results/agent/")
    ap.add_argument("--arms", required=True, help="comma-separated arm names, baseline FIRST (e.g. native,sg-fusion)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="source dataset jsonl (for query/gold_fqns)")
    ap.add_argument("--outlier-mult", type=float, default=2.5, help="flag tasks costing >Nx an arm's own median")
    args = ap.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if len(arm_names) < 2:
        raise SystemExit("need at least 2 arms to compare")

    arms = {a: load_arm(args.tag, a) for a in arm_names}
    for a, d in arms.items():
        print(f"loaded {a}: {len(d)} runs")

    common = set.intersection(*(set(d) for d in arms.values()))
    order = list(load_dataset(args.dataset).keys())
    common = [t for t in order if t in common]
    ds = load_dataset(args.dataset)
    print(f"\npaired common task_ids across {arm_names}: {len(common)}")
    if not common:
        raise SystemExit("no common tasks — check --tag/--arms/--dataset")

    baseline = arm_names[0]

    # ---- 1. overall ----
    section("1. OVERALL")
    header = f"{'arm':16} {'pass@1':>14} {'cost/task':>12} {'turns/task':>12} {'tok/task':>12}"
    print(header)
    stats = {}
    for a in arm_names:
        recs = [arms[a][t] for t in common]
        resolved = sum(1 for r in recs if r.get("resolved"))
        cost = sum(r.get("imputed_cost", 0) or 0 for r in recs)
        turns = sum(r.get("n_turns", 0) or 0 for r in recs)
        tok = sum(r.get("total_input_tokens", 0) or 0 for r in recs)
        n = len(recs)
        stats[a] = dict(resolved=resolved, cost=cost, turns=turns, tok=tok, n=n)
        print(f"{a:16} {pct(resolved, n):>14} {'$'+format(cost/n, '.3f'):>12} {turns/n:>12.1f} {tok/n:>12,.0f}")
    print()
    for a in arm_names[1:]:
        print(f"  {a} vs {baseline}: cost {delta(stats[a]['cost'], stats[baseline]['cost'])}  "
              f"turns {delta(stats[a]['turns'], stats[baseline]['turns'])}  "
              f"tokens {delta(stats[a]['tok'], stats[baseline]['tok'])}")

    # ---- 2. named vs unnamed ----
    section("2. NAMED vs UNNAMED (does the issue text already name the gold file/function?)")
    named_set = [t for t in common if names_gold(ds[t].get("query", ""), ds[t].get("gold_files", []), ds[t].get("gold_fqns", []))]
    unnamed_set = [t for t in common if t not in named_set]
    for label, subset in [("NAMED", named_set), ("UNNAMED", unnamed_set)]:
        if not subset:
            continue
        print(f"\n--- {label} (n={len(subset)}) ---")
        sub_stats = {}
        for a in arm_names:
            recs = [arms[a][t] for t in subset]
            resolved = sum(1 for r in recs if r.get("resolved"))
            cost = sum(r.get("imputed_cost", 0) or 0 for r in recs)
            turns = sum(r.get("n_turns", 0) or 0 for r in recs)
            n = len(recs)
            sub_stats[a] = dict(resolved=resolved, cost=cost, turns=turns, n=n)
            print(f"  {a:16} {pct(resolved, n):>14}   cost/task=${cost/n:.3f}   turns/task={turns/n:.1f}")
        for a in arm_names[1:]:
            print(f"    {a} vs {baseline}: cost {delta(sub_stats[a]['cost'], sub_stats[baseline]['cost'])}  "
                  f"turns {delta(sub_stats[a]['turns'], sub_stats[baseline]['turns'])}")

    # ---- 3. quadrants (last two arms) ----
    a1, a2 = arm_names[0], arm_names[-1]
    section(f"3. QUADRANTS — {a1} vs {a2}")
    both_pass = [t for t in common if arms[a1][t].get("resolved") and arms[a2][t].get("resolved")]
    both_fail = [t for t in common if not arms[a1][t].get("resolved") and not arms[a2][t].get("resolved")]
    a2_only = [t for t in common if not arms[a1][t].get("resolved") and arms[a2][t].get("resolved")]
    a1_only = [t for t in common if arms[a1][t].get("resolved") and not arms[a2][t].get("resolved")]
    print(f"both pass: {len(both_pass)}   both fail: {len(both_fail)}   "
          f"{a2}-only pass: {len(a2_only)}   {a1}-only pass: {len(a1_only)}")
    if both_pass:
        nc = sum(arms[a1][t]["imputed_cost"] for t in both_pass); fc = sum(arms[a2][t]["imputed_cost"] for t in both_pass)
        nt = sum(arms[a1][t]["n_turns"] for t in both_pass); ft = sum(arms[a2][t]["n_turns"] for t in both_pass)
        print(f"  both-pass efficiency (n={len(both_pass)}): cost {delta(fc, nc)}  turns {delta(ft, nt)}")
    if a1_only:
        print(f"\n  {a1}-only-pass tasks ({a1} won, {a2} lost):")
        for t in a1_only:
            r2 = arms[a2][t]
            print(f"    {t:38} {a2}_hit={r2.get('retrieval_hit')} rank={r2.get('retrieval_rank')} "
                  f"turns={arms[a1][t]['n_turns']}/{r2['n_turns']}")
    if a2_only:
        print(f"\n  {a2}-only-pass tasks ({a2} won, {a1} lost) — first 10:")
        for t in a2_only[:10]:
            r2 = arms[a2][t]
            print(f"    {t:38} {a2}_hit={r2.get('retrieval_hit')} rank={r2.get('retrieval_rank')} "
                  f"turns={arms[a1][t]['n_turns']}/{r2['n_turns']}")
        if len(a2_only) > 10:
            print(f"    ... ({len(a2_only)} total)")

    # ---- 4. retrieval quality: file vs function level ----
    section("4. RETRIEVAL QUALITY — file-level vs FUNCTION-level hit rate")
    tasks_with_fqn = [t for t in common if ds[t].get("gold_fqns")]
    for a in arm_names:
        # An SG-capable arm where the agent made ZERO sg_* tool calls never had
        # a chance to hit — that's the model choosing native tools that turn,
        # not a retrieval failure. Counting it as retrieval_hit=False silently
        # drags the aggregate down and misattributes an adoption question to a
        # quality question. Non-SG arms (native, cbmem, ...) ALWAYS report
        # sg_tool_calls==0 legitimately, so this must gate on the arm itself.
        used = ([t for t in common if arms[a][t].get("sg_tool_calls") != 0]
                if is_sg_arm(a) else list(common))
        skipped = len(common) - len(used)
        file_hit = sum(1 for t in used if arms[a][t].get("retrieval_hit"))
        used_fqn = [t for t in tasks_with_fqn if t in set(used)]
        func_hit = sum(1 for t in used_fqn if gold_func_hit(arms[a][t], ds[t].get("gold_fqns")))
        note = f"  ({skipped}/{len(common)} tasks: 0 sg_* calls, excluded)" if skipped else ""
        print(f"  {a:16} file-hit {pct(file_hit, len(used))}   function-hit {pct(func_hit, len(used_fqn))}{note}")

    # ---- 5. tool-call mix ----
    section("5. TOOL-CALL MIX (avg/task)")
    for a in arm_names:
        tc = Counter()
        for t in common:
            for k, v in (arms[a][t].get("tool_counts") or {}).items():
                tc[k] += v
        n = len(common)
        print(f"\n  {a}:")
        for k, v in tc.most_common():
            print(f"    {k:32} {v/n:.2f}/task")

    # ---- 6. outlier flag ----
    section(f"6. OUTLIERS (cost > {args.outlier_mult}x the arm's own median)")
    for a in arm_names:
        costs = sorted(arms[a][t].get("imputed_cost", 0) or 0 for t in common)
        med = statistics.median(costs) if costs else 0
        flagged = [(t, arms[a][t]["imputed_cost"]) for t in common if (arms[a][t].get("imputed_cost") or 0) > med * args.outlier_mult]
        if flagged:
            print(f"\n  {a} (median=${med:.3f}):")
            for t, c in sorted(flagged, key=lambda x: -x[1]):
                print(f"    {t:38} ${c:.3f}  ({c/med:.1f}x median)")
        else:
            print(f"  {a}: no outliers above {args.outlier_mult}x median (${med:.3f})")


if __name__ == "__main__":
    main()
