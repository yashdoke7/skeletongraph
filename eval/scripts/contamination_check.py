"""Contamination / memorization diagnostics from EXISTING runs (no agent calls).

Answers the question: "is pass@1 just memorization, and are the arm differences
real?" Two evidence streams, both from data already on disk:

1. PAIRED solve-contingency vs a reference arm (default `none` = no retrieval).
   For the SAME task + SAME model, memorization is a constant, so the 2x2
   contingency isolates retrieval's MARGINAL effect:
     arm-only  : retrieval solved, none failed  → genuine retrieval WIN
     none-only : none solved, retrieval failed   → retrieval HURT (distraction)
     both / neither
   net = arm_only - none_only  (>0 ⇒ retrieval adds value beyond memory).

2. MEMORIZATION FINGERPRINT for `none`: on the tasks `none` SOLVES with zero
   retrieval, how fast does it open the gold file? time_to_first_gold_read_turn
   near 0 with few file reads = it "knew" the path = memorization signature.

Usage:
    python -m eval.scripts.contamination_check
    python -m eval.scripts.contamination_check --runs-dir eval/results/agent/nemotron3_final --ref none
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _load(runs_dir: Path) -> list:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        if p.name.startswith("_") or p.name.startswith("summary") or p.name == "summary.json":
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("arm") and r.get("task_id"):
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("eval/results/agent/nemotron3_final"))
    ap.add_argument("--ref", default="none", help="reference arm (no-retrieval)")
    args = ap.parse_args()

    recs = _load(args.runs_dir)
    # resolved map per arm (only VERIFIED runs have `resolved`)
    solved: dict = defaultdict(dict)       # arm -> {task_id: bool}
    ttfgr: dict = defaultdict(list)        # arm -> [time_to_first_gold_read on solved]
    reads_to_gold: dict = defaultdict(list)
    for r in recs:
        if "resolved" not in r:
            continue
        arm, tid = r["arm"], r["task_id"]
        solved[arm][tid] = bool(r["resolved"])

    ref = args.ref
    if ref not in solved:
        raise SystemExit(f"reference arm {ref!r} has no verified runs in {args.runs_dir}")

    print(f"runs-dir: {args.runs_dir}")
    print(f"reference (no-retrieval): {ref}\n")
    print(f"{'arm':<18}{'nPair':>6}{'both':>6}{'armOnly':>8}{'noneOnly':>9}"
          f"{'neither':>8}{'net':>6}{'arm%':>7}{'ref%':>7}")
    print("-" * 75)
    ref_map = solved[ref]
    rows = []
    for arm in sorted(solved):
        if arm == ref:
            continue
        amap = solved[arm]
        shared = set(amap) & set(ref_map)
        if not shared:
            continue
        both = sum(1 for t in shared if amap[t] and ref_map[t])
        arm_only = sum(1 for t in shared if amap[t] and not ref_map[t])
        none_only = sum(1 for t in shared if not amap[t] and ref_map[t])
        neither = sum(1 for t in shared if not amap[t] and not ref_map[t])
        n = len(shared)
        net = arm_only - none_only
        rows.append((arm, n, both, arm_only, none_only, neither, net,
                     (both + arm_only) / n, (both + none_only) / n))
    # sort by net marginal value
    for arm, n, both, ao, no_, nei, net, ap_, rp_ in sorted(rows, key=lambda x: -x[6]):
        print(f"{arm:<18}{n:>6}{both:>6}{ao:>8}{no_:>9}{nei:>8}{net:>+6}"
              f"{ap_*100:>6.0f}%{rp_*100:>6.0f}%")

    # ── memorization fingerprint ──────────────────────────────────────────────
    print("\nMemorization fingerprint - time_to_first_gold_read_turn (TTFGR) on SOLVED tasks")
    print(f"{'arm':<18}{'nSolved':>8}{'medTTFGR':>10}{'<=1turn%':>10}{'meanReads':>11}")
    print("-" * 57)
    by_arm_solved: dict = defaultdict(list)
    for r in recs:
        if not r.get("resolved"):
            continue
        by_arm_solved[r["arm"]].append(r)
    for arm in sorted(by_arm_solved):
        rs = by_arm_solved[arm]
        tts = [r.get("time_to_first_gold_read_turn") for r in rs
               if r.get("time_to_first_gold_read_turn") is not None]
        if not tts:
            continue
        frac_fast = sum(1 for t in tts if t <= 1) / len(tts)
        reads = [len(r.get("files_read", []) or []) for r in rs]
        print(f"{arm:<18}{len(rs):>8}{st.median(tts):>10.1f}{frac_fast*100:>9.0f}%"
              f"{(sum(reads)/len(reads)):>11.1f}")

    print("\nRead: net>0 => retrieval solves tasks none cannot (genuine marginal value, "
          "memorization-controlled by pairing). Low TTFGR + few reads for `none` => it "
          "opens gold fast with no search = memorization signature.")

    _efficiency_section(recs)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def _efficiency_section(recs: list) -> None:
    """Does retrieval QUALITY track any OUTCOME? Per-arm table + correlations.

    The question the user is asking: is there ANY column where better retrieval
    shows up? Total turns/tokens are dominated by the patch loop, so the place to
    look is the LOCALIZATION phase: tte = turns-to-first-edit, and wall_s.
    """
    by_arm: dict = defaultdict(list)
    for r in recs:
        if "resolved" in r:                       # verified arms only
            by_arm[r["arm"]].append(r)

    def _m(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\nDoes retrieval QUALITY track any OUTCOME? (per arm)")
    print(f"{'arm':<18}{'pass1':>7}{'rec':>6}{'prec':>6}{'rank':>6}"
          f"{'tte':>6}{'wall_s':>8}{'turns':>7}{'intok':>9}")
    print("-" * 73)
    table = {}
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        # final cumulative recall = last search_call's cumulative_recall (0 if none)
        recs_cum = []
        for r in rs:
            sc = r.get("search_calls") or []
            recs_cum.append(sc[-1]["cumulative_recall"] if sc else 0.0)
        ranks = [r.get("retrieval_rank") for r in rs if (r.get("retrieval_rank") or 0) > 0]
        row = {
            "pass1": _m([1.0 if r.get("resolved") else 0.0 for r in rs]),
            "rec": _m(recs_cum),
            "prec": _m([r.get("retrieval_precision") for r in rs]),
            "rank": (sorted(ranks)[len(ranks) // 2] if ranks else float("nan")),
            "tte": _m([r.get("time_to_first_edit_turn") for r in rs]),
            "wall": _m([r.get("wall_s") for r in rs]),
            "turns": _m([r.get("n_turns") for r in rs]),
            "intok": _m([r.get("billed_input") for r in rs]),
        }
        table[arm] = row
        print(f"{arm:<18}{row['pass1']*100:>6.0f}%{row['rec']:>6.2f}{row['prec']:>6.2f}"
              f"{row['rank']:>6.1f}{row['tte']:>6.1f}{row['wall']:>8.0f}"
              f"{row['turns']:>7.1f}{row['intok']:>9.0f}")

    # cross-arm correlations: which OUTCOME tracks which retrieval-QUALITY signal?
    arms = [a for a in table if table[a]["rec"] == table[a]["rec"]]   # drop nan
    def col(k): return [table[a][k] for a in arms]
    print("\nCross-arm correlation (r) — does the OUTCOME move with retrieval quality?")
    for outcome in ("pass1", "tte", "wall", "turns"):
        for quality in ("rec", "prec", "rank"):
            xs = [table[a][quality] for a in arms if table[a][quality] == table[a][quality]]
            ys = [table[a][outcome] for a in arms if table[a][quality] == table[a][quality]]
            r = _pearson(xs, ys)
            print(f"  {outcome:<6} ~ {quality:<5} r={r:+.2f}")
    print("  (|r|>~0.6 with the EXPECTED sign = a real relationship; ~0 = retrieval "
          "quality is invisible in that outcome here.)")


if __name__ == "__main__":
    main()
