"""Per-run-tag cost figures: built-in tools vs SkeletonGraph, on whatever
paired tasks currently exist for that tag.

Why this is separate from make_paper_figures.py
-----------------------------------------------
make_paper_figures.py produces the paper's CANONICAL figures and is deliberately
frozen to the published conditions (`claude_v7`, n=100) so a reviewer can
regenerate the exact PDFs from the released artifact. Its fig_tail/fig_scatter
hardcode that n in several places -- `set_xlim(1, 100)`, the median rule at
x=50, annotation anchors at indices 21/91, and the literal "-42.5% at p95".
Pointed at a 32-task run those would silently mislabel the plot.

This script takes the same data path and the same palette (imported, not
re-declared, so the two can never drift) and makes every one of those quantities
a function of the data. Point it at any tag, mid-run or finished.

Pairing
-------
Arms are intersected on task_id, exactly as the paper does: if native has 33
runs and sg-fusion has 32, the 32 in common are plotted and the unpaired one is
named in the output. Cost comparisons are only meaningful within a task, so an
unpaired run is never averaged in.

Only runs that actually completed are counted -- a task killed by a rate limit
mid-run has a partial cost that would drag its arm down. `--include-failed`
overrides this if you want to see everything.

Usage
-----
    # one tag
    python -m eval.scripts.make_pair_figures --tag claude_rebench_v1

    # several, separate figures each
    python -m eval.scripts.make_pair_figures --tag claude_rebench_v1 --tag claude_rebench_prose_v1

    # every tag on disk that has both arms paired
    python -m eval.scripts.make_pair_figures --all

    # numbers only, no plotting -- fast mid-run progress check
    python -m eval.scripts.make_pair_figures --all --stats-only

Figures land in eval/figures/ (gitignored scratch) as
`fig_tail__<tag>` and `fig_paired_scatter__<tag>`. When a run is final and you
want it in the paper, copy it into docs/paper/figures/ explicitly -- this script
never writes there, so it cannot clobber a published figure.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Palette + helpers come from the canonical figure script so the two can never
# drift apart. _save is NOT imported: it writes to docs/paper/figures.
from eval.scripts.make_paper_figures import (  # noqa: E402
    BASELINE_AXIS, BLUE, GRID, INK, INK_2, MUTED, SURFACE,
    RUNS, _clean, _style, load_arm,
)

OUT = Path("eval/figures")

# A run only counts if the agent finished on its own terms. Anything else
# (rate limit, timeout, crash) has a truncated cost that is not comparable.
_DONE = ("submit", "max_turns")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    wrote {OUT / name}.pdf / .png")


def _pct(vals: list[float], p: float) -> float:
    """Percentile, CEILING convention: v[ceil((n-1)*p)].

    This must match the paper exactly or the same run reports two different p95
    figures depending on which script you ran. Verified against the published
    tail table (FINDINGS.md sec 4, claude_v7 n=100), which this convention
    reproduces to the digit on all four rows:

        p50 $0.255 -> $0.260  +1.9%
        p75 $0.581 -> $0.489 -15.8%
        p90 $1.010 -> $0.752 -25.6%
        p95 $1.559 -> $0.896 -42.5%

    Linear interpolation -- the more common default, and what numpy and
    statistics.quantiles(method='inclusive') give -- yields -41.5% at p95 here
    instead. Both are defensible; only one is the published number. Do not
    "fix" this to interpolate.

    Note this convention takes the next observation UP, so at small n a
    "percentile" is close to the maximum: at n=15, p95 is just the top task.
    Read the tail columns on small tags as directional only.
    """
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[math.ceil((len(s) - 1) * p)]


def collect(tag: str, a_base: str, a_sg: str, include_failed: bool):
    """Paired, completed runs for one tag. Returns (base, sg, report)."""
    d1, d2 = load_arm(tag, a_base), load_arm(tag, a_sg)

    def ok(r):
        return include_failed or r.get("stopped") in _DONE

    g1 = {t: r for t, r in d1.items() if ok(r)}
    g2 = {t: r for t, r in d2.items() if ok(r)}
    common = sorted(set(g1) & set(g2))
    report = {
        "n_base": len(d1), "n_sg": len(d2),
        "dropped_base": sorted(set(d1) - set(g1)),
        "dropped_sg": sorted(set(d2) - set(g2)),
        "unpaired": sorted((set(g1) | set(g2)) - set(common)),
        "n_common": len(common),
    }
    return [g1[t] for t in common], [g2[t] for t in common], report


def stats(base, sg) -> dict:
    b = [r["imputed_cost"] for r in base]
    s = [r["imputed_cost"] for r in sg]
    tb, ts = sum(b), sum(s)
    out = {
        "n": len(b),
        "mean_base": statistics.mean(b), "mean_sg": statistics.mean(s),
        "total_delta_pct": (ts - tb) / tb * 100.0 if tb else 0.0,
        "cheaper": sum(1 for x, y in zip(b, s) if y < x),
        "turns_base": statistics.mean([r.get("n_turns", 0) for r in base]),
        "turns_sg": statistics.mean([r.get("n_turns", 0) for r in sg]),
    }
    for p, name in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95")):
        pb, ps = _pct(b, p), _pct(s, p)
        out[name] = (pb, ps, (ps - pb) / pb * 100.0 if pb else 0.0)
    return out


def fig_tail(base, sg, tag: str, st: dict) -> None:
    """Tasks ordered cheapest->most expensive by the baseline. Every annotation
    position and every quoted number is derived from this tag's own data."""
    pairs = sorted(zip((r["imputed_cost"] for r in base),
                       (r["imputed_cost"] for r in sg)), key=lambda p: p[0])
    nv = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    n = len(nv)
    x = list(range(1, n + 1))

    # Window scales with n: the published figure used 9 at n=100. At n=32 a
    # fixed 9 would smooth away the very divergence the plot exists to show.
    w = max(3, int(round(n / 11.0)) | 1)

    def smooth(vals):
        out = []
        for i in range(len(vals)):
            lo, hi = max(0, i - w // 2), min(len(vals), i + w // 2 + 1)
            out.append(sum(vals[lo:hi]) / (hi - lo))
        return out

    ns, ss = smooth(nv), smooth(sv)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.fill_between(x, ss, ns, where=[a > b for a, b in zip(ns, ss)],
                    color=BLUE, alpha=0.13, linewidth=0, interpolate=True)
    ax.plot(x, ns, color=INK_2, lw=2, label="Claude Code built-in tools")
    ax.plot(x, ss, color=BLUE, lw=2, label="+ SkeletonGraph")

    mid = (n + 1) / 2.0
    ax.axvline(mid, color=BASELINE_AXIS, lw=1, zorder=0)
    ax.text(mid, 0.55, " median task", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=MUTED, ha="left", va="center", rotation=90)

    # Anchors as fractions of n, not absolute indices.
    i_lo = max(0, min(n - 1, int(round(n * 0.22)) - 1))
    i_hi = max(0, min(n - 1, int(round(n * 0.92)) - 1))
    ax.annotate("curves overlap:\nno saving here", (i_lo + 1, ns[i_lo]),
                textcoords="offset points", xytext=(0, 34), ha="center",
                fontsize=7.8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=BASELINE_AXIS, lw=0.8))
    d95 = st["p95"][2]
    ax.annotate(f"the gap is the saving\n({d95:+.1f}% at p95)",
                (i_hi + 1, (ns[i_hi] + ss[i_hi]) / 2),
                textcoords="offset points", xytext=(-104, 14), ha="left",
                fontsize=7.8, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    ax.set_xlabel("Tasks, ordered cheapest → most expensive (by baseline cost)")
    ax.set_ylabel("Cost per task (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.set_xlim(1, n)
    d50 = st["p50"][2]
    ax.set_title(f"{tag} (n={n} paired)\nmedian {d50:+.1f}%, p95 {d95:+.1f}%",
                 color=INK, loc="left", pad=8, fontsize=9)
    ax.legend(loc="upper left")
    _clean(ax)
    _save(fig, f"fig_tail__{tag}")


def fig_scatter(base, sg, tag: str, st: dict) -> None:
    xs = [r["imputed_cost"] for r in base]
    ys = [r["imputed_cost"] for r in sg]

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    lo = min(min(xs), min(ys)) * 0.8
    hi = max(max(xs), max(ys)) * 1.25
    ax.plot([lo, hi], [lo, hi], color=BASELINE_AXIS, lw=1, zorder=1)
    ax.scatter(xs, ys, s=22, color=BLUE, alpha=0.55, linewidths=0.8,
               edgecolors=SURFACE, zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fmt = FuncFormatter(lambda v, _: f"${v:g}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Cost with built-in tools only (USD)")
    ax.set_ylabel("Cost with SkeletonGraph (USD)")
    ax.text(0.04, 0.90, "above line:\nSG cost more", transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top", linespacing=1.4)
    ax.text(0.96, 0.10, "below line:\nSG cheaper", transform=ax.transAxes,
            fontsize=7.5, color=BLUE, va="bottom", ha="right", linespacing=1.4)
    ax.set_title(f"{tag}: cheaper on {st['cheaper']}/{st['n']} tasks",
                 color=INK, loc="left", pad=8, fontsize=9)
    _clean(ax, ygrid=False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, f"fig_paired_scatter__{tag}")


def discover(a_base: str, a_sg: str) -> list[str]:
    """Tags with at least one paired task in both arms."""
    found = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if set(load_arm(d.name, a_base)) & set(load_arm(d.name, a_sg)):
            found.append(d.name)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", default=[],
                    help="run tag (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="every tag under eval/results/agent with both arms")
    ap.add_argument("--base-arm", default="native")
    ap.add_argument("--sg-arm", default="sg-fusion")
    ap.add_argument("--stats-only", action="store_true",
                    help="print the table, generate no figures")
    ap.add_argument("--include-failed", action="store_true",
                    help="also count runs that did not stop cleanly")
    ap.add_argument("--min-n", type=int, default=5,
                    help="skip tags with fewer paired tasks (default 5)")
    args = ap.parse_args()

    tags = list(args.tag)
    if args.all:
        tags = discover(args.base_arm, args.sg_arm)
    if not tags:
        sys.exit("give --tag <name> (repeatable) or --all")

    _style()
    rows = []
    for tag in tags:
        base, sg, rep = collect(tag, args.base_arm, args.sg_arm,
                                args.include_failed)
        print(f"\n{tag}")
        print(f"  {args.base_arm}: {rep['n_base']} runs, "
              f"{args.sg_arm}: {rep['n_sg']} runs -> "
              f"{rep['n_common']} paired")
        for lbl, ids in (("incomplete " + args.base_arm, rep["dropped_base"]),
                         ("incomplete " + args.sg_arm, rep["dropped_sg"]),
                         ("unpaired", rep["unpaired"])):
            if ids:
                print(f"  {lbl} ({len(ids)}): {', '.join(ids[:4])}"
                      + (f" +{len(ids) - 4} more" if len(ids) > 4 else ""))
        if rep["n_common"] < args.min_n:
            print(f"  SKIP — fewer than {args.min_n} paired tasks")
            continue

        st = stats(base, sg)
        rows.append((tag, st))
        print(f"  mean  ${st['mean_base']:.3f} -> ${st['mean_sg']:.3f}  "
              f"({st['total_delta_pct']:+.1f}% total)")
        for p in ("p50", "p90", "p95"):
            b, s, d = st[p]
            print(f"  {p}   ${b:.3f} -> ${s:.3f}  ({d:+.1f}%)")
        print(f"  turns {st['turns_base']:.1f} -> {st['turns_sg']:.1f}   "
              f"cheaper on {st['cheaper']}/{st['n']}")

        if not args.stats_only:
            fig_tail(base, sg, tag, st)
            fig_scatter(base, sg, tag, st)

    if len(rows) > 1:
        print(f"\n{'tag':30s} {'n':>4} {'mean Δ':>9} {'p50 Δ':>9} {'p95 Δ':>9}")
        for tag, st in rows:
            print(f"{tag:30s} {st['n']:>4} {st['total_delta_pct']:>8.1f}% "
                  f"{st['p50'][2]:>8.1f}% {st['p95'][2]:>8.1f}%")


if __name__ == "__main__":
    main()
