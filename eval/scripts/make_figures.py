"""Research-grade paper figures from a run tag's summary.json (_rows).

    python -m eval.scripts.make_figures --tag nemotron_v2

Design rules (learned the hard way — the old one-plot-for-16-arms was an
unreadable blob):
  * Each figure makes ONE point. Main-table arms only in the headline figures;
    the SG-family ablations get their OWN figure so nothing is clustered by
    accident.
  * The aider token/cost outlier (~6x) gets a BROKEN axis, not a squished plot.
  * Direct labels next to points, not a 16-row legend.
  * Distinct colors (no two greys), generous whitespace, despined axes.

Writes PNG (200 dpi) + PDF to <run-dir>/figures/:
  fig1_pareto      — solve rate vs token cost (8 main arms, broken x for aider)
  fig2_retrieval   — file vs function recall (the "file != function" gap)
  fig3_cost        — agent vs one-time LLM index cost (graphify's hidden build)
  fig4_decoupling  — pass@1 is flat while retrieval quality varies (contamination)
  fig5_ablation    — SG-family variants: structure-as-reranker wins function recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
# Hero = sg-rerank (terracotta). SG core = navy. Baselines = cool blues/greys.
# Competitors = purple/amber. Every arm distinct — no grey-on-grey collisions.
C = {
    "sg":            "#1D3557",  # deep navy  (lean core)
    "sg-rerank":     "#E63946",  # terracotta (the method / hero)
    "bm25":          "#457B9D",  # steel blue
    "grep":          "#5C8A6F",  # muted green
    "hybrid":        "#8D99AE",  # slate
    "none":          "#B8C0C8",  # light grey (the no-retrieval baseline)
    "cbmem":         "#6A4C93",  # purple
    "aider":         "#BC6C25",  # amber
    "graphify":      "#9C6644",  # brown
    # SG-family ablations
    "sg-chain":          "#2A9D8F",
    "sg-embed":          "#8AB17D",
    "sg-hybrid-fusion":  "#E9C46A",
    "sg-dense-rerank":   "#F4A261",
    "sg-keyword-dense":  "#E76F51",
    "sg-seed":           "#A26769",
    "summary-dense":     "#6D6875",
}
# arms in the main comparison table (headline figures use ONLY these)
MAIN = ["sg", "sg-rerank", "bm25", "grep", "hybrid", "none", "cbmem", "aider"]
# SG-family index-use ablations (their own figure)
ABL = ["sg-rerank", "sg", "sg-chain", "sg-embed", "sg-hybrid-fusion",
       "sg-dense-rerank", "sg-keyword-dense", "sg-seed", "summary-dense"]


def _c(arm):  return C.get(arm, "#999999")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _load(run_dir: Path) -> dict:
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = s.get("_rows")
    if not rows:
        raise SystemExit("summary.json has no _rows — re-run aggregate.")
    return {r["key"]: r for r in rows}


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / (name + '.png')}")


# ── fig 1 — Pareto with a broken x-axis for the aider outlier ─────────────────
def fig_pareto(R, out):
    arms = [a for a in MAIN if a in R and R[a].get("pass1") is not None]
    xs = {a: R[a]["intok"] / 1000.0 for a in arms}
    ys = {a: R[a]["pass1"] * 100 for a in arms}
    cluster = [a for a in arms if xs[a] < 600]
    outliers = [a for a in arms if xs[a] >= 600]

    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True, figsize=(8.2, 4.8),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.06})
    for ax in (axL, axR):
        _style(ax)

    # left panel: the real comparison (zoom to the cluster). Per-arm label
    # offsets so nearby points (none/hybrid at ~245-250k, 44%) don't collide.
    lo = min(xs[a] for a in cluster) - 18
    hi = max(xs[a] for a in cluster) + 40
    axL.set_xlim(lo, hi)
    OFF = {  # dx, dy (pts), ha — none/hybrid/cbmem nearly coincide; fan them out
        "sg": (14, 7, "left"), "sg-rerank": (14, -16, "left"),
        "cbmem": (14, 7, "left"), "hybrid": (-16, 13, "center"),
        "none": (0, -22, "center"), "grep": (13, 1, "left"),
        "bm25": (13, 1, "left"),
    }
    for a in cluster:
        axL.scatter(xs[a], ys[a], s=320, c=_c(a), edgecolor="white",
                    linewidth=1.6, zorder=3)
        dx, dy, ha = OFF.get(a, (13, 4, "left"))
        hero = a in ("sg", "sg-rerank")
        axL.annotate(a, (xs[a], ys[a]), textcoords="offset points",
                     xytext=(dx, dy), fontsize=10.5,
                     fontweight="bold" if hero else "normal",
                     color=_c(a) if hero else "#333", ha=ha)

    # right panel: the off-scale outlier(s)
    if outliers:
        omin = min(xs[a] for a in outliers)
        axR.set_xlim(omin - 60, max(xs[a] for a in outliers) + 60)
        for a in outliers:
            axR.scatter(xs[a], ys[a], s=320, c=_c(a), edgecolor="white",
                        linewidth=1.6, zorder=3)
            axR.annotate(f"{a}\n{xs[a]:.0f}k tok", (xs[a], ys[a]),
                         textcoords="offset points", xytext=(0, 16),
                         ha="center", fontsize=10, color=_c(a), fontweight="bold")
        axR.set_xticks([round(min(xs[a] for a in outliers), -2)])
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False)

    # diagonal break marks between the panels
    d = .012
    kw = dict(transform=axL.transAxes, color="#666", clip_on=False, lw=1.1)
    axL.plot((1 - d, 1 + d), (-d, d), **kw); axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw = dict(transform=axR.transAxes, color="#666", clip_on=False, lw=1.1)
    axR.plot((-d * 4, d * 4), (-d, d), **kw); axR.plot((-d * 4, d * 4), (1 - d, 1 + d), **kw)

    axL.set_ylabel("Resolve rate  pass@1 (%)", fontsize=11.5)
    fig.suptitle("Solve rate vs. token cost — the SG family is cheapest at equal solve rate",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, 0.005, "Input tokens per task (thousands)  —  cheaper is left",
             ha="center", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, out, "fig1_pareto")


# ── fig 2 — file vs function recall (the gap) ─────────────────────────────────
def fig_retrieval(R, out):
    arms = [a for a in MAIN if a in R and R[a].get("reccum")]
    arms = sorted(arms, key=lambda a: -(R[a]["funcr"] or 0))
    x = np.arange(len(arms)); w = 0.38
    file_r = [R[a]["reccum"] or 0 for a in arms]
    func_r = [R[a]["funcr"] or 0 for a in arms]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style(ax)
    b1 = ax.bar(x - w / 2, file_r, w, color=[_c(a) for a in arms], alpha=0.45,
                edgecolor="white", zorder=3, label="file recall (cumulative)")
    b2 = ax.bar(x + w / 2, func_r, w, color=[_c(a) for a in arms],
                edgecolor="white", zorder=3, label="function recall@10")
    for xi, f in zip(x + w / 2, func_r):
        ax.text(xi, f + 0.015, f"{f:.2f}", ha="center", fontsize=9, color="#333")
    for xi, f in zip(x - w / 2, file_r):
        ax.text(xi, f + 0.015, f"{f:.2f}", ha="center", fontsize=8.5, color="#999")
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("Recall", fontsize=11.5)
    ax.set_ylim(0, max(file_r) + 0.16)
    ax.set_title("Most arms find the right file but not the right function\n"
                 "(faded = file recall · solid = function recall@10)",
                 fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    _save(fig, out, "fig2_retrieval")


# ── fig 3 — agent cost vs one-time LLM index/build cost ───────────────────────
def fig_cost(R, out):
    arms = [a for a in MAIN if a in R]
    if "graphify" in R and "graphify" not in arms:
        arms = arms + ["graphify"]
    arms = sorted(arms, key=lambda a: (R[a].get("total_cost") or R[a]["cost"]))
    agent = [R[a]["cost"] for a in arms]
    index = [R[a].get("idx_cost") or 0 for a in arms]
    has_index = any(i > 0 for i in index)   # graphify present & built?

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style(ax)
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.9, zorder=0); ax.grid(axis="y", visible=False)
    y = np.arange(len(arms))
    ax.barh(y, agent, color=[_c(a) for a in arms], edgecolor="white", zorder=3,
            label="agent-loop cost (every arm pays)")
    if has_index:
        ax.barh(y, index, left=agent, color="#222", alpha=0.85, edgecolor="white",
                zorder=3, hatch="///",
                label="one-time LLM index build (graph competitors)")
    for yi, a, ag, ix in zip(y, arms, agent, index):
        tot = ag + ix
        ax.text(tot + max(agent) * 0.012, yi, f"${tot:.3f}", va="center", fontsize=9,
                fontweight="bold" if ix else "normal", color="#222" if ix else "#555")
    ax.set_yticks(y); ax.set_yticklabels(arms, fontsize=10)
    ax.set_xlabel("Cost per task ($)", fontsize=11.5)
    if has_index:
        ax.set_title("graphify's true cost is the per-repo LLM graph build,\n"
                     "hidden by token-only benchmarks",
                     fontsize=12, fontweight="bold", pad=12)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    else:
        ax.set_title("Cost per task — the SG family is the cheapest of every arm",
                     fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlim(0, max(a + i for a, i in zip(agent, index)) * 1.18)
    fig.tight_layout()
    _save(fig, out, "fig3_cost")


# ── fig 4 — pass@1 flat while retrieval quality varies (contamination) ────────
def fig_decoupling(R, out):
    arms = [a for a in MAIN if a in R and R[a].get("pass1") is not None]
    arms = sorted(arms, key=lambda a: -(R[a]["funcr"] or 0))
    x = np.arange(len(arms))
    p1 = [R[a]["pass1"] * 100 for a in arms]
    fr = [(R[a]["funcr"] or 0) * 100 for a in arms]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style(ax)
    ax.bar(x, fr, 0.6, color=[_c(a) for a in arms], alpha=0.85, edgecolor="white",
           zorder=3, label="function recall@10 (%)  — VARIES")
    # pass@1 as a near-flat line on top
    ax.plot(x, p1, "o-", color="#222", lw=2, markersize=7, zorder=5,
            label="pass@1 (%)  — FLAT (~44%)")
    none_p1 = R["none"]["pass1"] * 100 if "none" in R else None
    if none_p1:
        ax.axhline(none_p1, color="#888", ls="--", lw=1, zorder=2)
        ax.text(0.05, none_p1 + 0.6, f"no-retrieval baseline ({none_p1:.0f}%)",
                ha="left", fontsize=9, color="#888", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=22, ha="right", fontsize=10)
    ax.set_ylabel("percent", fontsize=11.5)
    ax.set_ylim(0, max(max(p1), max(fr)) + 9)
    ax.set_title("Retrieval quality varies 2x across arms, but solve rate does not move\n"
                 "(evidence of benchmark contamination)",
                 fontsize=12.5, fontweight="bold", pad=12)
    # legend in the empty right region (grep/hybrid/none/aider have 0 func recall)
    ax.legend(frameon=False, fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.6))
    fig.tight_layout()
    _save(fig, out, "fig4_decoupling")


# ── fig 5 — SG-family ablation: function recall ───────────────────────────────
def fig_ablation(R, out):
    arms = [a for a in ABL if a in R]
    arms = sorted(arms, key=lambda a: -(R[a]["funcr"] or 0))
    x = np.arange(len(arms))
    fr = [(R[a]["funcr"] or 0) for a in arms]
    colors = ["#E63946" if a == "sg-rerank" else _c(a) for a in arms]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _style(ax)
    bars = ax.bar(x, fr, 0.62, color=colors, edgecolor="white", zorder=3)
    for xi, f in zip(x, fr):
        ax.text(xi, f + 0.008, f"{f:.2f}", ha="center", fontsize=9, color="#333")
    # group brackets
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=24, ha="right", fontsize=9.5)
    ax.set_ylabel("function recall@10", fontsize=11.5)
    ax.set_ylim(0, max(fr) + 0.07)
    ax.set_title("Ablation: structure as a reranker over lexical recall beats every\n"
                 "other use of the same index (graph traversal, dense, seeding)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.text(0.5, max(fr) + 0.045, "sg-rerank = the method", color="#E63946",
            fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "fig5_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--run-dir", default=None, type=Path)
    args = ap.parse_args()
    run_dir = args.run_dir or (Path("eval/results/agent") / args.tag)
    R = _load(run_dir)
    out = run_dir / "figures"
    plt.rcParams.update({"font.size": 10.5, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.titlesize": 12})
    fig_pareto(R, out)
    fig_retrieval(R, out)
    fig_cost(R, out)
    fig_decoupling(R, out)
    fig_ablation(R, out)
    print(f"Figures in {out}")


if __name__ == "__main__":
    main()
