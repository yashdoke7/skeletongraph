"""Figures for the v2 paper: the funnel, the per-setting cost effect, the release change.

    python -m eval.scripts.paper_v2_analysis --json eval/results_summary.json
    python -m eval.scripts.make_v2_paper_figures

Reads the results file written by paper_v2_analysis (no number is hardcoded here)
and writes the figures into SG_FIG_OUT (default docs/assets, PNG). SG_NUMBERS
overrides the results file, SG_LABEL the tool's display name, and SG_FIG_FORMATS
the formats written (e.g. "pdf,png").

Colour: slots 1-2 of the validated categorical palette (#2a78d6, #eb6834), which
clear the all-pairs CVD and normal-vision floors; every series is also direct-
labelled or legended, so identity never rests on colour alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMBERS = Path(os.environ.get("SG_NUMBERS", "eval/results_summary.json"))
OUT = Path(os.environ.get("SG_FIG_OUT", "docs/assets"))
TOOL = os.environ.get("SG_LABEL", "SkeletonGraph")
FORMATS = [f.strip() for f in os.environ.get("SG_FIG_FORMATS", "png").split(",") if f.strip()]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / name}.{{{','.join(FORMATS)}}}")


# ── the funnel ──────────────────────────────────────────────────────────────
PANELS = [("ReAct loop (v4)", "ReAct loop"),
          ("Claude Code 2.1.206-211", "Claude exploratory (July)"),
          ("Claude Code 2.1.274", "Claude exploratory (Sept.)"),
          ("Claude Code 2.1.278", "Claude lean"),
          ("Claude Code, SWE-rebench", "Claude exploratory, Rebench"),
          ("Codex CLI 0.155.0", "Codex CLI")]
STAGES = ["1st search\nhit", "saw the\ncode", "edited\nthe file", "made a\npatch", "solved"]


def fig_funnel(R):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), sharey=True)
    for ax, (key, title) in zip(axes.flat, PANELS):
        row = R["funnel"][key]
        a = [s[1] for s in row["stages"]]
        b = [s[2] for s in row["stages"]]
        x = range(len(STAGES))
        ax.plot(x, a, color=BLUE, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3,
                label="without the retriever")
        ax.plot(x, b, color=ORANGE, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3, label=f"with {TOOL}")
        # label the two ends of the funnel, which are the claim
        for i in (0, len(STAGES) - 1):
            hi, lo = (a[i], b[i]) if a[i] >= b[i] else (b[i], a[i])
            ax.annotate(f"{a[i]:.0f}", (i, a[i]), xytext=(0, 7 if a[i] >= b[i] else -11),
                        textcoords="offset points", ha="center", fontsize=6.5, color=BLUE)
            ax.annotate(f"{b[i]:.0f}", (i, b[i]), xytext=(0, 7 if b[i] > a[i] else -11),
                        textcoords="offset points", ha="center", fontsize=6.5, color=ORANGE)
        ax.set_title(f"{title}  (n={row['n']})", fontsize=7.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(STAGES, fontsize=5.8)
        ax.set_ylim(-10, 114)
        ax.set_xlim(-0.35, len(STAGES) - 0.65)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("tasks (%)")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig_funnel")


# ── the same retriever's token effect in each setting ───────────────────────
SETTING_ROWS = [("Claude Code, SWE-rebench prose", "Claude exploratory, Rebench prose"),
                ("Claude Code, SWE-rebench", "Claude exploratory, Rebench"),
                ("Claude Code 2.1.206-211", "Claude exploratory (July)"),
                ("Claude Code 2.1.274", "Claude exploratory (Sept.)"),
                ("ReAct loop (v4)", "ReAct loop (vs. no search)"),
                ("ReAct loop (v2)", "ReAct loop, earlier run"),
                ("Codex CLI 0.155.0", "Codex CLI 0.155.0"),
                ("Claude Code 2.1.278", "Claude lean")]


def fig_settings(R):
    rows = [(lab, R["settings"][k]) for k, lab in SETTING_ROWS]
    rows.sort(key=lambda r: r[1]["base_tokens"])
    fig, ax = plt.subplots(figsize=(7.1, 3.3))
    for i, (lab, v) in enumerate(rows):
        a = v["base_tokens"] / 1000
        b = (v["base_tokens"] + v["tokens_delta"]) / 1000
        ax.plot([a, b], [i, i], color="#cfcec9", linewidth=4, solid_capstyle="round", zorder=2)
        ax.scatter([a], [i], s=40, color=BLUE, zorder=3, edgecolors="white", linewidths=1)
        ax.scatter([b], [i], s=40, color=ORANGE, zorder=3, edgecolors="white", linewidths=1)
        sign = "−" if v["tokens_delta"] < 0 else "+"
        txt = (f"{v['tokens_pct']:+.0f}%  ·  {sign}{abs(v['tokens_delta']) / 1000:.0f}k tokens  ·  "
               f"{v['turns_delta']:+.1f} turns").replace("-", "−")
        ax.text(max(a, b) * 1.08, i, txt, va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(80, 7000)
    ax.set_xticks([100, 200, 500, 1000])
    ax.set_xticklabels(["100k", "200k", "500k", "1M"])
    ax.set_xlabel("input tokens per task (log scale)")
    ax.yaxis.grid(False)
    ax.scatter([], [], s=40, color=BLUE, label="without the retriever")
    ax.scatter([], [], s=40, color=ORANGE, label=f"with {TOOL}")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_axisbelow(True)
    save(fig, "fig_settings")


# ── what changed in the agent between two releases ──────────────────────────
METRICS = [("turns", "turns with\na tool call"), ("before_edit", "turns before\nfirst edit"),
           ("after_edit", "turns after\nfirst edit"), ("reads", "file reads"),
           ("code_runs", "code / test\nruns")]


def fig_release(R):
    rel, add = R["release"]["rows"], R["addition"]["rows"]
    groups = [("Exploratory, built-in", [rel[k][0] for k, _ in METRICS], "#9ec3ee"),
              ("Lean, built-in", [rel[k][1] for k, _ in METRICS], BLUE),
              (f"Lean, + {TOOL}", [add[k][1] for k, _ in METRICS], ORANGE)]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.1, 2.35))
    for j, (ax, (_, lab)) in enumerate(zip(axes, METRICS)):
        vals = [g[1][j] for g in groups]
        bars = ax.bar(range(3), vals, 0.72, color=[g[2] for g in groups], zorder=3)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + max(vals) * 0.03, f"{v:.1f}",
                    ha="center", fontsize=6.5, color=INK2)
        ax.set_title(lab, fontsize=7.2)
        ax.set_xticks([])
        ax.set_ylim(0, max(vals) * 1.22)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[2]) for g in groups]
    fig.legend(handles, [g[0] for g in groups], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.1), fontsize=7)
    fig.tight_layout()
    save(fig, "fig_release")


def main():
    style()
    R = json.loads(NUMBERS.read_text(encoding="utf-8"))
    fig_funnel(R)
    fig_settings(R)
    fig_release(R)


if __name__ == "__main__":
    main()
