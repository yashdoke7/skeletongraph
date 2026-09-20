"""Figures for the v2 framing ("retrieval quality is not the bottleneck").

Reads the run JSONs under eval/results/agent/ directly — no hardcoded numbers —
so every figure regenerates from the artifact. The v1 figures (docs/paper/figures/)
belong to the earlier cost-tail framing and are NOT regenerated here; the two sets
are kept side by side until the rewrite settles.

    python docs/paper/figures_v2/make_figures.py

Writes <name>.pdf (for LaTeX) and <name>.png (for reading) per figure.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "eval" / "results" / "agent"
OUT = Path(__file__).resolve().parent

# Categorical slots 1-3 of the reference palette — the only three that clear the
# all-pairs CVD and normal-vision floors, so they are safe in every chart form
# here. Every bar carries a direct label (the aqua slot sits under 3:1 contrast
# on white, so the relief rule requires visible labels).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(tag: str) -> dict:
    out: dict = defaultdict(dict)
    for f in RUNS.joinpath(tag).glob("*.json"):
        if f.name == "summary.json":
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "arm" in d and "resolved" in d:
            out[d["arm"]][d["task_id"]] = d
    return out


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def bar_labels(ax, bars, fmt="{:.0f}", dy=1.0):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=7.5, color=INK2)


# ── fig 1 — the localisation ceiling ────────────────────────────────────────
def fig1():
    """Gold-file edit rate: how often the agent touched the file the fix lives in.

    This is the quantity retrieval can move. Where it is already ~95% there is
    nothing left for a retrieval system to win on pass@1.
    """
    setups = [
        ("ReAct loop\nnemotron", "nemotron_v4", "none", "fusion"),
        ("Codex CLI\ngpt-5.6-terra", "codex_v1", "codex-native", "codex-sg-plain"),
        ("Claude Code\nSWE-rebench", "claude_rebench_v1", "native", "sg-fusion"),
        ("Claude Code\nSWE-bench Verified", "claude_v7_rep2", "native", "sg-fusion-plain"),
    ]
    labels, base_pct, sg_pct, notes = [], [], [], []
    for label, tag, a_base, a_sg in setups:
        D = load(tag)
        vals = []
        for arm in (a_base, a_sg):
            rs = list(D[arm].values())
            ed = [r for r in rs if r.get("edited_gold_file")]
            vals.append(100 * len(ed) / len(rs))
            notes.append(100 * sum(1 for r in ed if r["resolved"]) / max(1, len(ed)))
        labels.append(label)
        base_pct.append(vals[0])
        sg_pct.append(vals[1])

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    x = range(len(labels))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], base_pct, w, color=BLUE,
                label="built-in search", zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], sg_pct, w, color=ORANGE,
                label="SkeletonGraph retrieval", zorder=3)
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("runs that edited the gold file (%)")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Retrieval only has headroom where localisation is still failing",
                 pad=22)
    # Above the plot: at 97% the bars reach the top of the axes, so any in-axes
    # legend placement collides with them.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)
    save(fig, "fig1_localisation_ceiling")


# ── fig 2 — the ladder, and what it buys ────────────────────────────────────
def fig2():
    """Left: standalone file-level recall@5. Right: pass@1 for the same backends
    inside the controlled ReAct harness. Separate axes, never one dual-axis plot.
    """
    res = ROOT / "eval" / "results"
    order = [("grep", "grep"), ("bm25", "bm25"), ("sg-rerank", "sg-rerank"),
             ("bm25-dense", "bm25+dense"), ("bm25-dense-sg", "bm25+dense+SG")]
    recalls, names = [], []
    for backend, pretty in order:
        p = res / f"paper_verified_{backend}_file.json"
        if not p.exists():
            continue
        recalls.append(100 * json.loads(p.read_text(encoding="utf-8"))["aggregate"]["recall@5"])
        names.append(pretty)

    D = load("nemotron_v4")
    arms = [("none", "none"), ("grep", "grep"), ("bm25", "bm25"), ("fusion", "SG fusion")]
    solves, anames = [], []
    for arm, pretty in arms:
        rs = list(D[arm].values())
        solves.append(100 * sum(1 for r in rs if r["resolved"]) / len(rs))
        anames.append(pretty)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.9))
    b = ax1.barh(names, recalls, 0.6, color=BLUE, zorder=3)
    for r, n in zip(recalls, names):
        ax1.text(r + 1.5, n, f"{r:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("file recall@5 (%)")
    ax1.set_title("Retrieval quality spans a 3× range")
    ax1.yaxis.grid(False)
    ax1.set_axisbelow(True)

    b2 = ax2.bar(anames, solves, 0.55, color=AQUA, zorder=3)
    bar_labels(ax2, b2, fmt="{:.0f}%", dy=0.6)
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("pass@1 (%)")
    ax2.set_title("…and pass@1 barely moves")
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig2_retrieval_ladder")


# ── fig 3 — cost saving is a property of the harness ────────────────────────
def fig3():
    """Context carried per turn by the built-in agent, and what SG's retrieval
    does to total cost in that same harness. The saving tracks the waste.
    """
    setups = [("Codex CLI", "codex_v1", "codex-native", "codex-sg-plain"),
              ("Claude Code", "claude_v7_rep2", "native", "sg-fusion-plain")]
    names, ctx_base, ctx_sg, delta = [], [], [], []
    for label, tag, a_base, a_sg in setups:
        D = load(tag)
        common = sorted(set(D[a_base]) & set(D[a_sg]))
        cpt = lambda arm: st.mean(
            (D[arm][t].get("total_input_tokens") or 0) / max(1, D[arm][t].get("n_turns") or 1)
            for t in common)
        cost = lambda arm: st.mean(D[arm][t].get("imputed_cost") or 0 for t in common)
        names.append(label)
        ctx_base.append(cpt(a_base) / 1000)
        ctx_sg.append(cpt(a_sg) / 1000)
        delta.append(100 * (cost(a_sg) - cost(a_base)) / cost(a_base))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    x = range(len(names))
    w = 0.32
    b1 = ax1.bar([i - w / 2 for i in x], ctx_base, w, color=BLUE,
                 label="built-in search", zorder=3)
    b2 = ax1.bar([i + w / 2 for i in x], ctx_sg, w, color=ORANGE,
                 label="SG retrieval", zorder=3)
    bar_labels(ax1, b1, fmt="{:.0f}k", dy=0.8)
    bar_labels(ax1, b2, fmt="{:.0f}k", dy=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("context carried per turn (1000s of tokens)")
    ax1.set_ylim(0, 48)
    ax1.set_title("How much context the harness re-sends")
    ax1.legend(loc="upper left")
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)

    cols = [AQUA if d < 0 else ORANGE for d in delta]
    b3 = ax2.bar(names, delta, 0.5, color=cols, zorder=3)
    for bb, d in zip(b3, delta):
        ax2.text(bb.get_x() + bb.get_width() / 2, d + (2 if d > 0 else -5),
                 f"{d:+.0f}%", ha="center", fontsize=8, color=INK2)
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_ylabel("change in mean cost with SG (%)")
    ax2.set_ylim(-75, 25)
    ax2.set_title("What SG's retrieval saves there")
    ax2.xaxis.grid(False)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig3_cost_is_a_harness_property")


# ── fig 4 — the measurement noise floor ─────────────────────────────────────
def fig4():
    """Two runs of the SAME arm at the SAME config, two months apart: how many
    per-task verdicts flip, and how far per-task cost moves.
    """
    pairs = [("native", "built-in search"), ("sg-fusion", "SG (shipped config)")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.9))

    names, ft, tf = [], [], []
    ratios_all = {}
    for arm, pretty in pairs:
        A, B = load("claude_v7")[arm], load("claude_v7_rep2")[arm]
        c = sorted(set(A) & set(B))
        names.append(pretty)
        ft.append(sum(1 for t in c if not A[t]["resolved"] and B[t]["resolved"]))
        tf.append(sum(1 for t in c if A[t]["resolved"] and not B[t]["resolved"]))
        ratios_all[pretty] = [(B[t].get("imputed_cost") or 0) /
                              max(1e-9, A[t].get("imputed_cost") or 0) for t in c]

    x = range(len(names))
    w = 0.32
    b1 = ax1.bar([i - w / 2 for i in x], ft, w, color=BLUE,
                 label="unsolved → solved", zorder=3)
    b2 = ax1.bar([i + w / 2 for i in x], tf, w, color=ORANGE,
                 label="solved → unsolved", zorder=3)
    bar_labels(ax1, b1, dy=0.2)
    bar_labels(ax1, b2, dy=0.2)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("tasks out of 100 whose verdict flipped")
    ax1.set_ylim(0, 15)
    ax1.set_title("Identical config, re-run: verdicts move")
    ax1.legend(loc="upper left")
    ax1.xaxis.grid(False)
    ax1.set_axisbelow(True)

    for (pretty, rs), col in zip(ratios_all.items(), (BLUE, ORANGE)):
        rs = sorted(min(r, 3.0) for r in rs)
        ax2.plot(rs, [i / len(rs) * 100 for i in range(len(rs))],
                 color=col, linewidth=2, label=pretty)
    ax2.axvline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.set_xlabel("per-task cost, re-run ÷ original")
    ax2.set_ylabel("cumulative share of tasks (%)")
    ax2.set_xlim(0, 3)
    ax2.set_title("…and so does cost, on both arms alike")
    ax2.legend(loc="lower right")
    ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig4_measurement_noise_floor")


# ── fig 5 — the two paired 100-task sets ────────────────────────────────────
def fig5():
    """pass@1, mean cost and first-search rank-1 rate for the three arms in each
    harness, on the same 100 tasks.
    """
    panels = [("Claude Code · SWE-bench Verified", "claude_v7_rep2",
               [("native", "built-in"), ("sg-fusion", "SG shipped"),
                ("sg-fusion-plain", "SG plain")]),
              ("Codex CLI · SWE-bench Verified", "codex_v1",
               [("codex-native", "built-in"), ("codex-sg-fusion", "SG shipped"),
                ("codex-sg-plain", "SG plain")])]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for row, (title, tag, arms) in enumerate(panels):
        D = load(tag)
        common = sorted(set.intersection(*[set(D[a]) for a, _ in arms]))
        names = [p for _, p in arms]
        solved = [sum(1 for t in common if D[a][t]["resolved"]) for a, _ in arms]
        cost = [st.mean(D[a][t].get("imputed_cost") or 0 for t in common) for a, _ in arms]
        rank1 = [sum(1 for t in common if D[a][t].get("retrieval_rank") == 1) for a, _ in arms]
        for col, (vals, lab, fmt, col_) in enumerate([
                (solved, "pass@1 (of 100)", "{:.0f}", BLUE),
                (cost, "mean cost (USD)", "${:.3f}", ORANGE),
                (rank1, "gold file ranked 1st", "{:.0f}", AQUA)]):
            ax = axes[row][col]
            b = ax.bar(names, vals, 0.55, color=col_, zorder=3)
            top = max(vals) * 1.28
            for bb, v in zip(b, vals):
                ax.text(bb.get_x() + bb.get_width() / 2, v + top * 0.03,
                        fmt.format(v), ha="center", fontsize=7.5, color=INK2)
            ax.set_ylim(0, top)
            ax.set_ylabel(lab if col else f"{lab}", fontsize=8)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=7.5)
            if col == 1:
                ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    fig.tight_layout()
    save(fig, "fig5_paired_arms")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {OUT}")
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn()


# ── fig 6 — the same retriever, eight harness operating points ──────────────
def fig6():
    """The headline. One retrieval system, measured against every harness we ran.

    Left: its effect on tokens against how many tokens the harness spends on its
    own. Right: its effect on pass@1 against how often the harness reaches the
    gold file on its own. Neither effect is a property of the retriever.
    """
    import math
    SET = [("ReAct·nemotron v2", "nemotron_v2", "none", "nemotron_v2", "sg-rerank"),
           ("ReAct·nemotron v4", "nemotron_v4", "none", "nemotron_v4", "fusion"),
           ("CC 2.1.206-211", "claude_v7", "native", "claude_v7", "sg-fusion"),
           ("CC 2.1.211-14 (rebench)", "claude_rebench_v1", "native",
            "claude_rebench_v1", "sg-fusion"),
           ("CC 2.1.214 (prose)", "claude_rebench_prose_v1", "native",
            "claude_rebench_prose_v1", "sg-fusion"),
           ("CC 2.1.274", "claude_v7_rep2", "native", "claude_v7_rep2", "sg-fusion"),
           ("CC 2.1.278", "claude_v8", "native", "claude_v7_rep2", "sg-fusion-plain"),
           ("Codex 0.155.0", "codex_v1", "codex-native", "codex_v1", "codex-sg-plain")]

    def tok(d):
        v = d.get("total_input_tokens")
        if isinstance(v, (int, float)) and v:
            return v
        return ((d.get("billed_input") or 0) + (d.get("cached_input") or 0)
                + (d.get("cache_creation_input") or 0))

    pts = []
    for lab, tb, ab, tg, ag in SET:
        A, B = load(tb)[ab], load(tg)[ag]
        c = sorted(set(A) & set(B))
        m0 = st.mean(tok(A[t]) for t in c)
        m1 = st.mean(tok(B[t]) for t in c)
        reach = 100 * sum(1 for t in c if A[t].get("edited_gold_file")) / len(c)
        dsolve = sum(B[t]["resolved"] for t in c) - sum(A[t]["resolved"] for t in c)
        pts.append((lab, m0, 100 * (m1 - m0) / m0, reach, dsolve * 100 / len(c)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, xs, ys, xlab, ylab, title in [
            (ax1, [p[1] / 1000 for p in pts], [p[2] for p in pts],
             "tokens the harness spends on its own (1000s)",
             "effect on tokens (%)", "What retrieval costs, or saves"),
            (ax2, [p[3] for p in pts], [p[4] for p in pts],
             "runs reaching the gold file on its own (%)",
             "effect on pass@1 (points)", "What retrieval buys in accuracy")]:
        ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.scatter(xs, ys, s=52, color=BLUE, zorder=3,
                   edgecolors="white", linewidths=1.2)
        for (lab, *_), x, y in zip(pts, xs, ys):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=6.2, color=INK2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_axisbelow(True)
    ax1.set_xscale("log")
    ax1.set_ylim(-85, 88)
    ax2.set_xlim(58, 106)
    ax2.set_ylim(-16, 16)
    fig.tight_layout()
    save(fig, "fig6_effect_is_a_harness_property")


# ── fig 7 — the corroboration gap ───────────────────────────────────────────
def fig7():
    """Why faster localisation does not become a faster fix.

    Retrieval delivers the gold file several turns earlier, and the agent edits
    barely earlier: the saved turns are re-absorbed into corroboration. The turn
    of the first edit is near-invariant across every backend.
    """
    D = load("nemotron_v4")
    arms = [("none", "no retrieval"), ("graphify", "graphify"), ("grep", "grep"),
            ("bm25", "bm25"), ("fusion", "SG fusion")]
    names, reads, edits = [], [], []
    for arm, pretty in arms:
        rs = [r for r in D.get(arm, {}).values()
              if isinstance(r.get("time_to_first_gold_read_turn"), (int, float))
              and isinstance(r.get("time_to_first_edit_turn"), (int, float))]
        if not rs:
            continue
        names.append(pretty)
        reads.append(st.mean(r["time_to_first_gold_read_turn"] for r in rs))
        edits.append(st.mean(r["time_to_first_edit_turn"] for r in rs))

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = list(range(len(names)))
    for i, (a, b) in enumerate(zip(reads, edits)):
        ax.plot([a, b], [i, i], color="#cfcec9", linewidth=5,
                solid_capstyle="round", zorder=2)
        ax.text((a + b) / 2, i + 0.26, f"{b - a:.1f} turns", ha="center",
                fontsize=7, color=INK2)
    ax.scatter(reads, y, s=58, color=BLUE, zorder=3, label="first reads the gold file",
               edgecolors="white", linewidths=1.2)
    ax.scatter(edits, y, s=58, color=ORANGE, zorder=3, label="first edits anything",
               edgecolors="white", linewidths=1.2)
    for v, i in zip(reads, y):
        ax.text(v, i - 0.30, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    for v, i in zip(edits, y):
        ax.text(v, i - 0.30, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.25)
    ax.set_xlim(0, 16)
    ax.set_xlabel("turn number")
    ax.set_title("Retrieval arrives sooner; the edit does not", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    save(fig, "fig7_corroboration_gap")
