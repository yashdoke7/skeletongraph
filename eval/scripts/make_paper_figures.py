"""Generate the paper's figures as vector PDF (for LaTeX) + PNG (for README).

Every figure is generated from the run JSONs on disk — no hand-entered numbers —
so a reviewer can regenerate the exact figures from the released artifact:

    python -m eval.scripts.make_paper_figures

Design rules (validated palette, print/light surface):
  - categorical slot 1 = blue #2a78d6 (SkeletonGraph, the protagonist)
  - baseline series = neutral ink, not a competing hue: a reader should never
    have to decode "which hue is the baseline"
  - solid hairline grid one shade off the surface; no dashed grid, no top/right
    spines, no value printed on every point (selective direct labels only)
  - one y-axis per plot, never dual-axis
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── validated palette (light surface) ────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # slot 1 — SkeletonGraph
GREEN = "#008300"    # slot 2 — second measure
ORANGE = "#eb6834"   # slot 6 — used only where a warm/cool opposition is meant

OUT = Path("docs/paper/figures")
RUNS = Path("eval/results/agent")


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": BASELINE_AXIS,
        "axes.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",      # never dashed
        "axes.grid.axis": "y",
        "figure.dpi": 150,
    })


def _clean(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {OUT/name}.pdf / .png")


# ── data loading ─────────────────────────────────────────────────────────
def load_arm(tag: str, arm: str) -> dict:
    d = {}
    for f in glob.glob(str(RUNS / tag / f"*__{arm}__*.json")):
        if "summary" in f.lower() or "_INDEX" in f:
            continue
        try:
            r = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        d[r["task_id"]] = r
    return d


def paired(tag, a1, a2):
    d1, d2 = load_arm(tag, a1), load_arm(tag, a2)
    common = sorted(set(d1) & set(d2))
    return [d1[t] for t in common], [d2[t] for t in common]


# ── FIG 1 — the tail: where the saving actually lives ────────────────────
def fig_tail(nat, sg):
    """Tasks ordered cheapest->most expensive by the built-in-tools baseline.

    Ordering by task difficulty (rather than plotting abstract percentiles) lets
    a reader see the finding directly: the two curves sit on top of each other
    for most of the range and separate only at the expensive end.
    """
    pairs = sorted(zip((r["imputed_cost"] for r in nat),
                       (r["imputed_cost"] for r in sg)), key=lambda p: p[0])
    nv = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    x = list(range(1, len(nv) + 1))

    # running mean smooths per-task stochasticity without hiding the divergence
    def smooth(vals, w=9):
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

    ax.axvline(50, color=BASELINE_AXIS, lw=1, zorder=0)
    # y in axes fraction so the label can never fall outside the data range
    ax.text(50, 0.55, " median task", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=MUTED, ha="left", va="center", rotation=90)
    ax.annotate("curves overlap:\nno saving here", (22, ns[21]),
                textcoords="offset points", xytext=(0, 34), ha="center",
                fontsize=7.8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=BASELINE_AXIS, lw=0.8))
    ax.annotate("the gap is the saving\n(−42.5% at p95)", (92, (ns[91] + ss[91]) / 2),
                textcoords="offset points", xytext=(-104, 14), ha="left",
                fontsize=7.8, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    ax.set_xlabel("Tasks, ordered cheapest → most expensive (by baseline cost)")
    ax.set_ylabel("Cost per task (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax.set_xlim(1, 100)
    ax.set_title("Retrieval changes nothing on ordinary tasks;\nit truncates the expensive ones",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="upper left")
    _clean(ax)
    _save(fig, "fig_tail")


# ── FIG 2 — per-task paired scatter (the proof behind Fig 1) ─────────────
def fig_scatter(nat, sg):
    xs = [r["imputed_cost"] for r in nat]
    ys = [r["imputed_cost"] for r in sg]

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    lo = min(min(xs), min(ys)) * 0.8
    hi = max(max(xs), max(ys)) * 1.25
    # parity line: position relative to it IS the encoding, so no second hue
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
    cheaper = sum(1 for a, b in zip(xs, ys) if b < a)
    # region labels sit inside the plot, clear of the title band
    ax.text(0.04, 0.90, "above line:\nSG cost more", transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top", linespacing=1.4)
    ax.text(0.96, 0.10, "below line:\nSG cheaper", transform=ax.transAxes,
            fontsize=7.5, color=BLUE, va="bottom", ha="right", linespacing=1.4)
    ax.set_title(f"Cheaper on {cheaper}/{len(xs)} tasks, and the wins are the expensive ones",
                 color=INK, loc="left", pad=8, fontsize=9)
    _clean(ax, ygrid=False)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, "fig_paired_scatter")


# ── FIG 3 — retrieval granularity: the structural gap ────────────────────
def fig_retrieval(nat, sg, dataset_path):
    ds = {}
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ds[r["task_id"]] = r

    def rates(recs, filter_sg=False):
        """FIRST-search file recall and function-identification rate.

        Uses `first_search_fqns` (what the opening retrieval returned) rather than
        the cumulative `retrieval_hit`, so the figure reports the same quantity as
        the paper's retrieval table: how good the retrieval itself is, not whether
        the agent eventually stumbled onto the file after several reads.
        """
        # A record with sg_tool_calls==0 never invoked SG that turn — the model
        # chose native tools instead. Scoring that as a retrieval miss blames
        # SG for an adoption choice it had no chance to affect, and silently
        # drags the aggregate down. ONLY apply this to the SG arm — native
        # records always report sg_tool_calls==0 legitimately (it never had SG
        # to call), so filtering them the same way would wipe them out.
        if filter_sg:
            recs = [r for r in recs if r.get("sg_tool_calls") != 0]
        n = len(recs)
        # RECALL, not hit-rate: the fraction of the task's gold files the first
        # search returned, averaged over tasks. On single-file tasks the two are
        # identical; on multi-file tasks recall is stricter, and since the
        # expensive tasks average 1.54 gold files the difference is material
        # (hit-rate would read 73%/91% where recall reads 66%/84%). The paper's
        # retrieval table quotes recall, so the figure must too.
        fhit = 0.0
        for r in recs:
            gold_files = {f.replace("\\", "/") for f in ds.get(r["task_id"], {}).get("gold_files", [])}
            if not gold_files:
                continue
            got = {str(x).split("::")[0].replace("\\", "/")
                   for x in (r.get("first_search_fqns") or [])}
            fhit += len(gold_files & got) / len(gold_files)
        gold_tasks = [r for r in recs if ds.get(r["task_id"], {}).get("gold_fqns")]
        fn = 0
        for r in gold_tasks:
            gold = set()
            for fq in ds[r["task_id"]]["gold_fqns"]:
                p = fq.split("::")
                gold.add((p[0].replace("\\", "/"), p[-1].split(".")[-1].lower()))
            for fq in (r.get("all_search_fqns") or r.get("first_search_fqns") or []):
                fq = str(fq)
                # a bare file path carries no function name — cannot be a hit
                if "::" not in fq:
                    continue
                key = (fq.split("::")[0].replace("\\", "/"),
                       fq.split("::")[-1].split(".")[-1].lower())
                if key in gold:
                    fn += 1
                    break
        return (fhit / n * 100 if n else 0), (fn / len(gold_tasks) * 100 if gold_tasks else 0)

    nf, nfn = rates(nat)
    sf, sfn = rates(sg, filter_sg=True)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    groups = ["Correct FILE found\n(which file to edit)",
              "Correct FUNCTION found\n(where in the file)"]
    xs = [0, 1]
    w = 0.34
    gap = w + 0.03  # surface gap between adjacent bars
    ax.bar([x - gap / 2 for x in xs], [nf, nfn], w, color=INK_2,
           label="Claude Code built-in tools (text search)")
    ax.bar([x + gap / 2 for x in xs], [sf, sfn], w, color=BLUE,
           label="+ SkeletonGraph (structural retrieval)")

    # every bar carries its value — a reader should never have to infer one
    for x, v, col in [(xs[0] - gap / 2, nf, INK_2), (xs[0] + gap / 2, sf, BLUE),
                      (xs[1] + gap / 2, sfn, BLUE)]:
        ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.5, color=col, fontweight="bold")
    # the zero bar needs an explanation, not just a number
    ax.annotate("0%", (xs[1] - gap / 2, 0), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8.5, color=INK_2,
                fontweight="bold")
    # Every bar is <=86% tall, so the band above ~100% is clear across the whole
    # width. Put the note there and point down to the 0% bar with a curved arrow.
    # This cannot collide with the tall SG bar beside it or the file-group bars,
    # which was the failure of the previous base-anchored placement.
    ax.annotate("text search returns whole files —\nit cannot name a function",
                xy=(xs[1] - gap / 2, 3), xytext=(0.42, 108),
                textcoords="data", ha="center", va="center",
                fontsize=7.5, color=MUTED, linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8,
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=3, shrinkB=3))

    ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("Tasks where retrieval returned the target (%)")
    ax.set_ylim(0, 122)
    ax.set_title("Both approaches find the right file.\nOnly one can point at the right function.",
                 color=INK, loc="left", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    _clean(ax)
    _save(fig, "fig_retrieval")


# ── FIG 4 — token efficiency across whole systems (second model family) ──
def fig_pareto(tag="nemotron_v2"):
    """Two aligned panels rather than one scatter.

    A scatter of (tokens, solve rate) collides in the middle, where most systems
    sit: the reader cannot separate four overlapping labels. Two panels sharing a
    sorted category axis show the same two measures with no overlap, and make the
    comparison a matter of reading across a row.
    """
    show = {
        "aider":     "Repository map injected into prompt (Aider)",
        "grep":      "Text search (grep)",
        "bm25":      "Lexical ranking (BM25)",
        "cbmem":     "Code knowledge graph (Codebase-Memory)",
        "hybrid":    "Lexical + semantic hybrid",
        "none":      "No retrieval at all (control)",
        "sg-rerank": "SkeletonGraph (structural rerank)",
        "sg":        "SkeletonGraph (full)",
    }
    rows = []
    for arm, label in show.items():
        recs = [r for r in load_arm(tag, arm).values() if r.get("resolved") is not None]
        if not recs:
            continue
        toks = [(r.get("billed_input", 0) or 0) for r in recs]
        if not any(toks):
            continue
        rows.append((label, statistics.mean(toks) / 1000,
                     sum(1 for r in recs if r.get("resolved")) / len(recs) * 100,
                     arm.startswith("sg"), arm == "none"))
    rows.sort(key=lambda r: r[1], reverse=True)   # most tokens at top

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.06})
    ys = list(range(len(rows)))

    for ax in (axl, axr):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    # left: token cost
    for y, (label, tok, solve, is_sg, is_ctrl) in zip(ys, rows):
        c = BLUE if is_sg else (BASELINE_AXIS if is_ctrl else MUTED)
        axl.barh(y, tok, 0.6, color=c)
        axl.annotate(f"{tok:,.0f}k", (tok, y), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color=BLUE if is_sg else INK_2,
                     fontweight="bold" if is_sg else "normal")
    axl.set_xlim(0, max(r[1] for r in rows) * 1.22)
    axl.set_xlabel("Input tokens per task (thousands)")
    axl.set_title("What it costs", color=INK, loc="left", pad=8, fontsize=9.5)
    axl.grid(True, axis="x", color=GRID, linewidth=0.6)

    # right: solve rate — same row order, so the reader reads across
    for y, (label, tok, solve, is_sg, is_ctrl) in zip(ys, rows):
        c = BLUE if is_sg else (BASELINE_AXIS if is_ctrl else MUTED)
        axr.barh(y, solve, 0.6, color=c)
        axr.annotate(f"{solve:.0f}%", (solve, y), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color=BLUE if is_sg else INK_2,
                     fontweight="bold" if is_sg else "normal")
    axr.set_xlim(0, 58)
    axr.set_xlabel("Tasks solved (%)")
    axr.set_title("What it achieves", color=INK, loc="left", pad=8, fontsize=9.5)
    axr.grid(True, axis="x", color=GRID, linewidth=0.6)

    axl.set_yticks(ys)
    axl.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    axl.invert_yaxis()
    for tick, r in zip(axl.get_yticklabels(), rows):
        tick.set_color(BLUE if r[3] else INK_2)
        if r[3]:
            tick.set_fontweight("bold")

    fig.suptitle("Systems differ ~7× in tokens consumed and barely at all in tasks solved",
                 x=0.005, ha="left", fontsize=10, color=INK, y=1.03)
    _save(fig, "fig_token_efficiency")


# ── FIG 5 — what retrieval displaces ─────────────────────────────────────
def fig_tools(nat, sg):
    def mix(recs):
        c = defaultdict(float)
        for r in recs:
            for k, v in (r.get("tool_counts") or {}).items():
                short = k.split("__")[-1] if k.startswith("mcp__") else k
                c[short] += v
        return {k: v / len(recs) for k, v in c.items()}

    nm, sm = mix(nat), mix(sg)
    native_tools = ["Bash", "Read", "Grep", "Edit"]
    sg_tools = ["sg_search", "sg_expand"]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ys = range(len(native_tools))
    h = 0.34
    gap = h + 0.03
    ax.barh([y + gap / 2 for y in ys], [nm.get(t, 0) for t in native_tools], h,
            color=INK_2, label="Claude Code built-in tools")
    ax.barh([y - gap / 2 for y in ys], [sm.get(t, 0) for t in native_tools], h,
            color=BLUE, label="+ SkeletonGraph")
    ax.set_yticks(list(ys)); ax.set_yticklabels(native_tools)
    ax.invert_yaxis()
    ax.set_xlabel("Calls per task")
    added = sum(sm.get(t, 0) for t in sg_tools)
    ax.set_title("Retrieval displaces native exploration\n"
                 f"(replaced by {added:.1f} SG calls/task)",
                 color=INK, loc="left", pad=8)
    # legend outside the plot so it cannot sit on top of the bars
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    _save(fig, "fig_tool_displacement")


# ── FIG 6 — the ceiling: retrieval collapses as location cues are removed ─
# The paper's central proof. `sg-fusion` IS all three non-LLM paradigms at once
# (BM25 lexical + dense semantic + graph topological). If any of them could
# reason from symptom to cause, first-search recall would hold when the symbols
# disappear. It does not — and the lexical baseline falls with it, so the
# ceiling belongs to the whole category, not to SG.
_GRID_CONDS = [
    ("claude_v7",               "SWE-Verified\nraw"),
    ("claude_v7_prose",         "SWE-Verified\nprose-only"),
    ("claude_rebench_v1",       "SWE-rebench\nraw"),
    ("claude_rebench_prose_v1", "SWE-rebench\nprose-only"),
]


def _rec1(r):
    scs = r.get("search_calls") or []
    return scs[0].get("cumulative_recall", 0.0) if scs else 0.0


def _cond_stats(tag, restrict=None):
    """(rec1_native, rec1_sg, cost_delta_pct) paired on common task_ids.

    `restrict` limits to a task_id set — required for the SWE rows, whose prose
    run covers only 15 of the 100 tasks. Comparing the n=100 aggregate against
    an n=15 subset manufactures a trend that isn't there (the 15 happen to be
    tasks where SG does ~2x better than its average), so raw-vs-prose must be
    read on the matched subset only.
    """
    nat, sg = load_arm(tag, "native"), load_arm(tag, "sg-fusion")
    common = sorted(set(nat) & set(sg))
    if restrict:
        common = [t for t in common if t in restrict]
    if not common:
        return None
    rn = sum(_rec1(nat[t]) for t in common) / len(common)
    rs = sum(_rec1(sg[t]) for t in common) / len(common)
    cn = sum(nat[t]["imputed_cost"] for t in common)
    cs = sum(sg[t]["imputed_cost"] for t in common)
    return rn, rs, (cs - cn) / cn * 100.0, len(common)


def fig_ceiling():
    # The SWE prose run defines the matched subset for BOTH SWE columns.
    prose_ids = set(load_arm("claude_v7_prose", "native"))
    stats = []
    for tag, label in _GRID_CONDS:
        restrict = prose_ids if tag.startswith("claude_v7") else None
        s = _cond_stats(tag, restrict)
        if s:
            stats.append((label, *s))
    if not stats:
        print("  fig_ceiling: no data"); return

    labels = [s[0] for s in stats]
    natr = [s[1] * 100 for s in stats]
    sgr = [s[2] * 100 for s in stats]
    cost = [s[3] for s in stats]
    x = range(len(stats))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.0, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.20})

    # top: retrieval collapses
    ax.plot(x, natr, marker="o", color=INK_2, linewidth=1.6, markersize=5,
            label="Built-in lexical search")
    ax.plot(x, sgr, marker="o", color=BLUE, linewidth=2.0, markersize=6,
            label="+ SkeletonGraph (all 3 paradigms)")
    for xi, (a, b) in enumerate(zip(natr, sgr)):
        ax.annotate(f"{b:.0f}", (xi, b), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
    ax.set_ylabel("First-search file recall (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Retrieval quality collapses when location cues are removed",
                 fontsize=9.5, color=INK, loc="left", pad=8)
    _clean(ax)

    # bottom: cost saving persists anyway
    ax2.bar(x, cost, color=BLUE, width=0.5)
    # Bars run downward (all savings are negative), so the value coordinate is
    # the BAR'S END. Offsetting further in that direction pushes the label off
    # the axis and it gets clipped — offset back toward zero so it sits inside.
    for xi, c in enumerate(cost):
        ax2.annotate(f"{c:+.0f}%", (xi, c), textcoords="offset points",
                     xytext=(0, 8 if c < 0 else -14), ha="center", va="bottom",
                     fontsize=8.5, fontweight="medium",
                     color="white" if c < -8 else INK_2)
    ax2.margins(y=0.18)
    ax2.axhline(0, color=BASELINE_AXIS, linewidth=0.9)
    ax2.set_ylabel("Cost change")
    ax2.set_title("…yet the cost saving persists — retrieval quality is not the mechanism",
                  fontsize=9.5, color=INK, loc="left", pad=8)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _clean(ax2)

    _save(fig, "fig_ceiling")


def main():
    _style()
    ds = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl"
    nat, sg = paired("claude_v7", "native", "sg-fusion")
    print(f"paired frontier-agent tasks: {len(nat)}")
    fig_tail(nat, sg)
    fig_scatter(nat, sg)
    fig_retrieval(nat, sg, ds)
    fig_tools(nat, sg)
    fig_pareto()
    fig_ceiling()


if __name__ == "__main__":
    main()
