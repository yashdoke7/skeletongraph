"""Animated version of the architecture diagram — same layout/coords as
make_arch_diagram.py (imports draw_static + the exact box/arrow geometry), so
the two can never visually drift apart. A dot travels each arrow in sequence,
boxes light up as data reaches them, and a caption names the current step.
Meant for README / LinkedIn, not the paper (which uses the static PDF).

    python -m eval.scripts.make_arch_gif
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

from eval.scripts.make_arch_diagram import (
    draw_static, GREEN, BLUE, AMBER, INK, WHITE, OUT,
)

# ── geometry, copied verbatim from make_arch_diagram.draw_static ──────────
Y1, H1 = 73.5, 13.0
Y2, H2 = 40.0, 6.4

BOX = {  # name -> (x, y, w, h, band_color)
    "repo":      (5,  Y1, 13, H1, GREEN),
    "treesitter":(22, Y1, 12, H1, GREEN),
    "symtable":  (38, Y1 + 7, 20, 6.5, GREEN),
    "callgraph": (38, Y1,     20, 6.0, GREEN),
    "bm25idx":   (62, Y1 + 7, 15, 6.5, GREEN),
    "densevec":  (62, Y1,     15, 6.0, GREEN),
    "skg":       (81, Y1, 14, H1, GREEN),
    "issue":     (5,  45, 12, 8, BLUE),
    "bm25leg":   (30, Y2 + 15, 24, H2, BLUE),
    "densel":    (30, Y2 + 7.5, 24, H2, BLUE),
    "structleg": (30, Y2,       24, H2, BLUE),
    "rrf":       (60, Y2 + 7.5, 13, 8, BLUE),
    "ranked":    (78, Y2 + 7.5, 17, 8, BLUE),
    "mcp":       (14, 9, 40, 11, AMBER),
    "host":      (62, 9, 30, 11, AMBER),
}

# arrows: (x1, y1, x2, y2, rad)
A = {
    "repo_ts":     (18, 80, 22, 80, 0.0),
    "ts_sym":      (34, 80, 38, 83, 0.0),
    "ts_cg":       (34, 80, 38, 76.5, 0.0),
    "sym_bm25":    (58, 83, 62, 83, 0.0),
    "cg_dense":    (58, 76.5, 62, 76.5, 0.0),
    "bm25_skg":    (77, 83, 81, 80, 0.0),
    "dense_skg":   (77, 76.5, 81, 80, 0.0),
    "issue_bm25":  (17, 49, 30, 58.2, 0.03),
    "issue_dense": (17, 49, 30, 50.7, 0.03),
    "issue_struct":(17, 49, 30, 43.2, 0.03),
    "bm25_rrf":    (54, 58.2, 60, 51.5, 0.0),
    "dense_rrf":   (54, 50.7, 60, 51.5, 0.0),
    "struct_rrf":  (54, 43.2, 60, 51.5, 0.0),
    "rrf_ranked":  (73, 51.5, 78, 51.5, 0.0),
    "ranked_host": (78, 40, 78, 20, 0.0),
    "host_mcp":    (62, 17.5, 54, 17.5, 0.0),
    "mcp_host":    (54, 14.5, 62, 14.5, 0.0),
}

# ── the storyboard ───────────────────────────────────────────────────────
# (arrow keys active this beat, box keys newly lit this beat, caption)
BEATS = [
    ([], [], "SkeletonGraph — how one search actually flows"),
    (["repo_ts"], ["repo", "treesitter"], "1 · Parse the repo — tree-sitter, zero LLM, 10 languages"),
    (["ts_sym", "ts_cg"], ["symtable", "callgraph"], "1 · Build the symbol table + call graph"),
    (["sym_bm25", "cg_dense"], ["bm25idx", "densevec"], "1 · Index lexically (BM25) and semantically (dense)"),
    (["bm25_skg", "dense_skg"], ["skg"], "1 · Persist — rebuildable from source, no LLM ever"),
    ([], ["issue"], "2 · An issue arrives as plain text"),
    (["issue_bm25", "issue_dense", "issue_struct"], ["bm25leg", "densel", "structleg"],
     "2 · Three signals search the index in parallel"),
    (["bm25_rrf", "dense_rrf", "struct_rrf"], ["rrf"], "2 · Reciprocal rank fusion (k = 60)"),
    (["rrf_ranked"], ["ranked"], "2 · Ranked FUNCTIONS — not files"),
    (["ranked_host"], ["host"], "3 · Served to the agent"),
    (["host_mcp", "mcp_host"], ["mcp"], "3 · sg_search(query)  ↔  ranked results, over MCP"),
    ([], [], "SkeletonGraph — zero-LLM index, structural + lexical + semantic retrieval"),
]

MOVE_FRAMES = 6   # sub-frames of dot travel per beat
HOLD_FRAMES = 10  # frames to hold the settled beat before advancing


def _lerp_point(x1, y1, x2, y2, rad, t):
    """Point at parameter t along the same curve `arrow()` draws (straight for
    rad=0, else a light quadratic-Bezier approximation of matplotlib's arc3)."""
    if rad == 0.0:
        return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    dist = (dx ** 2 + dy ** 2) ** 0.5
    cx, cy = mx - rad * dy, my + rad * dx  # perpendicular offset, scaled like arc3
    # quadratic Bezier P0,C,P2
    u = 1 - t
    return (u * u * x1 + 2 * u * t * cx + t * t * x2,
            u * u * y1 + 2 * u * t * cy + t * t * y2)


def _highlight_box(ax, key, alpha):
    x, y, w, h, color = BOX[key]
    pad = 0.6
    ax.add_patch(FancyBboxPatch(
        (x - pad, y - pad), w + 2 * pad, h + 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.03", mutation_aspect=1,
        linewidth=2.4, edgecolor=color, facecolor=color, alpha=alpha, zorder=1))


def build_schedule():
    """Flatten BEATS into a list of (active_arrow_keys, t_or_None, lit_boxes_so_far, caption)."""
    frames = []
    lit = []
    for arrows, new_boxes, caption in BEATS:
        # settle frames from the PREVIOUS beat already appended; now animate this beat
        for i in range(MOVE_FRAMES):
            t = (i + 1) / MOVE_FRAMES
            frames.append((arrows, t, list(lit), caption))
        lit = lit + [b for b in new_boxes if b not in lit]
        for _ in range(HOLD_FRAMES):
            frames.append(([], None, list(lit), caption))
    return frames


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    schedule = build_schedule()

    def render(i):
        ax.clear()
        ax.set_xlim(0, 100); ax.set_ylim(-2, 106); ax.axis("off")
        draw_static(ax)
        arrows, t, lit, caption = schedule[i]
        for key in lit:
            _highlight_box(ax, key, alpha=0.16)
        if t is not None:
            for akey in arrows:
                x1, y1, x2, y2, rad = A[akey]
                px, py = _lerp_point(x1, y1, x2, y2, rad, t)
                ax.add_patch(Circle((px, py), 1.15, color=INK, zorder=10))
        ax.text(50, 103, caption, ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc=WHITE, ec="#c6d0db", lw=1.0))
        return []

    anim = FuncAnimation(fig, render, frames=len(schedule), blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "fig_architecture_animated.gif"
    anim.save(out_path, writer=PillowWriter(fps=12))
    print(f"  wrote {out_path}  ({len(schedule)} frames)")


if __name__ == "__main__":
    main()
