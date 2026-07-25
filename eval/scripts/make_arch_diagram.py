"""SkeletonGraph architecture diagram — standalone, no data dependencies.

Renders docs/paper/figures/fig_architecture.{pdf,png}. The PDF goes in the paper;
the PNG is for the README and social posts. Accurate to the real implementation:
- 10 language families via tree-sitter (build.py::_SUPPORTED_EXTENSIONS)
- zero-LLM index (symbol table + call graph + PageRank), BM25 + jina-code dense
- 3-way RRF retrieval (fusion) / 2-way (rerank)
- MCP tools: sg_overview, sg_search, sg_get, sg_expand, sg_constraint, sg_log

    python -m eval.scripts.make_arch_diagram
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── palette ──────────────────────────────────────────────────────────────
INK     = "#1b2430"
MUTED   = "#5c6b7a"
BLUE    = "#2f6fd0"; BLUE_FILL   = "#e9f1fc"
GREEN   = "#1f9d55"; GREEN_FILL  = "#e7f6 ee".replace(" ", "")
AMBER   = "#c9772a"; AMBER_FILL  = "#fbf0e1"
SLATE   = "#f2f5f9"
EDGE    = "#c6d0db"
ARROW   = "#8593a2"
WHITE   = "#ffffff"

OUT = Path("docs/paper/figures")


def box(ax, x, y, w, h, lines, *, fill=WHITE, edge=EDGE, tc=INK,
        title_fs=9.0, body_fs=7.6, lw=1.4, rounding=0.020, title_color=None):
    """Rounded box. `lines` = [title, body1, body2, ...]; title bold + colored."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding*100}",
        mutation_aspect=1, linewidth=lw, edgecolor=edge, facecolor=fill,
        joinstyle="round"))
    cx = x + w / 2
    title = lines[0]
    body = lines[1:]
    if body:
        ax.text(cx, y + h * 0.66, title, ha="center", va="center",
                fontsize=title_fs, color=title_color or tc, fontweight="bold")
        ax.text(cx, y + h * 0.30, "\n".join(body), ha="center", va="center",
                fontsize=body_fs, color=MUTED, linespacing=1.3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, color=title_color or tc, fontweight="bold")


def band(ax, x, y, w, h, label, num, accent):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=0, facecolor=SLATE, zorder=0))
    ax.text(x + 1.4, y + h - 2.2, f"{num}", ha="left", va="top",
            fontsize=13, color=accent, fontweight="bold")
    ax.text(x + 4.3, y + h - 2.4, label, ha="left", va="top",
            fontsize=9.5, color=INK, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, *, color=ARROW, lw=1.5, style="-|>",
          dashed=False, rad=0.0, label=None, lx=0, ly=0, lcolor=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
        linestyle=(0, (4, 2)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=5))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                va="center", fontsize=7.0, color=lcolor or MUTED,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec="none"))


def draw_static(ax):
    """Draw the full diagram onto `ax`. Shared by the static figure (main, below)
    and the animated GIF (make_arch_gif.py), so both are pixel-identical in
    layout and neither can silently drift from the other."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── Band 1: INDEX BUILD (offline, zero-LLM) ──────────────────────────
    band(ax, 2, 68, 96, 29, "INDEX BUILD  ·  offline, once per repo", "1", GREEN)
    y1 = 73.5; h1 = 13
    box(ax, 5,  y1, 13, h1, ["Repository", "10 languages", "Py·JS/TS·Go·Rust", "Java·C/C++·C#·Rb·PHP"], edge=EDGE)
    box(ax, 22, y1, 12, h1, ["tree-sitter", "parse → AST", "(no LLM)"], edge=GREEN, title_color=GREEN)
    box(ax, 38, y1+7, 20, 6.5, ["Symbol table", "FQN · line range · signature · docstring"], edge=EDGE, body_fs=7.2)
    box(ax, 38, y1,   20, 6.0, ["Call graph  +  PageRank", "callers / callees · centrality"], edge=EDGE, body_fs=7.2)
    box(ax, 62, y1+7, 15, 6.5, ["BM25 index", "lexical"], edge=BLUE, title_color=BLUE)
    box(ax, 62, y1,   15, 6.0, ["Dense vectors", "jina-code · hash-cached"], edge=BLUE, title_color=BLUE, body_fs=7.2)
    box(ax, 81, y1, 14, h1, [".skeletongraph/", "persisted index", "rebuildable · no LLM"], fill=GREEN_FILL, edge=GREEN, title_color=GREEN)

    arrow(ax, 18, y1+h1/2, 22, y1+h1/2)
    arrow(ax, 34, y1+h1/2, 38, y1+9.5)
    arrow(ax, 34, y1+h1/2, 38, y1+3.0)
    arrow(ax, 58, y1+9.5, 62, y1+9.5)
    arrow(ax, 58, y1+3.0, 62, y1+3.0)
    arrow(ax, 77, y1+9.5, 81, y1+h1/2)
    arrow(ax, 77, y1+3.0, 81, y1+h1/2)

    # ── Band 2: RETRIEVAL (per query) ────────────────────────────────────
    band(ax, 2, 34, 96, 30, "RETRIEVAL  ·  per query", "2", BLUE)
    y2 = 40; h2 = 6.4
    box(ax, 5, 45, 12, 8, ["Issue text", "natural language"], edge=EDGE)
    box(ax, 30, y2+15, 24, h2, ["BM25", "lexical match"], edge=BLUE, title_color=BLUE)
    box(ax, 30, y2+7.5, 24, h2, ["Dense (jina-code)", "semantic · 20s timeout → degrade"], edge=BLUE, title_color=BLUE, body_fs=7.1)
    box(ax, 30, y2, 24, h2, ["Structural rerank", "entity-anchor · PageRank · call-graph"], edge=BLUE, title_color=BLUE, body_fs=7.1)
    box(ax, 60, y2+7.5, 13, 8, ["RRF fuse", "k = 60"], fill=BLUE_FILL, edge=BLUE, title_color=BLUE)
    box(ax, 78, y2+7.5, 17, 8, ["Ranked functions", "edit-shaped payload"], edge=INK)
    ax.text(50, 36.0, "SG-Fusion = all 3 legs   ·   SG-Rerank = BM25 + structural (no dense)",
            ha="center", va="center", fontsize=7.6, color=MUTED, style="italic")

    for yy in (y2+15+h2/2, y2+7.5+h2/2, y2+h2/2):
        arrow(ax, 17, 49, 30, yy, rad=0.03)
        arrow(ax, 54, yy, 60, y2+7.5+4, rad=0.0)
    arrow(ax, 73, y2+7.5+4, 78, y2+7.5+4)
    # the persisted index feeds all three retrieval legs (not the fused output)
    arrow(ax, 88, 73, 55, 60.8, dashed=True, color=GREEN, rad=-0.18,
          label="legs read the persisted index", lx=6, ly=4, lcolor=GREEN)

    # ── Band 3: SERVE ────────────────────────────────────────────────────
    band(ax, 2, 4, 96, 26, "SERVE  ·  Model Context Protocol", "3", AMBER)
    box(ax, 14, 9, 40, 11, ["MCP server", "sg_overview · sg_search · sg_get",
                            "sg_expand · sg_constraint · sg_log"], fill=AMBER_FILL, edge=AMBER, title_color=AMBER)
    box(ax, 62, 9, 30, 11, ["Host agent", "Claude Code (headless)", "· react loop  · CLI"], edge=INK)

    arrow(ax, 78, 40, 78, 20, color=INK, label="ranked functions", lx=8, ly=0)
    arrow(ax, 54, 14.5, 62, 14.5, color=AMBER)
    arrow(ax, 62, 17.5, 54, 17.5, color=AMBER, rad=0.0, label="sg_search(query)", ly=1.8, lcolor=AMBER)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    draw_static(ax)
    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_architecture.{ext}", dpi=200, bbox_inches="tight")
    print(f"  wrote {OUT}/fig_architecture.pdf / .png")


if __name__ == "__main__":
    main()
