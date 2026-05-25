"""Paper figures from the agentic run JSONs.

    python -m eval.agent.plots                 # all arms in results dir
    python -m eval.agent.plots --stage 0-full  # restrict to one stage's arms

Reads eval/results/agent/*.json (same records aggregate.py uses) and writes
PNGs to eval/results/agent/figures/. Pure matplotlib — one chart per figure,
default colors, no seaborn.

Metric tables count only COMPLETED runs (submit / max_turns); incomplete runs
(error / no_tool) are excluded so a dead-endpoint or stale run never skews a
figure — exactly as aggregate.py does. The stop-reason figure uses ALL runs so
failures stay visible.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from . import config

_COMPLETE = ("submit", "max_turns")
# Stable arm order + readable labels for the x-axis.
_ARM_ORDER = ["none", "grep", "bm25", "hybrid", "aider", "cbmem",
              "sg-nosummary", "sg-nograph", "sg-norerank", "sg"]


def _load(stage: str | None) -> List[dict]:
    recs = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name == "summary.json":
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if stage and stage in config.STAGES:
            st = config.STAGES[stage]
            if r.get("arm") not in st.arms or r.get("model") not in st.models:
                continue
        recs.append(r)
    return recs


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _arms_present(by_arm: Dict[str, list]) -> List[str]:
    ordered = [a for a in _ARM_ORDER if a in by_arm]
    extra = [a for a in by_arm if a not in _ARM_ORDER]
    return ordered + sorted(extra)


def _label(arm: str) -> str:
    return config.ARMS[arm].label if arm in config.ARMS else arm


def _bar(ax, arms, values, title, ylabel, fmt="{:.3f}"):
    bars = ax.bar(range(len(arms)), values)
    # Highlight SG bars.
    for i, a in enumerate(arms):
        if a == "sg":
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([_label(a) for a in arms], rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=7)


def make_figures(stage: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = _load(stage)
    if not recs:
        raise SystemExit("no runs found")

    by_arm_all: Dict[str, list] = defaultdict(list)
    for r in recs:
        by_arm_all[r["arm"]].append(r)
    by_arm = {a: [r for r in rs if r.get("stopped") in _COMPLETE]
              for a, rs in by_arm_all.items()}
    arms = _arms_present(by_arm)

    out_dir = config.RUNS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{stage}" if stage else ""

    # ── 1. Retrieval quality: precision + recall(hit) ───────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar(ax1, arms, [_mean([r.get("retrieval_precision") for r in by_arm[a]])
                     for a in arms],
         "Retrieval precision (gold / returned)", "precision")
    _bar(ax2, arms, [_mean([1 if r.get("retrieval_hit") else 0 for r in by_arm[a]])
                     for a in arms],
         "Retrieval hit-rate (recall: any gold found)", "hit-rate")
    fig.tight_layout(); fig.savefig(out_dir / f"retrieval{suffix}.png", dpi=150)
    plt.close(fig)

    # ── 2. Efficiency: turns + cost ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar(ax1, arms, [_mean([r.get("n_turns") for r in by_arm[a]]) for a in arms],
         "Avg turns per task (lower = more efficient)", "turns", fmt="{:.1f}")
    _bar(ax2, arms, [_mean([r.get("imputed_cost") for r in by_arm[a]]) for a in arms],
         "Avg imputed cost per task ($)", "$ / task", fmt="${:.4f}")
    fig.tight_layout(); fig.savefig(out_dir / f"efficiency{suffix}.png", dpi=150)
    plt.close(fig)

    # ── 3. Consolidation gap ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bar(ax, arms, [_mean([(r.get("consolidation") or {}).get("consolidation_gap_files")
                           for r in by_arm[a]]) for a in arms],
         "Consolidation gap (1 - used/read; lower = less wasted retrieval)", "gap")
    fig.tight_layout(); fig.savefig(out_dir / f"consolidation{suffix}.png", dpi=150)
    plt.close(fig)

    # ── 4. Pareto: precision (want high) vs cost (want low) ──────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for a in arms:
        x = _mean([r.get("imputed_cost") for r in by_arm[a]])
        y = _mean([r.get("retrieval_precision") for r in by_arm[a]])
        ax.scatter(x, y, s=120 if a == "sg" else 70,
                   edgecolor="black" if a == "sg" else "none", zorder=3)
        ax.annotate(_label(a), (x, y), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("avg cost per task ($)  →  cheaper is left")
    ax.set_ylabel("retrieval precision  →  better is up")
    ax.set_title("Pareto: retrieval precision vs cost (top-left dominates)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"pareto{suffix}.png", dpi=150)
    plt.close(fig)

    # ── 5. Stop reasons (ALL runs — failures belong here) ───────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    reasons = ["submit", "max_turns", "error", "no_tool"]
    arms_all = _arms_present(by_arm_all)
    bottoms = [0] * len(arms_all)
    for reason in reasons:
        vals = [sum(1 for r in by_arm_all[a] if r.get("stopped") == reason)
                for a in arms_all]
        ax.bar(range(len(arms_all)), vals, bottom=bottoms, label=reason)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(arms_all)))
    ax.set_xticklabels([_label(a) for a in arms_all], rotation=30, ha="right", fontsize=8)
    ax.set_title("Stop reason (failure mode) by arm")
    ax.set_ylabel("runs"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / f"stop_reasons{suffix}.png", dpi=150)
    plt.close(fig)

    # ── 6. pass@1 — only if verify.py has written `resolved` ────────────────
    if any("resolved" in r for r in recs):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _bar(ax, arms, [_mean([1 if r.get("resolved") else 0 for r in by_arm[a]])
                        for a in arms],
             "pass@1 (resolved / attempted)", "pass@1")
        fig.tight_layout(); fig.savefig(out_dir / f"pass1{suffix}.png", dpi=150)
        plt.close(fig)
        figs = 6
    else:
        figs = 5

    print(f"Wrote {figs} figures to {out_dir}")
    for p in sorted(out_dir.glob(f"*{suffix}.png")):
        print(f"  {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--run-dir", default=None, type=Path,
                    help="Override RUNS_DIR (e.g. eval/results/agent/qwen7b_swebench)")
    args = ap.parse_args()
    if args.run_dir:
        config.RUNS_DIR = Path(args.run_dir).expanduser().resolve()
    make_figures(args.stage)


if __name__ == "__main__":
    main()
