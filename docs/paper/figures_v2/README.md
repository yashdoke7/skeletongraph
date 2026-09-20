# Figures — v2 framing

"Retrieval quality is not the bottleneck in agentic program repair."

These are **not** the figures of the preprint. `docs/paper/figures/` holds the v1
set, which belongs to the earlier cost-tail framing; both directories are kept
until the rewrite settles, and nothing here overwrites anything there.

Regenerate everything with:

    python docs/paper/figures_v2/make_figures.py

The script reads the run JSONs under `eval/results/agent/` directly — no numbers
are hardcoded — so a figure that disagrees with the text means the data moved,
not that the figure is stale. Each figure is written as `.pdf` (for LaTeX) and
`.png` (for reading).

| Figure | Claim it carries | Source tags |
|---|---|---|
| `fig1_localisation_ceiling` | Retrieval only has headroom where localisation still fails: gold-file edit rate is 65% for a ReAct agent but 90–97% in real harnesses, with or without SG. | `nemotron_v4`, `codex_v1`, `claude_rebench_v1`, `claude_v7_rep2` |
| `fig2_retrieval_ladder` | File recall@5 spans 24%→78% across backends; pass@1 over the same backends moves 35%→42%. | `paper_verified_*_file.json`, `nemotron_v4` |
| `fig3_cost_is_a_harness_property` | The cost saving tracks how much context the host harness re-sends per turn (Claude Code 39k → −58%; Codex 12k → −5%). | `codex_v1`, `claude_v7_rep2` |
| `fig4_measurement_noise_floor` | Same arm, same config, re-run: 9–16 of 100 verdicts flip and per-task cost moves by a median factor of ~0.7 on both arms alike. | `claude_v7` vs `claude_v7_rep2` |
| `fig5_paired_arms` | The two 100-task paired sets: pass@1, mean cost, first-search rank-1 rate. | `claude_v7_rep2`, `codex_v1` |

## Colour

Slots 1–3 of the reference categorical palette (`#2a78d6`, `#eb6834`, `#1baf7a`) —
the three that clear the all-pairs CVD and normal-vision separation floors, so
they are safe in every chart form used here. Every bar carries a direct value
label: the aqua slot sits below 3:1 contrast on white, which obliges visible
labels rather than colour-alone identity.

## Known caveat carried by fig3 and fig5

The `sg-fusion-plain` runs in `claude_v7_rep2` were executed 1–2 days after that
tag's `native` and `sg-fusion` runs. `fig4` shows a ~33% cost level shift between
two runs of an identical configuration, so the −58% in `fig3` is not yet cleanly
separated from a window effect. A same-window `native` control is required before
these two figures go in the paper.
