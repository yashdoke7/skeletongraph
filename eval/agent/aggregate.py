"""Aggregate every run JSON into the Axis 1/2/3/5 tables + significance tests.

    python -m eval.agent.aggregate
    python -m eval.agent.aggregate --stage B

Reads eval/results/agent/*.json (written by run_agent, verdicts by verify) and
writes eval/results/agent/SUMMARY.md.

Significance: McNemar's exact test on paired pass/fail (SG vs each baseline),
bootstrap CI on retrieval-hit rate. Pure-stdlib — no scipy needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from . import config

# Default gold-fqn dataset (tightened: true patched function from AST, not hunk
# headers — see eval/scripts/tighten_gold_fqns.py). Used for the file-vs-function
# localization table. Skipped gracefully if the file isn't present.
_DEFAULT_GOLD_FQNS = Path(
    r"C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100_fqn.jsonl")


def _load(stage: str | None, task_ids: set | None = None) -> list:
    recs = []
    for p in sorted(config.RUNS_DIR.glob("*.json")):
        # Skip non-record files: leading-underscore artifacts, and the
        # machine-readable summaries this script writes (summary.json AND
        # summary_<label>.json — the latter is why the bare name check wasn't
        # enough and aggregate KeyError'd on its own output).
        if p.name.startswith("_") or p.name.startswith("summary"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # A valid run record always has "arm". Anything else (stray/partial
        # JSON in a shared dir) is skipped rather than crashing the whole run.
        if not isinstance(r, dict) or "arm" not in r:
            continue
        if stage and stage in config.STAGES:
            st = config.STAGES[stage]
            if r.get("arm") not in st.arms or r.get("model") not in st.models:
                continue
        # task_ids filter → restrict to a task subset (e.g. the first-30 jsonl)
        # so a single 100-task run yields both a 30-task and a 100-task summary.
        if task_ids is not None and r.get("task_id") not in task_ids:
            continue
        recs.append(r)
    return recs


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _recall_first(r) -> float | None:
    """Fractional file-recall after the FIRST search: |gold ∩ first_hits| / |gold|.
    Rewards getting it right in one shot (SG's efficiency thesis). None if the
    task has no gold or the agent never searched."""
    if not (r.get("gold_files")):
        return None
    scs = r.get("search_calls") or []
    return scs[0].get("cumulative_recall", 0.0) if scs else 0.0


def _recall_cum(r) -> float | None:
    """Fractional file-recall across ALL searches (the eventual ceiling).
    cumulative_recall is monotonic, so the last call's value is the max."""
    if not (r.get("gold_files")):
        return None
    scs = r.get("search_calls") or []
    return scs[-1].get("cumulative_recall", 0.0) if scs else 0.0


def _cost_cached(r) -> float:
    """Supplementary: cost IF the static system-prompt prefix were cached each
    turn (the minimal, well-defined caching benefit). Token COUNT is unchanged —
    caching only re-bills the repeated prefix at the cheap cached rate. Computed
    from the turn-0 prompt size, so it works on existing runs with no re-run. This
    mostly matters for aider (its 48K repo map is re-sent every turn)."""
    base = r.get("imputed_cost") or 0.0
    turns = r.get("turns") or []
    nt = r.get("n_turns") or len(turns)
    if not turns or nt < 2:
        return round(base, 5)
    pre = (turns[0].get("usage") or {}).get("prompt_tokens", 0) or 0
    save = ((nt - 1) * pre
            * (config.PRICE_INPUT_PER_M - config.PRICE_CACHED_INPUT_PER_M) / 1e6)
    return round(max(0.0, base - save), 5)


# ── localization re-scoring (file vs function) ──────────────────────────────
# Re-score each run's FIRST search result (the raw file::symbol list the model
# saw, stored in turns) against gold_files / gold_fqns. Function match is fuzzy:
# same file AND last name token equal ('header.py::fromstring' matches
# 'header.py::Header.fromstring'). Surfaces the file→function gap.

_RANK_LINE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")


def _parse_ranked(result: str) -> list:
    if not result or result.startswith(("ERROR", "No results")):
        return []
    out = []
    for line in result.splitlines():
        m = _RANK_LINE.match(line)
        if m:
            # first whitespace-token = the FQN/file (drops trailing annotations
            # like cbmem's "  (qualified_name: …)" so the symbol parses cleanly)
            out.append(m.group(1).strip().split()[0])
    return out


def _first_search_fqns(rec: dict) -> list:
    for t in rec.get("turns", []):
        for call in t.get("tool_calls", []):
            if call.get("name") in ("search_code", "cbmem_search", "graphify_search"):
                return _parse_ranked(call.get("result", "") or "")
    return []


def _all_search_fqns(rec: dict) -> list:
    seen, out = set(), []
    for t in rec.get("turns", []):
        for call in t.get("tool_calls", []):
            if call.get("name") in ("search_code", "cbmem_search", "graphify_search"):
                for fq in _parse_ranked(call.get("result", "") or ""):
                    if fq not in seen:
                        seen.add(fq); out.append(fq)
    return out


def _file_of(fqn: str) -> str:
    return fqn.split("::", 1)[0].replace("\\", "/").strip()


def _func_of(fqn: str) -> str:
    tail = fqn.split("::", 1)[-1] if "::" in fqn else ""
    return tail.split(".")[-1].strip() if tail else ""


def _func_match(r: str, g: str) -> bool:
    return _file_of(r) == _file_of(g) and _func_of(r) and _func_of(r) == _func_of(g)


def _recall_rank(retrieved: list, golds: list, match) -> tuple:
    """(recall@10, hit, rank-of-first-gold) over the top-10."""
    if not golds:
        return (None, None, None)
    top = retrieved[:10]
    matched = sum(1 for g in golds if any(match(r, g) for r in top))
    rank = 0
    for i, r in enumerate(top, 1):
        if any(match(r, g) for g in golds):
            rank = i
            break
    return (matched / len(golds), 1 if matched else 0, rank)


def _load_gold_fqns(path: Path) -> dict:
    gold = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["task_id"]] = (d.get("gold_files", []), d.get("gold_fqns", []))
    return gold


def _load_repo_paths(path: Path) -> dict:
    """task_id -> repo_path (needed to locate each graphify graph's build-cost
    sidecar, <repo_path>/graphify-out/.graphify_analysis.json)."""
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[d["task_id"]] = d.get("repo_path", "")
    except Exception:
        pass
    return out


def _graphify_build_tokens(repo_path: str) -> tuple:
    """(input, output) one-time graph-extraction tokens for a graphify repo,
    read from the graph's <repo>/graphify-out/.graphify_analysis.json sidecar.
    (0, 0) if absent (graph not built, or arm doesn't build a graph)."""
    if not repo_path:
        return (0, 0)
    try:
        p = Path(repo_path) / "graphify-out" / ".graphify_analysis.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d.get("tokens") or {}
        return (int(t.get("input", 0) or 0), int(t.get("output", 0) or 0))
    except Exception:
        return (0, 0)


def _summary_build_tokens(repo_path: str) -> tuple:
    """(input, output) one-time summary generation tokens for the repo."""
    if not repo_path:
        return (0, 0)
    try:
        p = Path(repo_path) / ".skeletongraph" / "summcache_openai.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        u = d.get("__usage__") or {}
        return (int(u.get("input", 0) or 0), int(u.get("output", 0) or 0))
    except Exception:
        return (0, 0)


# Arms that build an LLM knowledge graph (one-time, per repo). For these the
# total cost = agent cost + amortized build cost. cbmem builds a graph too but
# its binary does not expose token counts, so its build cost is UNDER-COUNTED
# (flagged in the table) rather than silently zero.
_LLM_INDEX_ARMS = ("graphify", "summary-llm-bm25", "summary-llm-dense")


# ── significance ────────────────────────────────────────────────────────────


def mcnemar(pairs: list) -> tuple:
    """Exact McNemar on a list of (sg_pass, base_pass) bools. Returns (b, c, p)."""
    b = sum(1 for s, o in pairs if s and not o)   # SG wins
    c = sum(1 for s, o in pairs if o and not s)   # baseline wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial p
    p = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def bootstrap_ci(values: list, iters: int = 2000) -> tuple:
    """95% bootstrap CI of the mean of a 0/1 list."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(config.SEED)
    means = []
    n = len(values)
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (round(means[int(0.025 * iters)], 4),
            round(means[int(0.975 * iters)], 4))


# ── main ────────────────────────────────────────────────────────────────────


def aggregate(stage: str | None, task_ids: set | None = None,
              out_label: str = "", gold_fqns_path: Path | None = None) -> None:
    recs = _load(stage, task_ids)
    if not recs:
        raise SystemExit("no runs found")

    # group by arm (collapsing repeats/models for the headline table)
    by_arm_all: dict = defaultdict(list)
    for r in recs:
        by_arm_all[r["arm"]].append(r)
    # Metric tables count only COMPLETED runs. A run that errored (endpoint down)
    # or never produced a tool call has no meaningful retrieval/edit data, and
    # including it as zeros silently drags every average down — and lets a stale
    # failed run from an earlier session pollute the summary. Failures are still
    # surfaced in the Stop-reason table below.
    _COMPLETE = ("submit", "max_turns")
    by_arm: dict = {a: [r for r in rs if r.get("stopped") in _COMPLETE]
                    for a, rs in by_arm_all.items()}
    n_excluded = len(recs) - sum(len(v) for v in by_arm.values())

    # Gold fqns for the function-level columns (tightened gold; skipped if absent).
    gpath = gold_fqns_path if gold_fqns_path is not None else _DEFAULT_GOLD_FQNS
    gold = {}
    repo_of = {}
    if gpath and Path(gpath).exists():
        try:
            gold = _load_gold_fqns(gpath)
        except Exception:
            gold = {}
        repo_of = _load_repo_paths(gpath)

    def _loc(rs):
        """(funcR@10, funcHit, funcRank) for an arm's runs vs gold_fqns."""
        rr = [r for r in rs if r.get("task_id") in gold]
        if not rr:
            return (None, None, None)
        fr, fh, rk = [], [], []
        for r in rr:
            gq = gold[r["task_id"]][1]
            a, b, k = _recall_rank(_first_search_fqns(r), gq, _func_match)
            fr.append(a); fh.append(b)
            if k:
                rk.append(k)
        return (_mean(fr), _mean(fh),
                round(statistics.median(rk), 1) if rk else None)

    lines = ["# Agentic Evaluation — Summary", ""]
    if stage:
        lines.append(f"Stage: **{stage}**  ·  {config.STAGES[stage].note}")
    lines += [f"Runs: {len(recs)}  ·  {n_excluded} incomplete excluded  ·  "
              f"{len(by_arm)} arms", ""]

    # ── Build row data ───────────────────────────────────────────────────────
    arm_pass: dict = {}
    rows = []
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        _fr, _fh, _frk = _loc(rs)        # function-level localization (vs gold_fqns)
        # pass@1 denominator consistency: a completed run counts iff we can
        # decide pass/fail for it. An EMPTY patch is always a fail (it changes
        # nothing), so it counts even if verify never wrote a verdict. A
        # NON-empty run only counts once it has a `resolved` verdict (we can't
        # score an unverified patch). This stops the bug where arms with
        # unverified empty patches got an inflated pass@1 (smaller denominator).
        def _verdict(r):
            if "resolved" in r:
                return 1 if r.get("resolved") else 0
            if not (r.get("model_patch") or "").strip():
                return 0          # empty patch = guaranteed fail
            return None           # non-empty, unverified → exclude
        resolved = [v for v in (_verdict(r) for r in rs) if v is not None]
        arm_pass[arm] = {r["task_id"]: bool(_verdict(r))
                         for r in rs if _verdict(r) is not None}
        p1 = _mean(resolved) if resolved else None
        ranks = [r.get("retrieval_rank") for r in rs
                 if r.get("retrieval_rank")]          # nonzero = found
        med_rank = round(statistics.median(ranks), 1) if ranks else None
        label = config.ARMS.get(arm, arm).label if arm in config.ARMS else arm
        # ── LLM index/build cost (graphify/cbmem) — a per-repo cost SG never pays.
        # Each graphify task's checkout has its own graph; read its sidecar token
        # counts and price them at the EXTRACTION-model rate (≠ agent rate). The
        # build LLM is NOT cached, so it adds to both total_cost and total_cost_c.
        agent_cost = _mean([r.get('imputed_cost') for r in rs])
        agent_cost_c = _mean([_cost_cached(r) for r in rs])
        bi = bo = 0
        if arm in _LLM_INDEX_ARMS:
            for r in rs:
                rp = repo_of.get(r.get("task_id"), "")
                if "graphify" in arm:
                    i, o = _graphify_build_tokens(rp)
                elif "summary-llm" in arm:
                    i, o = _summary_build_tokens(rp)
                else:
                    i = o = 0
                bi += i; bo += o
        n_rs = max(1, len(rs))
        idx_in, idx_out = bi / n_rs, bo / n_rs          # mean per task
        idx_cost = config.impute_extract_cost(idx_in, idx_out) if arm in _LLM_INDEX_ARMS else None
        total_cost = round(agent_cost + (idx_cost or 0.0), 5)
        total_cost_c = round(agent_cost_c + (idx_cost or 0.0), 5)
        rows.append({
            "key": arm,            # short id (e.g. "sg-chain") — for tables
            "label": label,        # long description — for the legend only
            "n": len(rs),
            "pass1": p1,
            # hit = binary first-search hit-rate (≥1 gold file) — kept for
            # back-compat, but it rewards dumping many files. rec1/reccum are
            # the fair fractional recalls.
            "hit": _mean([1 if r.get('retrieval_hit') else 0 for r in rs]),
            "rec1": _mean([_recall_first(r) for r in rs]),
            "reccum": _mean([_recall_cum(r) for r in rs]),
            "prec": _mean([r.get('retrieval_precision') for r in rs]),
            "rank": med_rank,
            "funcr": _fr,          # function-level recall@10 (first search)
            "funchit": _fh,        # found gold FUNCTION in top-10
            "funcrank": _frk,      # median rank of first gold function
            # patch-rate = fraction of runs that produced ANY non-empty patch.
            "patchrate": _mean([1 if (r.get("consolidation") or {}).get(
                "files_in_patch_count", 0) > 0 else 0 for r in rs]),
            "egold": _mean([1 if r.get('edited_gold_file') else 0 for r in rs]),
            "turns": _mean([r.get('n_turns') for r in rs]),
            "intok": round(_mean([r.get('billed_input') for r in rs])),
            "outtok": round(_mean([r.get('billed_output') for r in rs])),
            "cost": agent_cost,                # AGENT-loop cost only (all arms)
            "cost_c": agent_cost_c,             # agent cost w/ prefix caching
            # index/build LLM cost (graphify/cbmem). None for arms with no LLM
            # index → renders as "—" / null in tables and figures.
            "idx_in": round(idx_in) if arm in _LLM_INDEX_ARMS else None,
            "idx_out": round(idx_out) if arm in _LLM_INDEX_ARMS else None,
            "idx_cost": idx_cost,
            "total_cost": total_cost,           # agent + index build
            "total_cost_c": total_cost_c,       # (agent cached) + index build
        })

    sorted_rows = sorted(rows, key=lambda r: -(r['pass1'] or 0))

    # Format helpers — consistent across all tables.
    def _pct(x):   return f"{x*100:.1f}%" if x is not None else "n/a"
    def _p0(x):    return f"{x*100:.0f}%" if x is not None else "n/a"
    def _f3(x):    return f"{x:.3f}" if x is not None else "—"
    def _rk(x):    return f"{x:.1f}" if x is not None else "—"

    # ── Compact monospace table (collapsible) ────────────────────────────────
    hdr = (f"{'arm':<18}{'n':>4}{'pass@1':>8}{'patch%':>8}{'rec@1':>8}"
           f"{'rec@cum':>9}{'prec':>7}{'fRank':>6}{'funcR@10':>9}{'funcHit':>8}"
           f"{'fnRank':>7}{'tokens':>9}{'turns':>7}{'cost$':>9}{'cost(c)':>9}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in sorted_rows:
        lines.append(
            f"{r['key']:<18}{r['n']:>4}{_pct(r['pass1']):>8}{_p0(r['patchrate']):>8}"
            f"{_f3(r['rec1']):>8}{_f3(r['reccum']):>9}{_f3(r['prec']):>7}"
            f"{_rk(r['rank']):>6}{_f3(r['funcr']):>9}{_p0(r['funchit']):>8}"
            f"{_rk(r['funcrank']):>7}{r['intok']:>9,}{r['turns']:>7.1f}"
            f"{('$'+format(r['cost'],'.4f')):>9}{('$'+format(r['cost_c'],'.4f')):>9}"
        )
    lines.append("")
    lines += ["_pass@1_ = resolved % (empty patch = fail) · _patch%_ = any patch · "
              "_rec@1/rec@cum_ = FILE-recall after first / all searches · _prec_ = "
              "first-search precision · _fRank_ = median rank of first gold FILE · "
              "_funcR@10/funcHit/fnRank_ = FUNCTION-level recall@10 / hit / median rank "
              "(gold function, tightened gold) · _tokens_ = mean input tokens (uncached, "
              "uniform) · _cost$_ = uncached cost · _cost(c)_ = cost if the static "
              "system-prompt prefix is cached (token count unchanged).",
              "", "</details>"]

    # ── LLM index/build cost table (graphify/cbmem only) ─────────────────────
    # Separates the THREE cost components the comparison must keep honest:
    #   (1) agent-loop cost  — the per-task LLM cost EVERY arm pays (cost$ above)
    #   (2) index/build cost — the one-time per-repo LLM graph build that ONLY
    #       graph competitors pay (SG's tree-sitter index is zero-LLM → null)
    #   (3) total            — (1)+(2), at the respective model prices
    # Priced at the EXTRACTION model rate (config.PRICE_EXTRACT_*; Llama-3.3-70B
    # on NIM), distinct from the agent model rate. Empty if no LLM-index arm ran.
    idx_rows = [r for r in sorted_rows if r.get("idx_cost") is not None]
    if idx_rows:
        lines += ["", "### LLM index/build cost (graph competitors only)", "",
                  "_Agent cost is the per-task loop cost all arms pay; build cost is "
                  "the one-time per-repo LLM graph extraction that only graph "
                  "competitors pay (SG = zero-LLM tree-sitter = null). Build tokens "
                  f"priced at the extraction model ({config.PRICE_EXTRACT_INPUT_PER_M}/"
                  f"{config.PRICE_EXTRACT_OUTPUT_PER_M} $/M in/out); agent tokens at "
                  f"{config.PRICE_INPUT_PER_M}/{config.PRICE_OUTPUT_PER_M}. Build cost "
                  "is per-task amortized over the arm's runs._", "",
                  "| Arm | build in-tok | build out-tok | build $ | agent $ "
                  "| total $ | total $(cached) |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for r in idx_rows:
            lines.append(
                f"| `{r['key']}` | {r['idx_in']:,} | {r['idx_out']:,} | "
                f"${r['idx_cost']:.4f} | ${r['cost']:.4f} | "
                f"${r['total_cost']:.4f} | ${r['total_cost_c']:.4f} |")
        lines += ["", "_cbmem builds a tree-sitter knowledge graph with NO LLM "
                  "(index_repository runs in ~30s, no API key) — so its build cost is "
                  "genuinely $0, like SG. Only graphify pays an LLM build. cbmem is thus "
                  "a fair zero-LLM structural control; SG wins on retrieval + tokens, not build._"]

    # ── Markdown pipe table (for paper — comes first for easy copy) ──────────
    # cost columns are kept SEPARATE on purpose (asked for explicitly):
    #   agent$   = per-task agent-loop cost EVERY arm pays (was cost$)
    #   index$   = one-time per-repo LLM build cost, ONLY graph competitors pay
    #              (SG/baselines = zero-LLM = —); never silently folded into agent$
    #   total$   = agent$ + index$ — the grand total of everything an arm costs
    def _idx(r):  return f"${r['idx_cost']:.4f}" if r.get('idx_cost') is not None else "—"
    def _tot(r):  return f"${r.get('total_cost', r['cost']):.4f}"
    lines += ["### Results (sorted by pass@1)", "",
              "| Arm | n | pass@1 | patch% | rec@1 | rec@cum | prec | fRank "
              "| funcR@10 | funcHit | fnRank | tokens | turns | agent$ | agent$(c) "
              "| index$ | total$ |",
              "| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:"
              "| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|"]
    for r in sorted_rows:
        lines.append(
            f"| `{r['key']}` | {r['n']} | {_pct(r['pass1'])} | {_p0(r['patchrate'])} "
            f"| {_f3(r['rec1'])} | {_f3(r['reccum'])} | {_f3(r['prec'])} | {_rk(r['rank'])} "
            f"| {_f3(r['funcr'])} | {_p0(r['funchit'])} | {_rk(r['funcrank'])} "
            f"| {r['intok']:,} | {r['turns']:.1f} | ${r['cost']:.4f} | ${r['cost_c']:.4f} "
            f"| {_idx(r)} | {_tot(r)} |"
        )

    # ── Arm legend (short id → what it is) ───────────────────────────────────
    lines += ["", "**Arms:**", ""]
    for r in sorted(rows, key=lambda r: r['key']):
        lines.append(f"- `{r['key']}` — {r['label']}")
    lines.append("")

    # (Function-level localization columns are folded into the main table above.
    # For the deep dive — funcMRR, cumFuncHit, fileR@10 — run
    # `python -m eval.scripts.localization_metrics`.)


    # ── Data integrity — SG runs that silently lost their embedding index ────
    lines += ["", "## Data integrity", ""]
    # sg-noembed runs WITHOUT embeddings on purpose — don't flag it as degraded.
    sg_runs = [r for a, rs in by_arm.items()
               if a.startswith("sg") and a != "sg-noembed" for r in rs]
    degraded = [r for r in sg_runs if r.get("embeddings_used") is False]
    if degraded:
        lines.append(f"**WARNING — {len(degraded)}/{len(sg_runs)} SG run(s) ran "
                      f"WITHOUT embeddings (BM25-only fallback). Their numbers "
                      f"are NOT real SG — exclude or re-run:**")
        for r in degraded:
            lines.append(f"- `{r.get('run_id')}`")
    elif sg_runs:
        lines.append(f"All {len(sg_runs)} SG run(s) used the embedding index. OK")
    else:
        lines.append("_No SG runs in this selection._")

    # ── Trajectory dynamics ──────────────────────────────────────────────────
    # When does the agent first edit? How many edits does it attempt vs land?
    # Does the empty-submit guard fire? Differentiates retrieval from capability.
    lines += ["", "## Trajectory dynamics", "",
              "| Arm | n | edits att | edits ok | succ% | guard% | tte |",
              "| --- | ---:| ---:| ---:| ---:| ---:| ---:|"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        att = [r.get("edits_attempted", 0) for r in rs]
        suc = [r.get("edits_successful", 0) for r in rs]
        rate = round(sum(suc) / sum(att), 4) if sum(att) else 0.0
        guard = _mean([1 if r.get("empty_submit_blocked") else 0 for r in rs])
        ttes = [r.get("time_to_first_edit_turn") for r in rs
                if r.get("time_to_first_edit_turn") is not None]
        mean_tte = round(sum(ttes) / len(ttes), 1) if ttes else None
        tte_s = f"{mean_tte:.1f}" if mean_tte is not None else "n/a"
        lines.append(
            f"| `{arm}` | {len(rs)} | {_mean(att):.2f} "
            f"| {_mean(suc):.2f} | {rate*100:.1f}% | {guard*100:.1f}% | {tte_s} |"
        )

    # ── Consolidation gap ────────────────────────────────────────────────────
    # Files retrieved but never appearing in the final patch. Lower is better.
    lines += ["", "## Consolidation gap", "",
              "| Arm | n | files read | files patched | gap |",
              "| --- | ---:| ---:| ---:| ---:|"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        gaps = [(r.get("consolidation") or {}).get("consolidation_gap_files")
                for r in rs]
        read_c = [(r.get("consolidation") or {}).get("files_read_count", 0)
                  for r in rs]
        patch_c = [(r.get("consolidation") or {}).get("files_in_patch_count", 0)
                   for r in rs]
        lines.append(
            f"| `{arm}` | {len(rs)} | {_mean(read_c):.2f} "
            f"| {_mean(patch_c):.2f} | {_mean(gaps):.3f} |"
        )

    # ── Search dynamics ──────────────────────────────────────────────────────
    # Did the agent thrash on retrieval? High search-calls + low precision = thrash.
    lines += ["", "## Search dynamics", "",
              "| Arm | n | searches | unique files | err% |",
              "| --- | ---:| ---:| ---:| ---:|"]
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        ncalls = [r.get("n_search_calls", 0) for r in rs]
        uniq = [r.get("unique_files_retrieved_total", 0) for r in rs]
        errs = []
        for r in rs:
            scs = r.get("search_calls") or []
            errs.append(sum(1 for sc in scs if sc.get("error")) / max(len(scs), 1))
        lines.append(
            f"| `{arm}` | {len(rs)} | {_mean(ncalls):.2f} "
            f"| {_mean(uniq):.2f} | {_mean(errs)*100:.1f}% |"
        )

    # ── Significance — SG vs each baseline (McNemar, paired) ────────────────
    lines += ["", "## Significance — SG vs each baseline (McNemar)", ""]
    if "sg" in arm_pass and any("resolved" in r for r in recs):
        lines += ["| Baseline | SG wins | base wins | p-value | verdict |",
                  "| --- | ---:| ---:| ---:| --- |"]
        sg = arm_pass["sg"]
        for arm in sorted(by_arm):
            if arm == "sg":
                continue
            base = arm_pass.get(arm, {})
            common = sorted(set(sg) & set(base))
            pairs = [(sg[t], base[t]) for t in common]
            if not pairs:
                continue
            b, c, p = mcnemar(pairs)
            verdict = ("SG better" if b > c and p < 0.05 else
                       "baseline better" if c > b and p < 0.05 else
                       "no sig. diff")
            lines.append(f"| `{arm}` | {b} | {c} | {p:.4f} | {verdict} |")
    else:
        lines.append("_Run verify.py first — no pass/fail verdicts yet._")

    # ── Retrieval-hit rate — 95% bootstrap CI ────────────────────────────────
    lines += ["", "## Retrieval-hit rate — 95% CI", "",
              "| Arm | hit rate | 95% CI |",
              "| --- | ---:| --- |"]
    for arm in sorted(by_arm):
        vals = [1 if r.get("retrieval_hit") else 0 for r in by_arm[arm]]
        lo, hi = bootstrap_ci(vals)
        lines.append(f"| `{arm}` | {_mean(vals)*100:.1f}% | [{lo:.3f}, {hi:.3f}] |")

    # ── Stop reason (failure mode) ───────────────────────────────────────────
    lines += ["", "## Stop reason by arm", "",
              "| Arm | submit | max_turns | error | no_tool |",
              "| --- | ---:| ---:| ---:| ---:|"]
    for arm in sorted(by_arm_all):
        rs = by_arm_all[arm]          # ALL runs — failures belong in this table
        cnt = defaultdict(int)
        for r in rs:
            cnt[r.get("stopped", "?")] += 1
        lines.append(f"| `{arm}` | {cnt['submit']} | {cnt['max_turns']} "
                     f"| {cnt['error']} | {cnt['no_tool']} |")

    sfx = f"_{out_label}" if out_label else ""
    out = config.RUNS_DIR / f"SUMMARY{sfx}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    # Console preview — the file is UTF-8, but a Windows cp1252 console can't
    # encode box-drawing/unicode chars, so sanitize for stdout only.
    preview = "\n".join(lines[:22])
    print(preview.encode("ascii", "replace").decode("ascii"))
    print(f"... ({len(lines)} total lines in SUMMARY.md — open the file for the full breakdown)")

    # Machine-readable summary alongside the human-readable SUMMARY.md.
    # Keyed by arm — retrieval/efficiency numbers for downstream scripting.
    summary = {}
    for arm in sorted(by_arm):
        recs = by_arm[arm]
        complete = [r for r in by_arm_all[arm]
                    if r.get("stopped") in ("submit", "max_turns")]
        pass_vals = []
        for r in recs:
            if "resolved" in r:
                pass_vals.append(1 if r.get("resolved") else 0)
            elif "verdict" in r and r.get("verdict") is not None:
                pass_vals.append(1 if r.get("verdict") else 0)
        summary[arm] = {
            "n": len(recs),
            "n_complete": len(complete),
            "retrieval_hit": _mean([r["retrieval_hit"] for r in recs
                                    if "retrieval_hit" in r]),
            "precision": _mean([r["retrieval_precision"] for r in recs
                                if "retrieval_precision" in r]),
            "edited_gold": _mean([r["edited_gold_file"] for r in recs
                                  if "edited_gold_file" in r]),
            "avg_turns": _mean([r["n_turns"] for r in complete
                                if "n_turns" in r]),
            "avg_input_tok": _mean([r["billed_input"] for r in complete
                                    if "billed_input" in r]),
            "avg_output_tok": _mean([r["billed_output"] for r in complete
                                     if "billed_output" in r]),
            "total_cost_usd": round(sum(r.get("imputed_cost", 0)
                                        for r in complete), 4),
            "pass1": _mean(pass_vals) if pass_vals else None,
        }
    summary["_rows"] = rows          # full per-arm rows (all metrics) for plotting
    jsout = config.RUNS_DIR / f"summary{sfx}.json"
    jsout.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {jsout}")


def _task_ids_from(path: Path) -> set:
    """Load the task_id set from a dataset jsonl (for the --tasks subset filter)."""
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None,
                    help="Filter to a specific stage's arms (e.g. 0-full)")
    ap.add_argument("--run-dir", default=None, type=Path,
                    help="Override RUNS_DIR (e.g. eval/results/agent/qwen7b_swebench)")
    ap.add_argument("--tasks", default=None, type=Path,
                    help="Restrict to task_ids in this jsonl (e.g. swebench_30.jsonl) "
                         "so a 100-task run also yields a 30-task subset summary.")
    ap.add_argument("--label", default="",
                    help="Suffix for output files: SUMMARY_<label>.md / "
                         "summary_<label>.json (e.g. --label n30). Avoids overwrite.")
    ap.add_argument("--gold-fqns", default=None, type=Path,
                    help="jsonl with tightened gold_fqns (default: swebench_100_fqn.jsonl) "
                         "for the file-vs-function localization table. Skipped if absent.")
    args = ap.parse_args()
    if args.run_dir:
        config.RUNS_DIR = Path(args.run_dir).expanduser().resolve()
    task_ids = _task_ids_from(args.tasks) if args.tasks else None
    aggregate(args.stage, task_ids, args.label, args.gold_fqns)


# `label` is referenced as a transient inside the headline-row loop, so the
# public parameter is named out_label to avoid shadowing it.


if __name__ == "__main__":
    main()
