"""Every number in the v2 paper's new sections, from the run records and transcripts.

    python -m eval.scripts.paper_v2_analysis            # print all tables
    python -m eval.scripts.paper_v2_analysis --json eval/results_summary.json

Sections it feeds:
  settings  - the same retriever in each agent setting: tokens, turns, cost, pass@1
  funnel    - how far the localisation gain travels toward a verified fix
  retrieval - first-search hit and rank-1 on the tasks where the retriever was used
  release   - what the native agent did differently under Claude Code 2.1.278
  addition  - why the retriever costs more on 2.1.278: native vs retriever, same release

Token definitions differ by harness and are NOT interchangeable:
  Claude Code / Codex  total_input_tokens = fresh + cache-read (+ cache-creation)
  ReAct loop           billed_input = sum of prompt_tokens over turns, which already
                       includes cached tokens (cached_input is a subset, never added)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import statistics as st
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "eval" / "results" / "agent"
EXEC = re.compile(r"\b(python|pytest|py\.test|pip|tox|nosetests|runtests)\b", re.I)

# (label, baseline tag, baseline arm, retriever tag, retriever arm, harness kind)
# The 2.1.278 row pairs claude_v8 native with claude_v7_rep2 sg-fusion-plain: both
# ran on 2.1.278, recorded under different tags a day apart.
SETTINGS = [
    ("ReAct loop (v4)", "nemotron_v4", "none", "nemotron_v4", "fusion", "react"),
    ("ReAct loop (v2)", "nemotron_v2", "none", "nemotron_v2", "sg-rerank", "react"),
    ("Claude Code 2.1.206-211", "claude_v7", "native", "claude_v7", "sg-fusion", "claude"),
    ("Claude Code 2.1.274", "claude_v7_rep2", "native", "claude_v7_rep2", "sg-fusion", "claude"),
    ("Claude Code 2.1.278", "claude_v8", "native", "claude_v7_rep2", "sg-fusion-plain", "claude"),
    ("Claude Code, SWE-rebench", "claude_rebench_v1", "native", "claude_rebench_v1", "sg-fusion", "claude"),
    ("Claude Code, SWE-rebench prose", "claude_rebench_prose_v1", "native",
     "claude_rebench_prose_v1", "sg-fusion", "claude"),
    ("Codex CLI 0.155.0", "codex_v1", "codex-native", "codex_v1", "codex-sg-plain", "codex"),
    ("Codex CLI 0.155.0, shipped integration", "codex_v1", "codex-native", "codex_v1",
     "codex-sg-fusion", "codex"),
]

_CACHE: dict = {}


def load(tag: str) -> dict:
    if tag not in _CACHE:
        out: dict = defaultdict(dict)
        for f in glob.glob(str(ROOT / tag / "*.json")):
            if os.path.basename(f) == "summary.json":
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict) and "arm" in d and "resolved" in d:
                out[d["arm"]][d["task_id"]] = d
        _CACHE[tag] = out
    return _CACHE[tag]


def tokens(d: dict) -> float:
    v = d.get("total_input_tokens")
    if isinstance(v, (int, float)) and v:
        return v
    return d.get("billed_input") or 0          # ReAct: already includes cached


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil((len(xs) - 1) * p))]


def boot_ratio(pairs, seed=42, n=10000):
    rnd = random.Random(seed)
    k = len(pairs)
    out = []
    for _ in range(n):
        s = [pairs[rnd.randrange(k)] for _ in range(k)]
        a = st.mean(x for x, _ in s)
        b = st.mean(y for _, y in s)
        out.append((b - a) / a * 100 if a else 0.0)
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n * 2)


def norm(p):
    return str(p).replace("\\", "/").lower()


# ── "did the agent have code from a gold file in front of it" ──────────────────
def saw_gold(tag, d, kind):
    gold = [norm(g) for g in d.get("gold_files") or []]
    if not gold:
        return None
    if kind == "react":
        for f in d.get("files_read") or []:
            path = f.get("path", "") if isinstance(f, dict) else f
            if (isinstance(f, dict) and f.get("was_gold")) or any(norm(path).endswith(g) for g in gold):
                return True
        return False
    sub = "_codex_transcripts" if kind == "codex" else "_claude_transcripts"
    t = ROOT / tag / sub / (d["run_id"] + ".jsonl")
    if not t.exists():
        return None
    for line in open(t, encoding="utf-8"):
        low = line.replace("\\\\", "/").replace("\\", "/").lower()
        if not any(g in low for g in gold):
            continue
        o = json.loads(line)
        if kind == "codex":
            if o.get("type") == "item.completed":
                return True
            continue
        for b in (o.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and b.get("name") == "Read":
                if any(norm((b.get("input") or {}).get("file_path", "")).endswith(g) for g in gold):
                    return True
            if b.get("type") == "tool_result":
                txt = json.dumps(b.get("content")).replace("\\\\", "/").replace("\\", "/").lower()
                if any(g in txt for g in gold):
                    return True
    return False


STAGE_NAMES = ["first search hits gold", "reads gold code", "edits gold file",
               "edits gold function", "makes a patch", "solved"]


def funnel_stages(tag, d, kind):
    """Per-run funnel. "reads gold code" requires code from a gold file to enter the
    context (a read, a content search, a printed file, or a retriever payload); a
    path-only listing does not count. "edits gold function" maps the patch's changed
    lines onto the base-commit file's functions (see funnel_strict)."""
    from eval.scripts import funnel_strict as fs      # lazy: fs imports this module
    if kind == "react":
        code = saw_gold(tag, d, kind)                  # ReAct read_file returns file content
    else:
        turn = (fs.claude_content_turn if kind == "claude" else fs.codex_content_turn)(tag, d)
        code = turn is not None
    return [bool(d.get("retrieval_hit")), code, bool(d.get("edited_gold_file")),
            fs.edited_gold_function(d), bool((d.get("model_patch") or "").strip()),
            bool(d.get("resolved"))]


def first_code_turn(tag, d, kind):
    from eval.scripts import funnel_strict as fs
    if kind == "claude":
        return fs.claude_content_turn(tag, d)
    if kind == "codex":
        return fs.codex_content_turn(tag, d)
    return None


# ── what the agent did, from a Claude Code transcript ─────────────────────────
def behaviour(path: Path) -> dict:
    names, calls, chars = {}, [], defaultdict(int)
    turn, first_edit = 0, None
    for line in open(path, encoding="utf-8"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        content = (o.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        if o.get("type") == "assistant":
            used = False
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    used = True
                    names[b.get("id")] = b.get("name")
                    calls.append((turn, b.get("name"), b.get("input") or {}))
                    if first_edit is None and b.get("name") in ("Edit", "Write", "MultiEdit"):
                        first_edit = turn
            turn += used
        elif o.get("type") == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    txt = c if isinstance(c, str) else "".join(
                        x.get("text", "") for x in (c or []) if isinstance(x, dict))
                    chars[names.get(b.get("tool_use_id"), "?")] += len(txt)
    cnt = Counter(n for _, n, _ in calls)
    greps = [i for _, n, i in calls if n == "Grep"]
    bashes = [(t, i.get("command", "")) for t, n, i in calls if n == "Bash"]
    return {
        "turns": turn,
        "before_edit": first_edit,
        "after_edit": None if first_edit is None else turn - first_edit,
        "reads": cnt.get("Read", 0),
        "searches": cnt.get("Grep", 0) + cnt.get("Glob", 0),
        "file_list_searches": sum(1 for i in greps
                                  if i.get("output_mode") in (None, "files_with_matches")),
        "code_runs": sum(1 for _, c in bashes if EXEC.search(c)),
        "ran_code_after_edit": first_edit is not None and any(
            EXEC.search(c) for t, c in bashes if t > first_edit),
        "tool_lookups": cnt.get("ToolSearch", 0),
        "retriever_calls": sum(v for k, v in cnt.items() if "skeletongraph" in k),
        "read_chars": chars.get("Read", 0),
        "all_tool_chars": sum(chars.values()),
    }


def behaviour_panel(tag, arm):
    out = {}
    for tid, d in load(tag)[arm].items():
        t = ROOT / tag / "_claude_transcripts" / (d["run_id"] + ".jsonl")
        if t.exists():
            b = behaviour(t)
            b["output_tokens"] = d.get("billed_output") or 0
            out[tid] = b
    return out


def summarise_behaviour(A, B):
    common = sorted(set(A) & set(B))
    keys = ["turns", "before_edit", "after_edit", "reads", "read_chars", "searches",
            "file_list_searches", "code_runs", "tool_lookups", "retriever_calls",
            "all_tool_chars", "output_tokens"]
    rows = {}
    for k in keys:
        a = [A[t][k] for t in common if A[t][k] is not None]
        b = [B[t][k] for t in common if B[t][k] is not None]
        rows[k] = (st.mean(a), st.mean(b))
    rows["ran_code_after_edit"] = (sum(A[t]["ran_code_after_edit"] for t in common),
                                   sum(B[t]["ran_code_after_edit"] for t in common))
    return len(common), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    report: dict = {}

    print("== settings: the same retriever in each agent setting (paired) ==")
    for lab, ta, aa, tb, ab, kind in SETTINGS:
        A, B = load(ta)[aa], load(tb)[ab]
        c = sorted(set(A) & set(B))
        pt = [(tokens(A[t]), tokens(B[t])) for t in c]
        m0, m1 = st.mean(x for x, _ in pt), st.mean(y for _, y in pt)
        lo, hi = boot_ratio(pt)
        turns0 = st.mean(A[t].get("n_turns") or 0 for t in c)
        turns1 = st.mean(B[t].get("n_turns") or 0 for t in c)
        usd0 = st.mean(A[t].get("imputed_cost") or 0 for t in c)
        usd1 = st.mean(B[t].get("imputed_cost") or 0 for t in c)
        s0 = sum(A[t]["resolved"] for t in c)
        s1 = sum(B[t]["resolved"] for t in c)
        bo = sum(1 for t in c if A[t]["resolved"] and not B[t]["resolved"])
        ro = sum(1 for t in c if B[t]["resolved"] and not A[t]["resolved"])
        a_tok, b_tok = [x for x, _ in pt], [y for _, y in pt]
        row = {"n": len(c), "base_tokens": m0, "tokens_delta": m1 - m0,
               "tokens_pct": (m1 - m0) / m0 * 100, "ci": [lo, hi],
               "turns_delta": turns1 - turns0, "usd_delta": usd1 - usd0,
               "p50_pct": (pct(b_tok, .5) - pct(a_tok, .5)) / pct(a_tok, .5) * 100,
               "p90_pct": (pct(b_tok, .9) - pct(a_tok, .9)) / pct(a_tok, .9) * 100,
               "cheaper_tasks": sum(1 for x, y in pt if y < x),
               "solved": [s0, s1], "discordant": [bo, ro], "mcnemar_p": mcnemar(bo, ro)}
        report.setdefault("settings", {})[lab] = row
        print(f"{lab:<40} n={len(c):>3} tokens {m0:>10,.0f} {m1 - m0:>+10,.0f} "
              f"({row['tokens_pct']:+.1f}% [{lo:+.0f},{hi:+.0f}]) turns {row['turns_delta']:+.1f} "
              f"${row['usd_delta']:+.3f} | p50 {row['p50_pct']:+.0f}% p90 {row['p90_pct']:+.0f}% "
              f"| solved {s0}->{s1} ({bo}/{ro}, p={row['mcnemar_p']:.3f})")

    print("\n== funnel ==")
    names = STAGE_NAMES
    SOLVED, FILE, FUNC = 5, 2, 3
    for lab, ta, aa, tb, ab, kind in SETTINGS:
        A, B = load(ta)[aa], load(tb)[ab]
        c = sorted(set(A) & set(B))
        SA = {t: funnel_stages(ta, A[t], kind) for t in c}
        SB = {t: funnel_stages(tb, B[t], kind) for t in c}
        stages = []
        for i, nm in enumerate(names):
            ok = [t for t in c if SA[t][i] is not None and SB[t][i] is not None]
            a = sum(bool(SA[t][i]) for t in ok)
            b = sum(bool(SB[t][i]) for t in ok)
            lost = sum(1 for t in ok if SA[t][i] and not SB[t][i])
            gain = sum(1 for t in ok if SB[t][i] and not SA[t][i])
            stages.append((nm, 100 * a / max(1, len(ok)), 100 * b / max(1, len(ok)),
                           len(ok), mcnemar(lost, gain)))
        newly = [t for t in c if not SA[t][FILE] and SB[t][FILE]]
        both = [t for t in c if SA[t][FILE] and SB[t][FILE]]
        disc = [t for t in c if SA[t][SOLVED] != SB[t][SOLVED]]
        fn_ok = [t for t in disc if SA[t][FUNC] is not None and SB[t][FUNC] is not None]
        row = {"n": len(c), "stages": stages, "newly_edit_gold": len(newly),
               "newly_edit_gold_newly_solved": sum(1 for t in newly if SB[t][SOLVED] and not SA[t][SOLVED]),
               "both_edit_gold": len(both), "discordant": len(disc),
               "discordant_both_edited": sum(1 for t in disc if t in both),
               "discordant_fn_resolved": len(fn_ok),
               "discordant_both_edited_function": sum(1 for t in fn_ok if SA[t][FUNC] and SB[t][FUNC]),
               "net_solve_both_edited": sum(1 for t in both if SB[t][SOLVED] and not SA[t][SOLVED])
               - sum(1 for t in both if SA[t][SOLVED] and not SB[t][SOLVED])}
        if kind != "react":
            ta_ = {t: first_code_turn(ta, A[t], kind) for t in c}
            tb_ = {t: first_code_turn(tb, B[t], kind) for t in c}
            reached = [t for t in c if ta_[t] is not None and tb_[t] is not None]
            row["first_code_turn"] = {
                "n": len(reached),
                "mean": [st.mean(ta_[t] for t in reached), st.mean(tb_[t] for t in reached)],
                "median": [st.median(ta_[t] for t in reached), st.median(tb_[t] for t in reached)],
                "baseline_on_first_call": sum(1 for t in c if ta_[t] == (0 if kind == "claude" else 1)),
                "baseline_within_three": sum(1 for t in c if ta_[t] is not None
                                             and ta_[t] < (3 if kind == "claude" else 4)),
            }
        report.setdefault("funnel", {})[lab] = row
        print(f"{lab}  (n={len(c)})")
        for nm, a, b, n_ok, p in stages:
            print(f"   {nm:<24}{a:>5.0f}% -> {b:>4.0f}%   (n={n_ok}, McNemar p={p:.3f})")
        print(f"   outcome differs on {row['discordant']} tasks; both edited a gold file on "
              f"{row['discordant_both_edited']}, a gold function on "
              f"{row['discordant_both_edited_function']} of {len(fn_ok)} resolvable")
        if "first_code_turn" in row:
            f = row["first_code_turn"]
            print(f"   first gold code at tool turn (both reached, n={f['n']}): mean {f['mean'][0]:.1f} -> "
                  f"{f['mean'][1]:.1f}, median {f['median'][0]:g} -> {f['median'][1]:g}; baseline on its "
                  f"first call {f['baseline_on_first_call']}, within three {f['baseline_within_three']}")

    print("\n== retrieval on the tasks where the retriever was used ==")
    for lab, ta, aa, tb, ab, kind in SETTINGS:
        if kind == "react":
            continue
        A, B = load(ta)[aa], load(tb)[ab]
        c = sorted(set(A) & set(B))
        inv = [t for t in c if any("sg_search" in k and v
                                   for k, v in (B[t].get("tool_counts") or {}).items())]
        f = lambda D, pred: 100 * sum(1 for t in inv if pred(D[t])) / len(inv)
        row = {"invoked": len(inv), "n": len(c),
               "hit": [f(A, lambda d: d.get("retrieval_hit")), f(B, lambda d: d.get("retrieval_hit"))],
               "rank1": [f(A, lambda d: d.get("retrieval_rank") == 1),
                         f(B, lambda d: d.get("retrieval_rank") == 1)]}
        report.setdefault("retrieval", {})[lab] = row
        print(f"{lab:<40} invoked {len(inv)}/{len(c)}  hit {row['hit'][0]:.0f}% -> {row['hit'][1]:.0f}%"
              f"  rank-1 {row['rank1'][0]:.0f}% -> {row['rank1'][1]:.0f}%")

    for key, (ta, aa, tb, ab, title) in {
            "release": ("claude_v7_rep2", "native", "claude_v8", "native",
                        "native agent, Claude Code 2.1.274 -> 2.1.278"),
            "addition": ("claude_v8", "native", "claude_v7_rep2", "sg-fusion-plain",
                         "Claude Code 2.1.278, native -> with retriever")}.items():
        n, rows = summarise_behaviour(behaviour_panel(ta, aa), behaviour_panel(tb, ab))
        report[key] = {"n": n, "rows": rows}
        print(f"\n== {title} (n={n}, per-task means) ==")
        for k, (a, b) in rows.items():
            print(f"   {k:<22}{a:>12,.1f}{b:>12,.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
