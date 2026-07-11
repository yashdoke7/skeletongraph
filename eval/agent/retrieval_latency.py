"""Retrieval-only latency benchmark — SG (fusion / rerank) vs native ripgrep.

WHY a separate benchmark instead of scraping the agent transcripts: only the
`user`/tool_result stream-json events carry timestamps, and the local search
tools (Grep/Glob/Read) report NO server-side duration — so per-call retrieval
latency cannot be isolated from model-thinking time inside a live run. This
script measures the retrieval CALL directly, warm-cache, median-of-N, so the
number is reproducible and isolates retrieval from agent/model variance.

It is deliberately honest about SG's weak spot: SG retrieval (bm25 enumeration +
structural rerank, and for fusion a dense-embedding leg) is heavier than a single
ripgrep. This reports that gap rather than hiding it — a quality paper states it.

Per task it times ONE retrieval action per arm (the fair unit — one sg_search vs
one grep the agent would issue):
  - rerank : retrieve_rerank(query, repo, k)          (bm25 + SG structural)
  - fusion : retrieve_fusion(query, repo, k)          (3-way RRF incl. dense)
  - native : `rg -l <salient query idents>` over repo (what Claude Code's Grep runs)

Cold vs warm: SG's FIRST call per process pays one-time build costs (bm25 corpus
enumeration ~1s, lean-engine lazy graph build ~10s, and — fusion only — the dense
corpus encode if the .npy cache is absent). Those are index-class, amortised once
per session, so the headline is the WARM median; COLD is reported separately.

Usage:
  python -m eval.agent.retrieval_latency --dataset eval/datasets/stage0.jsonl \
      --limit 5 --iters 5 --k 10
  # repos: prefers the prepared _claude_repos copies (indexed + dense-warm),
  # falls back to the dataset's repo_path.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .run_agent import load_tasks
from .run_claude_code import _copies_root, ARM_SG, ARM_FUSION


def _resolve_rg() -> Optional[str]:
    """Find a REAL ripgrep binary. `rg` on PATH here is a Claude Code shim
    function, not the exe, and shutil.which misses the bundled copies — so try an
    explicit override then known bundle locations. Any ripgrep 14.x is
    latency-equivalent to the one Claude Code's Grep uses."""
    import os
    env = os.environ.get("SG_RG_PATH")
    if env and Path(env).is_file():
        return env
    for c in (
        shutil.which("ripgrep"),
        r"C:\Users\ASUS\AppData\Local\OpenAI\Codex\bin\rg.exe",
        r"C:\Users\ASUS\.local\bin\rg.exe",
    ):
        if c and Path(c).is_file():
            return c
    # Last resort: shutil.which('rg') (works on non-shim setups / real CI).
    w = shutil.which("rg")
    return w


_RG = _resolve_rg()

# Where the prepared editable copies live. stage0.jsonl's repo_path points at
# eval/datasets/repos/*, but the Claude-Code copies were prepared under
# swebench-data/_claude_repos — so _copies_root(task) misses them. Default to the
# known root and let --repos-root override.
_DEFAULT_REPOS_ROOT = Path(
    r"C:\Users\ASUS\Desktop\CS\Projects\swebench-data\_claude_repos")


def _repo_for_task(task: dict, repos_root: Optional[Path]) -> Optional[Path]:
    """Prefer a prepared _claude_repos copy (SG index + warm dense cache); fall
    back to the raw dataset clone. Searches the explicit --repos-root, the
    repo_path-derived root, and the known default root."""
    tid = task["task_id"]
    roots = [r for r in (repos_root, _copies_root(task), _DEFAULT_REPOS_ROOT) if r]
    for base in roots:
        base = Path(base)
        for name in (tid, f"{tid}__{ARM_FUSION}", f"{tid}__{ARM_SG}"):
            cand = base / name
            if cand.is_dir() and (cand / ".skeletongraph").is_dir():
                return cand
    rp = Path(task.get("repo_path", ""))
    return rp if rp and rp.is_dir() else None


def _salient_idents(query: str, n: int = 6) -> List[str]:
    """Identifier-like tokens from the query — what a developer/agent would grep.
    De-duped, longest-first (more specific terms first), capped at n."""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")
    stop = {"the", "and", "for", "with", "when", "this", "that", "from", "into",
            "fix", "bug", "issue", "should", "does", "not", "但", "使用"}
    seen, out = set(), []
    for t in sorted(toks, key=lambda s: -len(s)):
        lt = t.lower()
        if lt in stop or lt in seen:
            continue
        seen.add(lt); out.append(t)
    return out[:n]


def _time(fn, iters: int) -> List[float]:
    """Return per-iteration wall times (seconds)."""
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _rg_native(repo: Path, idents: List[str]) -> float:
    """One ripgrep localization pass (files-with-matches) for the query idents —
    the native retrieval action. Returns wall seconds; -1 if rg unavailable."""
    if not _RG or not idents:
        return -1.0
    pattern = "|".join(re.escape(i) for i in idents)
    t0 = time.perf_counter()
    subprocess.run([_RG, "-l", "--no-messages", "-e", pattern, str(repo)],
                   capture_output=True, text=True)
    return time.perf_counter() - t0


def _stats(xs: List[float]) -> dict:
    xs = [x for x in xs if x is not None and x >= 0]
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    p90 = xs_sorted[min(len(xs_sorted) - 1, int(round(0.9 * (len(xs_sorted) - 1))))]
    return {
        "n": len(xs),
        "median_ms": round(1000 * statistics.median(xs), 1),
        "mean_ms": round(1000 * statistics.fmean(xs), 1),
        "p90_ms": round(1000 * p90, 1),
        "min_ms": round(1000 * min(xs), 1),
        "max_ms": round(1000 * max(xs), 1),
    }


def bench_task(task: dict, iters: int, k: int,
               repos_root: Optional[Path]) -> Optional[dict]:
    from skeletongraph.retrieval.fusion import retrieve_rerank, retrieve_fusion
    from skeletongraph.retrieval import dense as _dense

    repo = _repo_for_task(task, repos_root)
    if repo is None:
        print(f"  SKIP {task['task_id']}: no prepared repo found")
        return None
    query = task.get("query", "")
    idents = _salient_idents(query)

    # Build the dense .npy cache FULLY first (no timeout — this is the one-time
    # index-class encode, ~minutes cold on CPU, seconds on GPU). Without it the
    # dense leg hits SG_DENSE_TIMEOUT_S and fusion silently degrades to 2-way, so
    # the "fusion" latency would be a lie. dense_cold captures that one-time cost.
    t0 = time.perf_counter(); _dense.retrieve(query, repo, k)
    dense_cold_ms = round(1000 * (time.perf_counter() - t0), 1)

    # COLD structural: first heuristic_query on a fresh engine pays the graph
    # build (measured ~one-time per process); captured separately.
    cold = {"dense_ms": dense_cold_ms}
    t0 = time.perf_counter(); retrieve_rerank(query, repo, k)
    cold["rerank_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    t0 = time.perf_counter(); retrieve_fusion(query, repo, k)
    cold["fusion_ms"] = round(1000 * (time.perf_counter() - t0), 1)

    # Now everything is warm (bm25 index cached, engine graph built, dense .npy on
    # disk + loaded). Time WARM medians — the steady-state per-call cost.
    rerank_t = _time(lambda: retrieve_rerank(query, repo, k), iters)
    fusion_t = _time(lambda: retrieve_fusion(query, repo, k), iters)
    native_t = [_rg_native(repo, idents) for _ in range(iters)]

    return {
        "task_id": task["task_id"],
        "repo": str(repo),
        "query_idents": idents,
        "k": k,
        "cold": cold,
        "warm": {
            "rerank": _stats(rerank_t),
            "fusion": _stats(fusion_t),
            "native_rg": _stats(native_t),
        },
    }


def _fmt_table(rows: List[dict]) -> str:
    def med(r, arm):
        s = r["warm"][arm]
        return f"{s.get('median_ms','—')}" if s.get("n") else "—"
    out = ["## Retrieval latency (warm median, ms) — one retrieval action per arm",
           "",
           "| Task | rerank | fusion | native `rg` | dense encode (1-time) |",
           "| --- | ---:| ---:| ---:| ---:|"]
    for r in rows:
        out.append(f"| `{r['task_id']}` | {med(r,'rerank')} | {med(r,'fusion')} "
                   f"| {med(r,'native_rg')} | {r['cold'].get('dense_ms','—')} |")
    # aggregate medians across tasks
    def agg(arm):
        vals = [r["warm"][arm]["median_ms"] for r in rows if r["warm"][arm].get("n")]
        return round(statistics.median(vals), 1) if vals else "—"
    out += ["",
            f"**Median across tasks** — rerank {agg('rerank')} ms · "
            f"fusion {agg('fusion')} ms · native `rg` {agg('native_rg')} ms",
            "",
            "_Warm = per-call cost once per-process caches are built (the steady "
            "state inside a session — bm25 index + engine graph cached, dense .npy "
            "loaded). dense encode = one-time cost to embed the whole function "
            "corpus (minutes on CPU, seconds on GPU); it is index-class, built once "
            "per repo before serving, NOT charged per retrieval. native `rg` is a "
            "single ripgrep localization pass over the repo._"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="eval/datasets/stage0.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--iters", type=int, default=5, help="warm iterations per arm")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repos-root", default="",
                    help="dir holding the prepared _claude_repos copies "
                         "(default: the known swebench-data/_claude_repos root)")
    ap.add_argument("--out", default="eval/results/agent/retrieval_latency.json")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.dataset))
    if args.limit:
        tasks = tasks[:args.limit]

    repos_root = Path(args.repos_root) if args.repos_root else None
    print(f"ripgrep: {_RG or 'NOT FOUND — native latency omitted'}")

    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['task_id']} ...", flush=True)
        try:
            r = bench_task(task, args.iters, args.k, repos_root)
        except Exception as e:
            print(f"  ERROR {task['task_id']}: {e}")
            continue
        if r:
            rows.append(r)
            w = r["warm"]
            print(f"    warm median ms — rerank {w['rerank'].get('median_ms','—')} "
                  f"| fusion {w['fusion'].get('median_ms','—')} "
                  f"| native_rg {w['native_rg'].get('median_ms','—')} "
                  f"| (fusion cold {r['cold'].get('fusion_ms')} ms)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"iters": args.iters, "k": args.k, "rows": rows},
                              indent=2), encoding="utf-8")
    md = _fmt_table(rows)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nWrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
