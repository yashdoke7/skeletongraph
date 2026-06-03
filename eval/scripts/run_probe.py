"""Phase-0 retrieval probe: run several backends over one dataset and print ONE
comparison table (recall / precision / MRR / nDCG + latency). No agent loop, no
agent tokens — this is the cheap go/no-go gate BEFORE building any nemotron arm.

    # fast half (local summaries + baselines) — runs today, any hardware:
    python -m eval.scripts.run_probe --dataset eval/datasets/stage0.jsonl \
        --granularity file --k 1 5 10 20 \
        --backends bm25 dense sg sg-chain summary-bm25-local summary-dense-local

    # llm half (needs Ollama up) — start small with --limit, the cache makes
    # repeats free:
    python -m eval.scripts.run_probe --dataset eval/datasets/stage0.jsonl \
        --granularity file --k 1 5 10 20 --limit 8 \
        --backends summary-bm25-llm summary-dense-llm

Reads the gate like this: does summary-dense lift recall@10 ABOVE bm25/sg-chain
WITHOUT collapsing precision@1? And at what latency? If yes → Phase 1 arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.retrieval_eval import BACKENDS, run_eval


def _g(agg: dict, key: str) -> str:
    v = agg.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-0 retrieval probe")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--backends", nargs="+", required=True,
                    help=f"any of: {', '.join(BACKENDS)}")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--granularity", choices=["fqn", "file"], default="file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval/results/probe"))
    args = ap.parse_args()

    bad = [b for b in args.backends if b not in BACKENDS]
    if bad:
        raise SystemExit(f"unknown backend(s): {bad}\navailable: {list(BACKENDS)}")

    ks = sorted(args.k)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in args.backends:
        print(f"\n=== {b} ===")
        report = run_eval(args.dataset, b, ks, args.granularity, limit=args.limit)
        (args.out_dir / f"probe_{b}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        agg = report["aggregate"]
        rows.append((b, report["n_tasks"], report["wall_seconds"], agg))

    # ── one comparison table, sorted by recall@max-k ──────────────────────────
    rk = f"recall@{ks[-1]}"
    rows.sort(key=lambda r: -(r[3].get(rk) or 0))

    kcols = []
    for k in ks:
        kcols += [f"R@{k}"]
    pcols = [f"P@{ks[0]}", f"P@{ks[-1]}"]
    hdr = (f"{'backend':<22}{'n':>4}" + "".join(f"{c:>7}" for c in kcols)
           + "".join(f"{c:>7}" for c in pcols)
           + f"{'MRR':>7}{'nDCG':>7}{'ms/task':>9}{'wall_s':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    lines = [hdr, "-" * len(hdr)]
    for b, n, wall, agg in rows:
        ms = agg.get("latency_ms")
        line = (f"{b:<22}{n:>4}"
                + "".join(f"{_g(agg, f'recall@{k}'):>7}" for k in ks)
                + f"{_g(agg, f'precision@{ks[0]}'):>7}{_g(agg, f'precision@{ks[-1]}'):>7}"
                + f"{_g(agg, 'mrr'):>7}{_g(agg, f'ndcg@{ks[-1]}'):>7}"
                + f"{(f'{ms:.0f}' if isinstance(ms,(int,float)) else '—'):>9}"
                + f"{wall:>8.0f}")
        print(line)
        lines.append(line)

    out = args.out_dir / "PROBE_SUMMARY.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote per-backend JSON + {out}")


if __name__ == "__main__":
    main()
