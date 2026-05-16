"""Stage 0 orchestrator — the GO/NO-GO run.

Answers one question: does SkeletonGraph's structural retrieval beat flat BM25
(and naive grep) at recalling the files a bug fix actually touches — and at
what context-token cost?

Runs entirely on CPU. No API, no GPU, no Docker.

  python eval/make_dataset.py --n 30      # build dataset first (one-time)
  python eval/run_stage0.py               # then this — keep it in the background

Outputs under eval/results/stage0/:
  retrieval_{grep,bm25,sg}.json   per-backend retrieval metrics
  context_tokens.json             Axis 3 — SG context vs naive full-file read
  kv_cache.csv                    Axis 4 — analytical serving density
  SUMMARY.md                      the GO/NO-GO verdict table
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))          # repo root on path

from eval.retrieval_eval import run_eval           # noqa: E402

DATASET = EVAL_DIR / "datasets" / "stage0.jsonl"
RESULTS = EVAL_DIR / "results" / "stage0"
BACKENDS = ["grep", "bm25", "sg"]
KS = [1, 3, 5, 10]


# ── Axis 3: context tokens ─────────────────────────────────────────────────


def measure_context_tokens(dataset_path: Path) -> dict:
    """SG assembled-context tokens vs naive full-file-read tokens, per task."""
    from skeletongraph.engine import SGEngine
    try:
        from skeletongraph.eval.token_counter import measure_text_tokens as count
    except Exception:
        count = lambda s: len(s) // 4   # noqa: E731  fallback estimate

    tasks = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for i, task in enumerate(tasks, 1):
        repo_path = Path(task["repo_path"])
        # naive baseline: read every gold file in full
        naive = 0
        for gf in task.get("gold_files", []):
            fp = repo_path / gf
            if fp.exists():
                try:
                    naive += count(fp.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
        # SG: assembled context for the query (SLM disabled for determinism)
        sg_tokens = 0
        try:
            engine = SGEngine(project_root=repo_path)
            engine.get_config().enable_slm_fallback = False
            result = engine.query(task["query"][:4000], delivery="cli")
            sg_tokens = getattr(result, "context_tokens", 0) or count(result.context_text)
        except Exception as e:
            print(f"  [{i}/{len(tasks)}] {task['task_id']}: token measure error: {e}")
        ratio = (naive / sg_tokens) if sg_tokens else 0.0
        rows.append({
            "task_id": task["task_id"],
            "sg_context_tokens": sg_tokens,
            "naive_fullfile_tokens": naive,
            "reduction_ratio": round(ratio, 2),
        })
        print(f"  [{i}/{len(tasks)}] {task['task_id']}: "
              f"SG={sg_tokens}  naive={naive}  {ratio:.1f}x")

    import statistics
    valid = [r["reduction_ratio"] for r in rows if r["reduction_ratio"] > 0]
    avg_ratio = round(sum(valid) / len(valid), 2) if valid else 0.0
    median_ratio = round(statistics.median(valid), 2) if valid else 0.0
    return {
        # NOTE: a retrieval MISS yields a tiny SG context -> inflated ratio.
        # Median is robust to that; read alongside recall, not in isolation.
        "avg_reduction_ratio": avg_ratio,
        "median_reduction_ratio": median_ratio,
        "avg_sg_tokens": round(sum(r["sg_context_tokens"] for r in rows) / max(1, len(rows))),
        "avg_naive_tokens": round(sum(r["naive_fullfile_tokens"] for r in rows) / max(1, len(rows))),
        "per_task": rows,
    }


# ── summary ────────────────────────────────────────────────────────────────


def write_summary(reports: dict, ctx: dict) -> None:
    lines = [
        "# Stage 0 — GO/NO-GO Results",
        "",
        f"Dataset: `{DATASET}`  ·  tasks: {reports['sg']['n_tasks']}  ·  "
        f"granularity: file-level recall",
        "",
        "## Axis 2 — Retrieval quality (did we surface the right files?)",
        "",
        "| Backend | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | avg latency |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in BACKENDS:
        a = reports[b]["aggregate"]
        lines.append(
            f"| {b} | {a.get('recall@1',0):.3f} | {a.get('recall@5',0):.3f} | "
            f"{a.get('recall@10',0):.3f} | {a.get('mrr',0):.3f} | "
            f"{a.get('ndcg@10',0):.3f} | {a.get('latency_ms',0):.0f}ms |"
        )

    sg_r10 = reports["sg"]["aggregate"].get("recall@10", 0)
    bm_r10 = reports["bm25"]["aggregate"].get("recall@10", 0)
    delta = sg_r10 - bm_r10
    verdict = (
        "GO — SG beats flat BM25 on file recall"
        if delta > 0.02 else
        "INVESTIGATE — SG not clearly ahead of BM25; check observations before Stage 1"
    )

    lines += [
        "",
        "## Axis 3 — Context efficiency",
        "",
        f"- Avg SG assembled context: **{ctx['avg_sg_tokens']:,} tokens**",
        f"- Avg naive full-file read: **{ctx['avg_naive_tokens']:,} tokens**",
        f"- Median reduction ratio: **{ctx['median_reduction_ratio']}x**  (robust)",
        f"- Mean reduction ratio: {ctx['avg_reduction_ratio']}x  "
        f"(inflated by retrieval-miss tasks — read with recall)",
        "",
        "## Axis 4 — Analytical serving density",
        "",
        "See `kv_cache.csv` (run `python eval/kv_cache.py`).",
        "",
        "## Verdict",
        "",
        f"Recall@10:  SG={sg_r10:.3f}  vs  BM25={bm_r10:.3f}  (delta {delta:+.3f})",
        "",
        f"**{verdict}**",
    ]
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    if not DATASET.exists():
        print(f"No dataset at {DATASET}\nRun:  python eval/make_dataset.py --n 30")
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    reports = {}
    for b in BACKENDS:
        print(f"\n{'=' * 60}\nBackend: {b}\n{'=' * 60}")
        rep = run_eval(DATASET, b, KS, granularity="file")
        (RESULTS / f"retrieval_{b}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        reports[b] = rep
        a = rep["aggregate"]
        print(f"  -> recall@10={a.get('recall@10',0):.3f}  mrr={a.get('mrr',0):.3f}")

    print(f"\n{'=' * 60}\nAxis 3: context tokens\n{'=' * 60}")
    ctx = measure_context_tokens(DATASET)
    (RESULTS / "context_tokens.json").write_text(json.dumps(ctx, indent=2), encoding="utf-8")

    write_summary(reports, ctx)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Stage 0 complete in {elapsed/60:.1f} min")
    print(f"Results: {RESULTS}")
    print(f"Verdict: {RESULTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
