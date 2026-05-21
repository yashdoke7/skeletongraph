"""Intrinsic retrieval evaluation harness.

Measures how well a retrieval backend recalls the functions/files that a
gold patch actually touches — independent of any downstream agent.

This is Axis 2 of the eval plan. It runs entirely on CPU (no GPU, no API).

Dataset format — one JSON object per line (.jsonl):
    {
      "task_id": "django__django-11099",
      "repo_path": "eval/datasets/repos/django__django-11099",
      "query": "<issue title + body>",
      "gold_fqns":  ["path/file.py::Class.method", ...],   # from the gold patch
      "gold_files": ["path/file.py", ...]                   # fallback granularity
    }

Build such a dataset from SWE-bench gold patches with `make_retrieval_dataset.py`
(see eval/README.md §3.2), or from ContextBench's annotated gold contexts.

Run:
    python eval/retrieval_eval.py --dataset eval/datasets/swebench_retrieval.jsonl \
        --backend sg --k 5 10 20 --out eval/results/retrieval_sg.json

Backends: sg | bm25 | dense | grep   (see eval/backends/)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence


# ── Metrics ───────────────────────────────────────────────────────────────


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k == 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    gold_set = set(gold)
    hit = sum(1 for r in top if r in gold_set)
    return hit / min(k, len(top))


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the FIRST gold hit."""
    gold_set = set(gold)
    for i, r in enumerate(retrieved, 1):
        if r in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    gold_set = set(gold)
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Backend protocol ──────────────────────────────────────────────────────
# A backend is: (query: str, repo_path: Path, top_n: int) -> List[str]
# It must return a ranked list of FQNs (or file paths, matched against gold).

RetrieverFn = Callable[[str, Path, int], List[str]]


def backend_sg(query: str, repo_path: Path, top_n: int) -> List[str]:
    """SkeletonGraph knowledge-aware structural retrieval."""
    from skeletongraph.engine import SGEngine
    engine = SGEngine(project_root=repo_path)          # auto-builds index if missing
    result = engine.heuristic_query(query, top_n=top_n)
    return [c.skeleton.fqn for c in result.candidates]


def backend_bm25(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Flat BM25 over function bodies — no graph, no structure.

    Implemented by reusing SG's BM25 index but disabling graph expansion
    and centrality re-ranking. See eval/backends/bm25_flat.py.
    """
    from eval.backends.bm25_flat import retrieve as bm25_retrieve
    return bm25_retrieve(query, repo_path, top_n)


def backend_dense(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Dense embedding retrieval (code embedding model, cosine similarity)."""
    from eval.backends.dense import retrieve as dense_retrieve
    return dense_retrieve(query, repo_path, top_n)


def backend_hybrid(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Strong baseline: BM25 ∪ dense → cross-encoder rerank (the deployed RAG
    default). Returns file paths. See eval/backends/hybrid.py."""
    from eval.backends.hybrid import retrieve as hybrid_retrieve
    return hybrid_retrieve(query, repo_path, top_n)


def backend_grep(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Keyword-grep simulation — the naive-agent baseline."""
    from eval.backends.grep_sim import retrieve as grep_retrieve
    return grep_retrieve(query, repo_path, top_n)


BACKENDS: Dict[str, RetrieverFn] = {
    "sg": backend_sg,
    "bm25": backend_bm25,
    "dense": backend_dense,
    "hybrid": backend_hybrid,
    "grep": backend_grep,
}


# ── Eval loop ─────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> List[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def _normalize(items: Sequence[str], granularity: str) -> List[str]:
    """Reduce FQNs to file granularity if the dataset only has gold_files."""
    if granularity == "file":
        return [it.split("::")[0] for it in items]
    return list(items)


def run_eval(
    dataset_path: Path,
    backend: str,
    ks: List[int],
    granularity: str = "fqn",
) -> dict:
    tasks = load_dataset(dataset_path)
    retriever = BACKENDS[backend]
    max_k = max(ks)
    # FQN-returning backends (sg, bm25) cluster candidates within files, so
    # fetching exactly max_k FQNs can collapse to far fewer unique files after
    # normalization. Over-fetch by 5× so the k cutoff is applied in file-space
    # (where recall is measured) rather than FQN-space. grep already returns
    # files, so 5× just gives it a larger candidate pool — no harm.
    fetch_n = max(50, max_k * 5)

    per_task = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        repo_path = Path(task["repo_path"])
        # File granularity uses exact gold_files; FQN granularity uses gold_fqns
        # (best-effort, parsed from diff hunk headers).
        if granularity == "file":
            gold = task.get("gold_files") or _normalize(task.get("gold_fqns") or [], "file")
            gold = list(dict.fromkeys(gold))
        else:
            gold = list(task.get("gold_fqns") or [])

        t_task = time.time()
        try:
            retrieved = retriever(task["query"], repo_path, fetch_n)
        except Exception as e:
            print(f"  [{i}/{len(tasks)}] {task['task_id']}: ERROR {e}")
            retrieved = []
        latency_ms = (time.time() - t_task) * 1000
        retrieved = _normalize(retrieved, granularity)
        # Dedup preserving rank order. File-normalization collapses multiple FQNs
        # from one file; without dedup nDCG/precision can exceed 1.0.
        # After dedup the list may be longer than max_k — metric functions slice
        # at k themselves, so this is correct (k is in file-space, not FQN-space).
        retrieved = list(dict.fromkeys(retrieved))

        row = {
            "task_id": task["task_id"],
            "n_gold": len(gold),
            "n_retrieved": len(retrieved),
            "latency_ms": round(latency_ms, 1),
            "mrr": round(mrr(retrieved, gold), 4),
        }
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(retrieved, gold, k), 4)
            row[f"precision@{k}"] = round(precision_at_k(retrieved, gold, k), 4)
            row[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, gold, k), 4)
        per_task.append(row)
        print(f"  [{i}/{len(tasks)}] {task['task_id']}: "
              f"R@10={row.get('recall@10', 0):.2f} MRR={row['mrr']:.2f}")

    # Aggregate (macro-average)
    agg: Dict[str, float] = {}
    metric_keys = [k for k in per_task[0] if k not in ("task_id", "n_gold", "n_retrieved")]
    for mk in metric_keys:
        agg[mk] = round(sum(r[mk] for r in per_task) / len(per_task), 4)

    return {
        "backend": backend,
        "dataset": str(dataset_path),
        "granularity": granularity,
        "n_tasks": len(tasks),
        "ks": ks,
        "wall_seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "per_task": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intrinsic retrieval eval")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--granularity", choices=["fqn", "file"], default="fqn")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"Running {args.backend} on {args.dataset} ...")
    report = run_eval(args.dataset, args.backend, sorted(args.k), args.granularity)

    print("\n=== AGGREGATE ===")
    for k, v in report["aggregate"].items():
        print(f"  {k:<16} {v}")

    out = args.out or Path(f"eval/results/retrieval_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
