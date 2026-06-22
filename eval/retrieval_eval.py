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
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

# Silence HuggingFace / sentence-transformers model-load progress bars before
# any transformers/huggingface_hub import (the hybrid + sg backends load models).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Metrics ───────────────────────────────────────────────────────────────


def _bare(fqn: str) -> str:
    """Reduce 'file::Type.method' -> 'file::method' (leaf name). Files (no '::')
    are returned unchanged."""
    if "::" not in fqn:
        return fqn
    f, s = fqn.split("::", 1)
    return f"{f}::{s.split('.')[-1]}"


def _aug(retrieved_window: Sequence[str]) -> set:
    """Window membership set augmented with the BARE form of each retrieved FQN.

    This makes a BARE gold name (`file::Func`, as in SWE-bench Pro where gold is
    100% leaf-names) match a QUALIFIED retrieved FQN (`file::Type.Func`, what SG
    emits for Go/TS/JS/etc.). A QUALIFIED gold (Python/Verified, has a dot) only
    ever equals an exact retrieved FQN — the added bare leaf (no dot) can never
    equal it — so Verified file/function recall is byte-identical. File-granularity
    items have no '::' so _bare is a no-op."""
    out = set(retrieved_window)
    for r in retrieved_window:
        out.add(_bare(r))
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = _aug(retrieved[:k])
    hit = sum(1 for g in gold if g in top)
    return hit / len(gold)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k == 0 or not retrieved:
        return 0.0
    top = retrieved[:k]
    gold_set = set(gold)
    hit = sum(1 for r in top if r in gold_set or _bare(r) in gold_set)
    return hit / min(k, len(top))


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Reciprocal rank of the FIRST gold hit."""
    gold_set = set(gold)
    for i, r in enumerate(retrieved, 1):
        if r in gold_set or _bare(r) in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    gold_set = set(gold)
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in gold_set or _bare(r) in gold_set:
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


def backend_sg_chain(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Current best agent arm — SG+BM25 fusion + graph-path bridging. The bar the
    summary-search probe must clear on recall. Returns FQNs (drops the summary)."""
    from eval.agent.tools import _retrieve_chain
    return [fqn for fqn, _ in _retrieve_chain(query, Path(repo_path), top_n)]


def backend_sg_chain_nopath(query: str, repo_path: Path, top_n: int) -> List[str]:
    """sg-chain without graph-path bridging (plain SG+BM25 fusion)."""
    from eval.agent.tools import _retrieve_chain
    return [fqn for fqn, _ in _retrieve_chain(query, Path(repo_path), top_n,
                                              use_path=False)]


def backend_sg_rerank(query: str, repo_path: Path, top_n: int) -> List[str]:
    """sg-rerank (the paper's method / product default): a wide BM25 recall pool
    reordered by SG structural confirmation. Same logic the agent arm uses, so
    the deterministic retrieval number is directly comparable to Table 1."""
    from eval.agent.tools import _retrieve_rerank
    return _retrieve_rerank(query, Path(repo_path), top_n)


def backend_bm25_dense(query: str, repo_path: Path, top_n: int) -> List[str]:
    """RRF fusion of lexical (BM25) + semantic (dense, code-search model via
    SG_DENSE_MODEL). Tests whether COMBINING beats either alone: BM25 supplies the
    lexical-anchor precision, dense supplies the NL-only recall. RRF is rank-based
    (k=60), so a noisy dense ranking can't sink a strong BM25 hit — the fusion is
    at worst ~BM25. Run with SG_DENSE_MODEL set to a code-search embedder."""
    try:
        from eval.backends.bm25_flat import retrieve as bm25_retrieve
        from eval.backends.dense import retrieve as dense_retrieve
    except Exception:
        from backends.bm25_flat import retrieve as bm25_retrieve
        from backends.dense import retrieve as dense_retrieve
    rp = Path(repo_path)
    deep = max(top_n * 3, 60)
    lists = [bm25_retrieve(query, rp, deep), dense_retrieve(query, rp, deep)]
    scores: dict = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(scores, key=lambda it: -scores[it])[:top_n]


def backend_bm25_dense_sg(query: str, repo_path: Path, top_n: int) -> List[str]:
    """3-way RRF fusion: BM25 (lexical) + Dense (semantic) + SG (structural).

    The thesis: BM25 knows WHAT you want (keyword match), Dense knows the CODE
    (semantic similarity), and SG knows the ARCHITECTURE (graph dependencies,
    PageRank centrality). RRF merges rank-lists safely — a noisy signal from one
    retriever can't override a strong signal from two others.

    Equal-weight RRF (k=60): each retriever contributes 1/(60+rank+1).
    """
    try:
        from eval.backends.bm25_flat import retrieve as bm25_retrieve
        from eval.backends.dense import retrieve as dense_retrieve
    except Exception:
        from backends.bm25_flat import retrieve as bm25_retrieve
        from backends.dense import retrieve as dense_retrieve
    rp = Path(repo_path)
    deep = max(top_n * 3, 60)
    bm25_list = bm25_retrieve(query, rp, deep)
    dense_list = dense_retrieve(query, rp, deep)
    sg_list = backend_sg_rerank(query, rp, deep)
    scores: dict = {}
    for lst in [bm25_list, dense_list, sg_list]:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60 + rank + 1)
    return sorted(scores, key=lambda it: -scores[it])[:top_n]


def backend_bm25_dense_sg_w(query: str, repo_path: Path, top_n: int) -> List[str]:
    """Weighted 3-way RRF: BM25 (1×) + Dense (1×) + SG (2×).

    Same as bm25-dense-sg but gives the structural retriever DOUBLE weight.
    Rationale: SG-rerank already proved strong on its own (recall@10 0.55) and
    its graph-structural signal is orthogonal to lexical/semantic — it should
    get more say in the final ranking. Tests whether the structural signal is
    so valuable it deserves privileged weighting.
    """
    try:
        from eval.backends.bm25_flat import retrieve as bm25_retrieve
        from eval.backends.dense import retrieve as dense_retrieve
    except Exception:
        from backends.bm25_flat import retrieve as bm25_retrieve
        from backends.dense import retrieve as dense_retrieve
    rp = Path(repo_path)
    deep = max(top_n * 3, 60)
    bm25_list = bm25_retrieve(query, rp, deep)
    dense_list = dense_retrieve(query, rp, deep)
    sg_list = backend_sg_rerank(query, rp, deep)
    # Weighted RRF: BM25 and dense at 1×, SG at 2×.
    scores: dict = {}
    weights = [(bm25_list, 1.0), (dense_list, 1.0), (sg_list, 2.0)]
    for lst, w in weights:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + w / (60 + rank + 1)
    return sorted(scores, key=lambda it: -scores[it])[:top_n]


def _summary_backend(source: str, method: str) -> RetrieverFn:
    """Factory: summary-search backend with a fixed (source, method)."""
    def fn(query: str, repo_path: Path, top_n: int) -> List[str]:
        from eval.backends.summary_search import retrieve as summ_retrieve
        return summ_retrieve(query, Path(repo_path), top_n,
                             source=source, method=method)
    return fn


BACKENDS: Dict[str, RetrieverFn] = {
    "sg": backend_sg,
    "bm25": backend_bm25,
    "dense": backend_dense,                  # dense over CODE (control)
    "hybrid": backend_hybrid,
    "grep": backend_grep,
    "sg-rerank": backend_sg_rerank,          # the paper's method (Table 1)
    "bm25-dense": backend_bm25_dense,        # RRF fusion: lexical + semantic
    "bm25-dense-sg": backend_bm25_dense_sg,  # 3-way RRF: lexical + semantic + structural
    "bm25-dense-sg-w": backend_bm25_dense_sg_w,  # weighted 3-way RRF (SG 2×)
    "sg-chain": backend_sg_chain,            # the recall bar to beat
    "sg-chain-nopath": backend_sg_chain_nopath,
    # ── the probe: summaries × {source, matcher} ──────────────────────────────
    "summary-bm25-local": _summary_backend("local", "bm25"),
    "summary-dense-local": _summary_backend("local", "dense"),
    "summary-bm25-llm": _summary_backend("ollama", "bm25"),
    "summary-dense-llm": _summary_backend("ollama", "dense"),
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


_CODE_EXTS = (".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
              ".go", ".rs", ".java", ".cs", ".cpp", ".cxx", ".cc", ".c", ".h",
              ".hpp", ".rb", ".php")


def _is_code_path(item: str) -> bool:
    """True if the gold item is a source file SG can index. Excludes the non-code
    gold (changelogs, .yml/.yaml, .md/.rst, .lock, .json/.toml, .svg, LICENSE)
    that a code retriever correctly never returns and which otherwise deflates
    multi-language recall (the code-only reframe)."""
    p = item.split("::", 1)[0].lower()
    return p.endswith(_CODE_EXTS)


def run_eval(
    dataset_path: Path,
    backend: str,
    ks: List[int],
    granularity: str = "fqn",
    limit: int | None = None,
    code_only: bool = False,
    rebuild: bool = False,
) -> dict:
    tasks = load_dataset(dataset_path)
    if limit:
        tasks = tasks[:limit]      # cap tasks — handy for the slow Ollama probe
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

        if code_only:
            gold = [g for g in gold if _is_code_path(g)]
        # A task with no (code) gold cannot score retrieval — skip it instead of
        # counting a forced 0.0 that deflates the average (empty gold_fqns, or
        # all-non-code gold). This is the honest denominator.
        if not gold:
            continue

        # Force a fresh index: SG's staleness check is hash-based on SOURCE, which
        # is fixed at base_commit — it does NOT notice that the PARSER improved, so
        # an index built before a parser fix is silently reused (measured: NodeBB
        # 462 vs 3285 functions stale-vs-fresh). Rebuild guarantees current parsers.
        if rebuild:
            import shutil as _sh
            _sh.rmtree(repo_path / ".skeletongraph", ignore_errors=True)

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
            "language": task.get("language", "?"),
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
    metric_keys = [k for k in per_task[0]
                   if k not in ("task_id", "language", "n_gold", "n_retrieved")]
    for mk in metric_keys:
        agg[mk] = round(sum(r[mk] for r in per_task) / len(per_task), 4)

    # Per-language breakdown (where multi-language retrieval is strong/weak).
    by_lang: Dict[str, Dict[str, float]] = {}
    langs = sorted({r["language"] for r in per_task})
    for lang in langs:
        rows = [r for r in per_task if r["language"] == lang]
        d = {"n": len(rows)}
        for mk in metric_keys:
            d[mk] = round(sum(r[mk] for r in rows) / len(rows), 4)
        by_lang[lang] = d

    return {
        "backend": backend,
        "dataset": str(dataset_path),
        "granularity": granularity,
        "code_only": code_only,
        "n_tasks": len(per_task),
        "n_skipped": len(tasks) - len(per_task),
        "ks": ks,
        "wall_seconds": round(time.time() - t0, 1),
        "aggregate": agg,
        "by_language": by_lang,
        "per_task": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intrinsic retrieval eval")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--granularity", choices=["fqn", "file"], default="fqn")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to first N tasks (use for the slow Ollama probe)")
    ap.add_argument("--code-only", action="store_true",
                    help="score only code-file gold (drop changelogs/yaml/md/etc.) "
                         "and skip tasks with no code gold — the fair multi-language "
                         "number (esp. SWE-bench Pro).")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete + rebuild each repo's SG index before retrieving "
                         "(REQUIRED after any parser change — stale indexes are "
                         "silently reused otherwise).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print(f"Running {args.backend} on {args.dataset} "
          f"(code_only={args.code_only}, rebuild={args.rebuild}) ...")
    report = run_eval(args.dataset, args.backend, sorted(args.k), args.granularity,
                      limit=args.limit, code_only=args.code_only, rebuild=args.rebuild)

    print(f"\n=== AGGREGATE (n={report['n_tasks']}, "
          f"skipped={report['n_skipped']}) ===")
    for k, v in report["aggregate"].items():
        print(f"  {k:<16} {v}")
    print("\n=== BY LANGUAGE ===")
    for lang, d in report["by_language"].items():
        print(f"  {lang:8} n={d['n']:<3} "
              f"R@10={d.get('recall@10', 0):.3f}  MRR={d.get('mrr', 0):.3f}")

    out = args.out or Path(f"eval/results/retrieval_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
