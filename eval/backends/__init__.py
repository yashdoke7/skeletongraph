"""Retrieval backends for the controlled comparison.

Each backend exposes:  retrieve(query: str, repo_path: Path, top_n: int) -> List[str]
returning a ranked list of FQNs (or file paths) to be matched against gold.

  grep_sim  — naive keyword-grep agent baseline
  bm25_flat — flat BM25 over function bodies (standard RAG, no graph)
  dense     — dense embedding retrieval (Stage 1+)
  sg        — SkeletonGraph (lives in eval/retrieval_eval.py:backend_sg)
"""
