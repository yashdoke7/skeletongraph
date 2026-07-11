"""Retrieval strategies.

`fusion`/`sg-rerank` (the algorithms that win the SWE-bench benchmarks) are
exposed here so the product MCP/CLI can serve the same retrieval the paper
numbers come from — historically these lived only in the eval harness while the
shipped MCP ran the engine's weaker raw `heuristic_query`.
"""
from .fusion import retrieve_fusion, retrieve_rerank, warm

__all__ = ["retrieve_fusion", "retrieve_rerank", "warm"]
