"""Graphify backend — knowledge-graph RAG via tree-sitter entity graph.

Graphify (~52K GitHub stars) builds an entity/relationship knowledge graph
of a codebase using tree-sitter parsing + LLM-driven semantic extraction +
NetworkX + Leiden community clustering. Claimed to outperform pure-vector
RAG on multi-hop code retrieval questions.

  GitHub: https://github.com/safishamsi/graphify
  Install: pip install graphifyy

THIS IS A PLACEHOLDER STUB.  Running the `graphify` arm will fail fast
with a clear NotImplementedError until this is implemented.

Implementation steps:
  1. pip install graphifyy
  2. Read the API surface: https://github.com/safishamsi/graphify
  3. Confirm the index-build call, search call, and result schema.
  4. Replace the NotImplementedError below with real calls.
  5. Test on a single task:
       python -m eval.agent.run_agent --task-id sympy__sympy-24066 --arm graphify
  6. Run the full stage:
       python -m eval.agent.run_stage --stage full --workers 4

Notes:
  - Graphify builds a persistent cache (.graphify/ inside the repo dir).
    This is consistent with how SG caches in .skeletongraph/.
  - Index build is slow on first call (~30-120 s per repo); subsequent
    calls on the same checkout reuse the cache. The eval harness gives each
    run its own isolated workspace (see isolation.py), so every run pays the
    build cost. Consider pre-building if that's a bottleneck.
  - Results are file-level (no FQN yet). Precision/recall are measured at
    file granularity for graphify, matching the aggregate.py metric path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def retrieve(query: str, repo: Path, k: int = 10) -> List[str]:
    """Query the Graphify knowledge-graph index built over `repo`.

    Returns a ranked list of file paths relative to `repo`.
    Raises NotImplementedError until the stub is filled in.
    """
    raise NotImplementedError(
        "graphify backend not yet implemented. "
        "See eval/backends/graphify.py for implementation instructions."
    )
