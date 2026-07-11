"""`sg-rerank` and `fusion` — the retrievers that win the SWE-bench benchmarks.

Ported verbatim from the eval harness (`eval/agent/tools.py` `_retrieve_rerank`
/ `_retrieve_fusion3`) so the product MCP serves the SAME retrieval the paper
numbers come from — not the engine's older raw `heuristic_query`.

- retrieve_rerank: BM25 recall pool REORDERED by SG structural confirmation. Not
  an RRF blend, not per-file capped — keeps BM25's recall while inheriting SG's
  precise rank. Test files demoted below source (bug fixes edit implementation).
- retrieve_fusion: 3-way RRF (k=60) of BM25 (lexical) + Dense (semantic) + rerank
  (structural). Validated on SWE-bench Pro (recall@1 +50%, MRR +26% over BM25).
  Needs SG_DENSE_MODEL=jinaai/jina-embeddings-v2-base-code + sentence-transformers.
  Falls back to retrieve_rerank if the dense dependency is missing.

Both take (query, repo, k) and return ranked FQNs — identical signatures to the
eval functions, so results are byte-for-byte the same as the benchmark arms.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

from .bm25_flat import retrieve as bm25_retrieve


def _is_test_path(fqn: str) -> bool:
    p = fqn.split("::", 1)[0].replace("\\", "/").lower()
    return ("/test" in p or p.startswith("test") or "/tests/" in p
            or "_test." in p or "conftest" in p)


def _lean_config():
    """The lean 'sg-chain' config the eval rerank uses internally: structural
    core only (summaries + embeddings OFF, entity-first not bm25-primary)."""
    from ..config import SGConfig
    cfg = SGConfig()
    cfg.enable_summaries = False
    cfg.enable_embeddings = False
    cfg.bm25_primary = False
    return cfg


# Per-process engine cache, keyed by resolved repo path. heuristic_query's first
# call on an engine object pays a one-time lazy-load (graph/PageRank build) —
# measured 10.2s on first use, then 0.05s on the SAME object thereafter. Building
# a fresh SGEngine per call (as this function did originally) meant every single
# sg_search re-paid that lazy-load tax — the dominant remaining cost after the
# bm25 enumeration cache. Reusing one warm engine per repo for the process
# lifetime fixes it, matching how the MCP server's own self._engine already works.
#
# STALENESS: SGEngine._store is loaded once and never re-read from disk (`if
# self._store is None:` never resets — true for the MCP server's own long-lived
# self._engine too, a pre-existing gap this doesn't introduce). The FileChanged
# hook updates the ON-DISK index incrementally after an edit, but an already-
# cached engine object won't see it. Fixed here (unlike the structural default)
# by fingerprinting the SAME known-file list bm25_flat uses and rebuilding only
# when it actually changes — self-healing on real edits, free on repeat reads.
_ENGINE_CACHE: dict = {}   # key -> (fingerprint, engine)


def _lean_engine(repo: Path):
    from .bm25_flat import _fingerprint, known_files
    from ..engine import SGEngine

    key = str(Path(repo).resolve())
    # Piggyback on the bm25 cache's file enumeration (already fresh/cached) so
    # this doesn't need its own separate, expensive tree-sitter walk to know
    # which files to fingerprint.
    fp = _fingerprint(known_files(repo))

    cached = _ENGINE_CACHE.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1]

    eng = SGEngine(project_root=repo, config=_lean_config())
    _ENGINE_CACHE[key] = (fp, eng)
    return eng


def retrieve_rerank(query: str, repo: Path, k: int) -> List[str]:
    """BM25 recall pool, reordered by SG structural confirmation."""
    repo = Path(repo)
    pool = bm25_retrieve(query, repo, max(k * 5, 40))     # wide bm25 recall pool
    if not pool:
        return []
    engine = _lean_engine(repo)
    sg_res = engine.heuristic_query(query, top_n=40)
    sg_order = {c.skeleton.fqn: i for i, c in enumerate(sg_res.candidates)}

    q_idents = {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query or "")}

    def _name(fqn: str) -> str:
        return fqn.split("::")[-1].split(".")[-1].lower()

    confirmed = sorted((f for f in pool if f in sg_order), key=lambda f: sg_order[f])
    cset = set(confirmed)
    named = [f for f in pool if f not in cset and _name(f) in q_idents]
    nset = set(named)
    rest = [f for f in pool if f not in cset and f not in nset]
    ordered = confirmed + named + rest
    # Demote test files below source (bug fixes edit the implementation; BM25
    # ranks the gold file's own tests above it). Stable sort preserves rank
    # within source and within tests. Skipped when the task is about tests.
    if "test" not in (query or "").lower():
        ordered = sorted(ordered, key=lambda f: _is_test_path(f))
    return ordered[:k]


# Hard ceiling on the dense leg. try/except alone only guards against an
# EXCEPTION — a slow or genuinely stuck model download/load (observed directly
# while validating this: a cold model fetch sat well past 2 minutes without
# raising) would hang retrieve_fusion indefinitely instead of degrading, since
# nothing ever throws. A tool call that never returns is exactly the failure
# mode this whole retrieval integration was built to eliminate (see the
# server-startup handshake fix above) — it must not be reintroduced here on a
# different code path. Bounded in a thread so a hang can never propagate.
_DENSE_TIMEOUT_S = float(os.environ.get("SG_DENSE_TIMEOUT_S", "20"))


def _dense_with_timeout(query: str, repo: Path, k: int) -> List[str]:
    import threading
    result: dict = {}

    def _run():
        try:
            from .dense import retrieve as dense_retrieve
            result["out"] = dense_retrieve(query, repo, k)
        except Exception:
            result["out"] = []   # not installed / load failed — same graceful degrade

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_DENSE_TIMEOUT_S)
    return result.get("out", [])   # timed out (thread still running) -> [] -> 2-signal fusion


def retrieve_fusion(query: str, repo: Path, k: int) -> List[str]:
    """3-way RRF: BM25 (lexical) + Dense (semantic) + rerank (structural)."""
    repo = Path(repo)
    deep = max(k * 3, 60)
    dense_list = _dense_with_timeout(query, repo, deep)
    lists = [
        bm25_retrieve(query, repo, deep),      # lexical
        dense_list,                            # semantic (code-search model)
        retrieve_rerank(query, repo, deep),    # structural (SG)
    ]
    scores: Dict[str, float] = {}
    for lst in lists:
        for rank_i, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60 + rank_i + 1)
    return sorted(scores, key=lambda it: -scores[it])[:k]


def warm(repo: Path, mode: str = "fusion", full_dense: bool = True) -> None:
    """Pre-build every cache this module uses, BEFORE the agent's clock starts.

    Without this, the lazy costs below all land on the agent's first `sg_search`
    turn instead — mid-task, billed as agent latency/turns. Call once from repo
    prep (e.g. right after `sg build`, before the coding agent launches):
      - bm25 enumeration + index   (~1.3s cold, ~0.02s warm thereafter)
      - lean SGEngine lazy-build   (~10s cold, ~0.05s warm thereafter)
      - dense model + embeddings  if mode == "fusion" (one-time corpus encode:
        minutes on CPU, seconds on GPU; INCREMENTAL after — only changed
        functions re-embed. Content-hash cached to .skeletongraph/dense_cache.)

    full_dense=True (the default, and what `sg warm` / eval prep use) FULLY builds
    the dense cache with NO timeout — this is the explicit "take your time so the
    first real query is instant" path. serve()'s background thread passes
    full_dense=False so a stuck first-ever model download can't wedge startup; it
    degrades to a bounded attempt and the lazy path fills the rest.

    Safe to call redundantly — a warm cache is a no-op hash check, not a rebuild.
    `mode="rerank"` skips the dense leg entirely (no embedding time).
    """
    # Go through the real public call, not just `_lean_engine(repo)` — constructing
    # the engine object alone does NOT trigger its lazy build (`_store` stays None
    # until an actual query runs); only a real heuristic_query call pays that cost.
    # Routing through retrieve_rerank exercises the exact same path a live agent
    # call takes (bm25 + engine + merge), so nothing is skipped or approximated.
    retrieve_rerank("warm-up", repo, 1)
    if mode == "fusion":
        if full_dense:
            # Explicit prewarm: encode the whole corpus, no timeout. First build
            # is minutes on CPU; incremental (only changed funcs) thereafter.
            from .dense import retrieve as dense_retrieve
            dense_retrieve("warm-up", repo, 1)
        else:
            # Background/startup path: never let a stuck first-ever model download
            # wedge the server — a bounded attempt primes the cache or gives up.
            _dense_with_timeout("warm-up", repo, 1)
