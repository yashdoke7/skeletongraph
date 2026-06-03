"""Summary-search retrieval (Phase 0 probe).

Searches over function SUMMARIES (purpose/intent), not raw code — modeling how a
developer recalls "there's a function that validates HTTP headers" then fetches
it. Targets the lexical gap that BOTH bm25 (code tokens) and structural SG
(symbol names) share: an issue's prose vs the code's identifiers.

Two orthogonal knobs:
  source — how each function's summary is produced
    "local"  : build_local_summary() — zero compute, any hardware. Docstring
               first line if present, else name/params/keywords. Deterministic.
    "ollama" : qwen2.5-coder:1.5b one-liner via local Ollama. FALLS BACK to the
               local summary when Ollama is unreachable or returns nothing, so a
               missing server degrades quality, never crashes. Opt-in (heavier).
  method — how the query is matched against the summaries
    "bm25"   : Okapi BM25 over tokenized summaries (lexical).
    "dense"  : all-MiniLM-L6-v2 cosine similarity (semantic — the point of summaries).

Chunking is identical to bm25_flat (SG tree-sitter enumeration), so the ONLY
variable vs `bm25` is "search code vs search summaries", and vs `dense` (code)
the only variable is "embed code vs embed summary". That makes the probe a clean
attribution: any lift is the summary representation, not the chunker or matcher.

Summaries are content-hashed and cached on disk per repo, so re-runs are cheap
and deterministic; the (slow) Ollama pass is paid once per repo. Dense doc
embeddings are cached by dense.py.

retrieve() returns ranked FQNs — scored by retrieval_eval against gold files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_OLLAMA_MODEL = os.environ.get("SG_SUMMARY_OLLAMA_MODEL", "qwen2.5-coder:1.5b")
_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Safety cap so a single huge repo (e.g. django ~30k functions) can't turn the
# Ollama probe into a multi-hour stall. When exceeded, the LONGEST-body funcs are
# summarized with the LLM and the rest fall back to local — with a loud warning,
# because recall for that repo is then partially local. Raise for a faithful run.
_OLLAMA_MAX = int(os.environ.get("SG_SUMMARY_OLLAMA_MAX", "6000"))


def _ensure_paths() -> None:
    here = Path(__file__).resolve()
    for p in (str(here.parent.parent), str(here.parents[2])):
        if p not in sys.path:
            sys.path.insert(0, p)


def _enumerate(repo: Path) -> List[Tuple[str, object, str]]:
    """(fqn, skeleton, body) for every function — embeddings OFF (cheap, honest)."""
    _ensure_paths()
    from skeletongraph.engine import SGEngine
    from skeletongraph.config import SGConfig

    cfg = SGConfig()
    cfg.enable_embeddings = False
    store = SGEngine(project_root=repo, config=cfg).get_store()

    by_file: Dict[str, list] = {}
    for fqn, sk in store.skeleton_table.items():
        by_file.setdefault(sk.file_path, []).append((fqn, sk))

    out: List[Tuple[str, object, str]] = []
    for rel, items in by_file.items():
        fp = repo / rel
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
        for fqn, sk in items:
            body = ""
            if lines:
                s = max(0, getattr(sk, "line_start", 1) - 1)
                e = min(len(lines), getattr(sk, "line_end", s + 1))
                body = "\n".join(lines[s:e])
            out.append((fqn, sk, body))
    return out


def _summary_docs(repo: Path, source: str) -> Tuple[List[str], List[str]]:
    """Return (fqns, summaries) aligned. Cached by content hash per repo+source."""
    _ensure_paths()
    from skeletongraph.summary.local import build_local_summary

    funcs = _enumerate(repo)
    cache_path = repo / ".skeletongraph" / f"summcache_{source}.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    # ── Ollama availability + budget (only when source == "ollama") ──────────
    ollama_ok = False
    llm_budget = set()           # fqns allowed to use the LLM under the cap
    if source == "ollama":
        from skeletongraph.summary.ollama import is_ollama_available
        ollama_ok = is_ollama_available(_OLLAMA_BASE)
        if not ollama_ok:
            sys.stderr.write(
                f"[summary-search] Ollama not reachable at {_OLLAMA_BASE} — "
                f"falling back to LOCAL summaries (the llm variant == local here).\n")
        else:
            # uncached funcs only; longest-body first; cap to _OLLAMA_MAX
            uncached = []
            for fqn, sk, body in funcs:
                bhash = hashlib.sha1((body or "").encode("utf-8")).hexdigest()[:12]
                c = cache.get(fqn)
                if not (c and c.get("h") == bhash and c.get("s")):
                    uncached.append((fqn, len(body or "")))
            uncached.sort(key=lambda t: -t[1])
            llm_budget = {f for f, _ in uncached[:_OLLAMA_MAX]}
            if len(uncached) > _OLLAMA_MAX:
                sys.stderr.write(
                    f"[summary-search] {repo.name}: {len(uncached)} functions need "
                    f"LLM summaries but cap is {_OLLAMA_MAX}; the rest use LOCAL. "
                    f"Raise SG_SUMMARY_OLLAMA_MAX for a faithful run.\n")

    gen = used_llm = 0
    fqns: List[str] = []
    summaries: List[str] = []
    new_cache: Dict[str, dict] = {}
    for i, (fqn, sk, body) in enumerate(funcs):
        bhash = hashlib.sha1((body or "").encode("utf-8")).hexdigest()[:12]
        c = cache.get(fqn)
        if c and c.get("h") == bhash and c.get("s"):
            summ = c["s"]
        else:
            summ = ""
            if (source == "ollama" and ollama_ok and body and fqn in llm_budget):
                from skeletongraph.summary.ollama import generate_summary_ollama
                summ = generate_summary_ollama(
                    fqn=fqn, signature=getattr(sk, "signature", "") or "",
                    body=body, model=_OLLAMA_MODEL, base_url=_OLLAMA_BASE,
                    timeout=20) or ""
                gen += 1
                if summ:
                    used_llm += 1
                if gen % 50 == 0:
                    sys.stderr.write(
                        f"[summary-search] {repo.name}: LLM-summarized {gen} "
                        f"functions...\n")
            if not summ:
                try:
                    summ = build_local_summary(sk)
                except Exception:
                    summ = ""
        new_cache[fqn] = {"h": bhash, "s": summ}
        fqns.append(fqn)
        summaries.append(summ or fqn.split("::")[-1])   # never empty doc

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(new_cache, ensure_ascii=False),
                              encoding="utf-8")
    except Exception:
        pass
    if source == "ollama" and ollama_ok:
        sys.stderr.write(
            f"[summary-search] {repo.name}: {used_llm} LLM summaries this run "
            f"(cache now {len(new_cache)} funcs).\n")
    return fqns, summaries


def retrieve(query: str, repo_path: Path, top_n: int,
             source: str = "local", method: str = "bm25") -> List[str]:
    """Ranked FQNs from searching function summaries.

    source: "local" | "ollama"      method: "bm25" | "dense"
    """
    repo = Path(repo_path)
    fqns, summaries = _summary_docs(repo, source)
    if not fqns:
        return []

    if method == "bm25":
        _ensure_paths()
        try:
            from eval.backends.bm25_flat import _BM25, _tokenize
        except Exception:
            from backends.bm25_flat import _BM25, _tokenize
        docs = [_tokenize(s) for s in summaries]
        bm = _BM25(docs)
        q = _tokenize(query)
        order = sorted(range(len(fqns)), key=lambda i: -bm.score(q, i))
        return [fqns[i] for i in order[:top_n]]

    # dense over summaries
    _ensure_paths()
    try:
        from eval.backends.dense import rank
    except Exception:
        from backends.dense import rank
    cache = repo / ".skeletongraph" / "dense_cache"
    return rank(query, fqns, summaries, top_n, cache, tag=f"summ_{source}")
