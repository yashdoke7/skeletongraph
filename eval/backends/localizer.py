"""`sg-understand` — an iterative small-model localizer over SG's structure.

WHY THIS EXISTS (read before changing it)
-----------------------------------------
Measured on the released runs: re-querying `sg_search` SATURATES. On the
decontaminated benchmark SG issues 2.33 searches per task and recovers +0.000
recall between the first search and the last. Reformulations on an unfamiliar
repository all draw from the same impoverished vocabulary (the issue text), so
different phrasings produce the same ranking — no new information enters the
loop. Native grep does better (+0.111) only because it has a real feedback loop:
grep -> read a file -> learn the repo's actual vocabulary -> grep better.

So a localizer built on repeated `sg_search` calls is a DEAD END, and this module
deliberately is not one. Each round must inject information the query did not
contain, which is what the two new tools do:

  sg_outline    the module/package map — supplies repo vocabulary and structure
                that the issue text never contains, so the NEXT query differs in
                substance rather than being a paraphrase.
  sg_neighbors  call-graph traversal — reaches code that is causally related but
                lexically dissimilar, i.e. files no query ranking would surface.
  sg_search     kept, but as a PROBE for testing a hypothesis, not the engine.

The point is to substitute cheap tokens for expensive ones: a small model burns
the navigation turns and hands the frontier model a precise answer.

CONTRACT
--------
`retrieve(query, repo, top_n) -> List[str]` — identical to every other backend,
so retrieval_eval.py and tools.py treat it like any ranked retriever. The agent
NEVER sees the localizer's transcript; only the final ranked FQNs surface, which
keeps token accounting honest.

NEVER-WORSE FLOOR
-----------------
Every failure path (no API key, model error, bad tool args, budget exhausted,
empty result) returns the plain `fusion` ranking. This arm can tie fusion; by
construction it cannot do worse than it, which is what makes it safe to run.

GATING
------
By default the loop only fires when fusion looks UNCERTAIN (see `_is_confident`).
On symbol-bearing queries fusion is already correct and the extra calls are pure
cost/latency. Set SG_LOCALIZER_ALWAYS=1 to force the loop on every query (useful
for measuring the loop in isolation on the prose conditions).

ENV
---
  SG_LOCALIZER_MODEL     default meta/llama-3.3-70b-instruct
  SG_LOCALIZER_BUDGET    max tool calls before it must answer (default 8)
  SG_LOCALIZER_ALWAYS    1 = skip the confidence gate
  SG_EVAL_API_BASE / SG_EVAL_API_KEYS  reuse the harness's endpoint + rotation
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_MODEL = os.environ.get("SG_LOCALIZER_MODEL", "meta/llama-3.3-70b-instruct")
_BUDGET = int(os.environ.get("SG_LOCALIZER_BUDGET", "8"))
_ALWAYS = os.environ.get("SG_LOCALIZER_ALWAYS", "") == "1"
_CALL_TIMEOUT = float(os.environ.get("SG_LOCALIZER_CALL_TIMEOUT", "90"))
# Whole-loop deadline: caps TOTAL added latency regardless of turn budget, so a
# host agent (Claude Code, the react loop) never waits longer than this for one
# search — the per-call timeout alone cannot guarantee that (8 turns x 90s is
# still 12 minutes). Falls back to the never-worse floor if exceeded.
#
# 45s (the original default) was measured directly to allow at most 1-2 turns
# with this model (nvidia/llama-3.3-nemotron-super-49b-v1.5 runs 20-80s/turn),
# so EVERY non-gated task hit loop_error/loop_empty before the loop could ever
# reach `commit` -- 0/15 commits across multiple runs. That's cutting the loop
# off before it can show whether it works, not evidence it doesn't. Raised to
# give it real room (~4 turns worst-case) to actually reach a conclusion; this
# is a one-time diagnostic budget, not necessarily the production default.
_MAX_WALL_S = float(os.environ.get("SG_LOCALIZER_MAX_WALL_S", "300"))

# The localizer's own model is priced SEPARATELY from whatever SG_EVAL_MODEL
# is (nemotron on the react loop, Sonnet on Claude Code). config.PRICE_INPUT_
# PER_M/PRICE_OUTPUT_PER_M is the MAIN model's price and must not be reused
# here — mixing the two would silently misattribute cost between "the model
# doing the work" and "the cheap model finding where to work". Reference NIM
# list pricing for Llama-3.3-70B-Instruct; override if you switch models.
_PRICE_IN_PER_M = float(os.environ.get("SG_LOCALIZER_PRICE_IN", "0.10"))
_PRICE_OUT_PER_M = float(os.environ.get("SG_LOCALIZER_PRICE_OUT", "0.28"))


def _localizer_cost(tok_in: int, tok_out: int) -> float:
    return round(tok_in / 1e6 * _PRICE_IN_PER_M + tok_out / 1e6 * _PRICE_OUT_PER_M, 6)

# The fallback-to-fusion floor is silent BY DESIGN (that's what makes it a
# floor) — which means a missing key, a dead endpoint, or a bad model name all
# look IDENTICAL to a working run: same return type, no exception, plausible
# numbers (because they ARE fusion's numbers). This was measured directly: a
# run with no key set produced a full 15/15 "result" in 29s with zero network
# calls. So every call is counted here and a summary is printed once at
# interpreter exit — a fallback rate you cannot fail to notice.
import atexit
import time
_STATS = {"total": 0, "no_key": 0, "gated_confident": 0, "loop_ran": 0,
          "loop_empty": 0, "loop_error": 0, "loop_committed": 0,
          "localizer_tokens_in": 0, "localizer_tokens_out": 0,
          "localizer_cost_usd": 0.0}

# Per-CALL stats for the arm's OWN model, kept separate from whatever the main
# agent's model spends. This is the thing you asked for: when this arm runs
# inside the react loop or (later) Claude Code, the harness needs to attribute
# cost to "the cheap model that found the code" separately from "the model that
# wrote the fix" — summing everything into one number would hide exactly the
# substitution (cheap tokens for expensive ones) this arm exists to prove.
# retrieve()'s signature must stay List[str] (every backend shares that
# contract), so this is a side-channel: call last_call_stats() immediately
# after retrieve() to get the stats for THAT call.
_LAST_CALL: dict = {}


def last_call_stats() -> dict:
    """Localizer-only cost/turns for the most recent retrieve() call.

    {ran: bool, turns: int, tokens_in: int, tokens_out: int, cost_usd: float,
     outcome: str, elapsed_s: float}. `ran=False` means it hit the confidence
     gate or fell back — zero localizer cost was spent, which is itself a
     result worth recording (adoption/gating rate), not just a null.
    """
    return dict(_LAST_CALL)


def _trace(event: str) -> None:
    _STATS["total"] = _STATS.get("total", 0)
    _STATS[event] = _STATS.get(event, 0) + 1
    if os.environ.get("SG_LOCALIZER_VERBOSE", "1") != "0":
        print(f"[sg-understand] {event}", file=sys.stderr)


def dump_stats() -> dict:
    """Call this yourself for a stats dict; it also auto-prints once at exit."""
    return dict(_STATS)


@atexit.register
def _report_at_exit():
    t = _STATS["total"]
    if t == 0:
        return
    ran = _STATS["loop_committed"]
    print(f"\n[sg-understand] SUMMARY: {t} calls | "
          f"{_STATS['gated_confident']} skipped (confident) | "
          f"{_STATS['no_key']} FELL BACK — no API key/endpoint | "
          f"{ran} loop actually committed an answer | "
          f"{_STATS['loop_empty']} loop returned nothing | "
          f"{_STATS['loop_error']} loop errored | "
          f"localizer-only cost so far: ${_STATS['localizer_cost_usd']:.4f} "
          f"({_STATS['localizer_tokens_in']} in / {_STATS['localizer_tokens_out']} out tok)",
          file=sys.stderr)
    if _STATS["no_key"] == t:
        print("[sg-understand] *** EVERY call fell back to plain fusion — "
              "no LLM call was ever made. Check SG_EVAL_API_KEYS/"
              "SG_EVAL_API_KEY. This run measures fusion, not the localizer. ***",
              file=sys.stderr)


_MAX_OUTLINE_SYMBOLS = 12       # per file, before truncating
_MAX_OUTLINE_CHARS = 6000       # hard cap so a monorepo can't blow the context


# ── tool implementations (all cheap, all local) ──────────────────────────
# Per-process cache: the ACTUAL cause of a task appearing to hang. `get_store()`
# triggers `_ensure_loaded()`, a FULL index build, the first time it's called on
# a fresh SGEngine. Without this cache, every sg_outline/sg_neighbors call built
# a brand-new SGEngine and paid that build cost again — with an 8-turn budget
# that's up to 8 redundant full rebuilds for ONE task, on top of the model's own
# latency. Same pattern as fusion.py's _ENGINE_CACHE (keyed on resolved path;
# never invalidated mid-task, which is fine — a task's repo doesn't change under
# the localizer since it only reads).
_STORE_CACHE: dict = {}


def _store(repo: Path):
    key = str(Path(repo).resolve())
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached
    from skeletongraph.engine import SGEngine
    from skeletongraph.config import SGConfig
    cfg = SGConfig()
    cfg.enable_embeddings = False        # outline/neighbors need neither
    cfg.enable_summaries = False
    store = SGEngine(project_root=Path(repo), config=cfg).get_store()
    _STORE_CACHE[key] = store
    return store


def sg_outline(repo: Path, path: str = "") -> str:
    """Hierarchical map: directories -> files -> top-level symbols.

    This is the tool that makes iteration worth anything — it is the only one
    that hands the model vocabulary the issue text did not contain. `path`
    narrows to a subtree so a large repo can be drilled into lazily instead of
    dumped at once.
    """
    store = _store(repo)
    by_file: Dict[str, List[str]] = defaultdict(list)
    for fqn, sk in store.skeleton_table.items():
        fp = str(getattr(sk, "file_path", "")).replace("\\", "/")
        if path and not fp.startswith(path.strip("/")):
            continue
        by_file[fp].append(fqn.split("::")[-1])

    if not by_file:
        return f"(no files under {path!r})" if path else "(empty index)"

    # Group by top directory so the model sees architecture, not a flat list.
    by_dir: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for fp, syms in sorted(by_file.items()):
        d = "/".join(fp.split("/")[:-1]) or "."
        by_dir[d].append((fp.split("/")[-1], syms))

    out: List[str] = []
    for d, files in sorted(by_dir.items()):
        out.append(f"{d}/")
        for fname, syms in files:
            shown = syms[:_MAX_OUTLINE_SYMBOLS]
            more = f" (+{len(syms) - len(shown)} more)" if len(syms) > len(shown) else ""
            out.append(f"  {fname}: {', '.join(shown)}{more}")
    text = "\n".join(out)
    if len(text) > _MAX_OUTLINE_CHARS:
        text = (text[:_MAX_OUTLINE_CHARS]
                + f"\n... truncated. Call sg_outline with a `path` to drill in.")
    return text


def sg_neighbors(repo: Path, fqn: str) -> str:
    """Callers and callees of `fqn` — traversal to causally-related code.

    Reaches functions that no lexical or semantic ranking would surface, because
    graph adjacency is independent of textual similarity. This is the mechanism
    for multi-hop symptom->cause chains.
    """
    store = _store(repo)
    graph = getattr(store, "graph", None)
    if graph is None:
        return "(no call graph available for this repository)"

    # Resolve a loosely-specified fqn (the model may give just a symbol name).
    target = fqn
    if fqn not in store.skeleton_table:
        cands = [k for k in store.skeleton_table
                 if k.split("::")[-1].split(".")[-1] == fqn.split("::")[-1].split(".")[-1]]
        if not cands:
            return f"(unknown symbol {fqn!r} — use sg_search or sg_outline to find a valid one)"
        target = cands[0]

    try:
        callers = graph.blast_radius(target, max_depth=1) or {}
        callees = graph.dependency_chain(target, max_depth=1) or {}
    except Exception as e:
        return f"(graph traversal failed: {e})"

    def _fmt(d: Dict[str, int], label: str) -> str:
        items = [k for k, dist in sorted(d.items(), key=lambda kv: kv[1]) if k != target][:12]
        return f"{label}: " + (", ".join(items) if items else "(none)")

    return f"{target}\n{_fmt(callers, 'CALLED BY')}\n{_fmt(callees, 'CALLS')}"


def _fusion(query: str, repo: Path, k: int) -> List[str]:
    from skeletongraph.retrieval.fusion import retrieve_fusion
    return retrieve_fusion(query, Path(repo), k)


# ── confidence gate ──────────────────────────────────────────────────────
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _is_confident(query: str, repo: Path, ranked: List[str]) -> bool:
    """Does the query name a symbol that fusion actually returned at rank 1-3?

    This is the entity-anchored case where fusion is already right and the loop
    would be pure overhead. Deliberately conservative: when in doubt, run the
    loop (the floor guarantees we cannot end up worse than fusion).
    """
    if not ranked:
        return False
    q = {w.lower() for w in _IDENT.findall(query or "")}
    for fqn in ranked[:3]:
        name = fqn.split("::")[-1].split(".")[-1].lower()
        if name in q:
            return True
    return False


# ── the loop ─────────────────────────────────────────────────────────────
_TOOLS = [
    {"type": "function", "function": {
        "name": "sg_outline",
        "description": ("Map of the repository: directories -> files -> the symbols "
                        "they define. Use this FIRST to learn where things live and "
                        "what vocabulary this codebase uses. Optional `path` narrows "
                        "to a subtree."),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "optional subtree, e.g. 'src/auth'"}}}}},
    {"type": "function", "function": {
        "name": "sg_search",
        "description": ("Ranked structural search for code matching a query. Use to TEST "
                        "a hypothesis you formed from the outline. Re-phrasing the same "
                        "idea returns the same results, so change the substance of the "
                        "query (use vocabulary you learned from the outline) rather than "
                        "the wording."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "sg_neighbors",
        "description": ("Callers and callees of a function. Use to walk from a function "
                        "you found toward the one actually responsible — reaches related "
                        "code that shares no words with your query."),
        "parameters": {"type": "object", "properties": {
            "fqn": {"type": "string", "description": "file.py::Class.method or a symbol name"}},
            "required": ["fqn"]}}},
    {"type": "function", "function": {
        "name": "commit",
        "description": "Report the files/functions most likely to need editing. Call when confident or out of budget.",
        "parameters": {"type": "object", "properties": {
            "fqns": {"type": "array", "items": {"type": "string"},
                     "description": "ranked, most likely first"},
            "why": {"type": "string", "description": "one line"}},
            "required": ["fqns"]}}},
]

_SYSTEM = """You localize bugs in an unfamiliar repository.

You are given an issue report. Find WHICH FUNCTIONS must be edited to fix it.

The issue often describes a SYMPTOM, not the code. Your job is to reason from the
symptom to the responsible code:
  symptom -> what behaviour produces it -> which component owns that behaviour
  -> which function implements it.

Strategy that works:
 1. sg_outline first — learn the repo's structure and its vocabulary.
 2. Form a hypothesis about which module owns the behaviour.
 3. sg_search to test it, using words you learned from the outline.
 4. sg_neighbors to walk the call graph toward the real cause.
 5. commit with a ranked list.

Do not repeatedly rephrase the same search — identical meaning returns identical
results. If a search disappoints, get NEW information (outline a subtree, or
traverse neighbours) instead.

Be decisive. You have a small budget."""


def _resolve_key() -> Tuple[str, str, int]:
    """(base, key, timeout).

    `config.get_api_key()` ONLY returns a thread-local key that `run_stage.py`
    assigns per worker for its multi-account rotation — it never reads
    SG_EVAL_API_KEYS directly. A caller with no thread pool (the retrieval-only
    probe, a smoke test, this module's own __main__) gets nothing from it even
    when SG_EVAL_API_KEYS is set. Measured directly: two probe runs with
    SG_EVAL_API_KEYS exported still fell back to fusion, because of exactly
    this gap. Fall through every plausible source, in order, instead of
    trusting one function that was written for a different call path.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    base = os.environ.get("SG_EVAL_API_BASE", "https://integrate.api.nvidia.com/v1")
    timeout = 120
    try:
        from agent import config as cfg
        base = cfg.API_BASE
        timeout = cfg.REQUEST_TIMEOUT
        key = cfg.get_api_key()
        if key and key != "EMPTY":
            return base, key, timeout
        if getattr(cfg, "_NIM_KEYS", None):
            return base, cfg._NIM_KEYS[0], timeout    # rotation pool, no thread assigned
    except Exception:
        pass
    key = os.environ.get("SG_EVAL_API_KEY", "")
    if key and key != "EMPTY":
        return base, key, timeout
    pool = [k.strip() for k in os.environ.get("SG_EVAL_API_KEYS", "").split(",") if k.strip()]
    return base, (pool[0] if pool else ""), timeout


def _run_loop(query: str, repo: Path, k: int) -> Optional[List[str]]:
    t_start = time.monotonic()
    _LAST_CALL.clear()
    _LAST_CALL.update(ran=True, turns=0, tokens_in=0, tokens_out=0,
                      cost_usd=0.0, outcome="pending", elapsed_s=0.0)

    def _finish(outcome: str, result):
        _LAST_CALL["outcome"] = outcome
        _LAST_CALL["elapsed_s"] = round(time.monotonic() - t_start, 1)
        return result

    try:
        from openai import OpenAI
    except Exception:
        _trace("no_key")
        return _finish("no_key", None)

    # A separate, SHORT timeout from the harness's own REQUEST_TIMEOUT (300s,
    # sized for the MAIN model's long coding turns). Left at 300s, one stalled
    # call in an 8-turn budget can hang for up to 40 minutes with zero visible
    # progress — indistinguishable from a genuine hang without the trace lines
    # below. The localizer's calls are small (a short hypothesis + a tool
    # call), so 60s is generous, not tight.
    base, key, _harness_timeout = _resolve_key()
    if not key or key == "EMPTY":
        _trace("no_key")
        return _finish("no_key", None)

    # max_retries=0: the SDK's default (2 retries -> 3 attempts) silently
    # multiplies any timeout by ~3x. Measured directly: a 6s per-call timeout
    # failed after 20.6s, a 22s timeout failed after 66.7s — both ~3x, not the
    # requested value. The wall-clock cap is meaningless if a single "failed"
    # call can secretly retry twice more inside it.
    client = OpenAI(base_url=base, api_key=key, timeout=_CALL_TIMEOUT, max_retries=0)
    msgs = [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"ISSUE:\n{(query or '')[:6000]}"}]

    # Measured directly: nvidia/llama-3.3-nemotron-super-49b-v1.5 (a reasoning-
    # capable model) took 5-104s PER TURN for a ~700-token completion, and 12/15
    # tasks in one probe run never emitted a tool call at all ("answered in
    # text"). Both symptoms match a reasoning model buffering an internal
    # chain-of-thought before answering — exactly the failure mode react.py
    # already disables for the MAIN model via this same extra_body flag. The
    # localizer never reused it. Do so here, identically.
    extra_body = {}
    try:
        from agent import config as cfg
        if getattr(cfg, "DISABLE_THINKING", True):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    except Exception:
        if os.environ.get("SG_EVAL_DISABLE_THINKING", "1") != "0":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    tok_in = tok_out = 0
    turn = 0
    for turn in range(1, _BUDGET + 1):
        # Whole-LOOP deadline, separate from the per-call timeout. The per-call
        # timeout only bounds ONE turn; without a total ceiling an 8-turn budget
        # at 60s/call can legitimately take 8 minutes, which is exactly what
        # was observed (up to 205s for one task) and is unacceptable latency to
        # impose on a host agent waiting for a search result. Breaking here
        # falls through to the never-worse floor exactly like any other failure.
        elapsed = time.monotonic() - t_start
        if elapsed > _MAX_WALL_S:
            print(f"[sg-understand] turn {turn}/{_BUDGET} SKIPPED — wall-clock "
                 f"budget ({_MAX_WALL_S:.0f}s) exceeded at {elapsed:.1f}s, "
                 f"falling back to fusion", file=sys.stderr)
            break
        t_turn = time.monotonic()
        # The client's own timeout (_CALL_TIMEOUT) bounds ONE call in isolation,
        # but was measured to blow straight through _MAX_WALL_S anyway: a call
        # starting at elapsed=40s with a 60s allowance still runs to 100s total
        # (deltares-hydrolib hit 101.7s this way). Cap this call's timeout to
        # whatever wall budget is actually left so the 45s ceiling is real.
        call_timeout = max(5.0, min(_CALL_TIMEOUT, _MAX_WALL_S - elapsed))
        print(f"[sg-understand] turn {turn}/{_BUDGET} — calling {_MODEL} "
              f"(timeout {call_timeout:.0f}s, {_MAX_WALL_S - elapsed:.0f}s left "
              f"in wall budget)...", file=sys.stderr)
        try:
            resp = client.chat.completions.create(
                model=_MODEL, messages=msgs, tools=_TOOLS,
                # "required" not "auto": measured 3/6 non-gated tasks getting a
                # bare-text answer instead of a tool call (including never
                # calling `commit`), despite the system prompt insisting on
                # tool use. commit IS one of the four tools, so this can't
                # block a real answer — it only removes the silent-noop path.
                # 700 was cutting function-call JSON off mid-generation under
                # tool_choice="required" -- measured directly as two 400s
                # ("EOF while parsing a value/object", i.e. a truncated arg
                # string), not a malformed-tool-choice issue.
                tool_choice="required", temperature=0.0, max_tokens=1024,
                extra_body=extra_body or None, timeout=call_timeout)
        except Exception as e:
            dt = time.monotonic() - t_turn
            print(f"[sg-understand] turn {turn} FAILED after {dt:.1f}s: {e}",
                 file=sys.stderr)
            _trace("loop_error")
            _LAST_CALL.update(turns=turn, tokens_in=tok_in, tokens_out=tok_out,
                              cost_usd=_localizer_cost(tok_in, tok_out))
            return _finish("loop_error", None)

        u = getattr(resp, "usage", None)
        ci = getattr(u, "prompt_tokens", 0) or 0
        co = getattr(u, "completion_tokens", 0) or 0
        tok_in += ci; tok_out += co
        dt = time.monotonic() - t_turn
        m = resp.choices[0].message
        calls = getattr(m, "tool_calls", None) or []
        tool_desc = ", ".join(c.function.name for c in calls) if calls else "(no tool call — model answered in text)"
        print(f"[sg-understand] turn {turn}/{_BUDGET} done in {dt:.1f}s "
              f"| +{ci} in / +{co} out tok | tool: {tool_desc}", file=sys.stderr)

        if not calls:
            break
        # Measured directly: this model sometimes emits an empty-string
        # `arguments` field on a tool call. json.loads below already guards
        # against that for OUR parsing, but the raw empty string was still
        # going straight into conversation history -- and the NEXT turn's
        # request gets rejected with a 400 ("EOF while parsing a value") when
        # the server re-validates that malformed historical tool call. "{}" is
        # always valid JSON, so history stays sendable either way.
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name,
                                                  "arguments": c.function.arguments or "{}"}}
                                    for c in calls]})
        wall_blown = False
        for c in calls:
            # Re-check mid-turn: a model can request several tool calls in one
            # turn, and sg_neighbors/sg_outline on a real (uncached-per-call)
            # repo graph can each take seconds — the pre-turn check above only
            # bounds the LLM call itself, so a slow multi-tool turn was measured
            # blowing straight through _MAX_WALL_S to 95-118s. Bail mid-turn
            # instead of waiting for the next top-of-loop check.
            if time.monotonic() - t_start > _MAX_WALL_S:
                print(f"[sg-understand] turn {turn}/{_BUDGET} SKIPPED remaining "
                     f"tool call(s) — wall-clock budget exceeded mid-turn, "
                     f"falling back to fusion", file=sys.stderr)
                wall_blown = True
                break
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except Exception:
                args = {}
            if name == "commit":
                got = [str(x) for x in (args.get("fqns") or []) if x]
                cost = _localizer_cost(tok_in, tok_out)
                print(f"[sg-understand] committed after {turn} model call(s) "
                     f"in {time.monotonic() - t_start:.1f}s, {len(got)} fqn(s), "
                     f"localizer cost ${cost:.4f} ({tok_in} in / {tok_out} out)",
                     file=sys.stderr)
                _trace("loop_committed")
                _STATS["localizer_tokens_in"] += tok_in
                _STATS["localizer_tokens_out"] += tok_out
                _STATS["localizer_cost_usd"] += cost
                _LAST_CALL.update(turns=turn, tokens_in=tok_in, tokens_out=tok_out,
                                  cost_usd=cost)
                return _finish("committed", got[:k] or None)
            if name == "sg_outline":
                out = sg_outline(repo, str(args.get("path") or ""))
            elif name == "sg_search":
                hits = _fusion(str(args.get("query") or query), repo, 10)
                out = "\n".join(hits) if hits else "(no matches)"
            elif name == "sg_neighbors":
                out = sg_neighbors(repo, str(args.get("fqn") or ""))
            else:
                out = f"(unknown tool {name})"
            msgs.append({"role": "tool", "tool_call_id": c.id,
                         "content": out[:4000]})
        if wall_blown:
            break

    cost = _localizer_cost(tok_in, tok_out)
    used_turns = turn
    print(f"[sg-understand] loop ended without commit after {used_turns} turn(s), "
         f"{time.monotonic() - t_start:.1f}s, localizer cost ${cost:.4f} "
         f"({tok_in} in / {tok_out} out)", file=sys.stderr)
    _trace("loop_empty")
    _STATS["localizer_tokens_in"] += tok_in
    _STATS["localizer_tokens_out"] += tok_out
    _STATS["localizer_cost_usd"] += cost
    _LAST_CALL.update(turns=used_turns, tokens_in=tok_in, tokens_out=tok_out, cost_usd=cost)
    return _finish("loop_empty", None)


def _resolve_loop_candidates(loop: List[str], repo: Path) -> List[str]:
    """Map the model's raw picks (bare symbol, dotted name, or file path) onto
    real indexed FQNs. Anything unresolvable is dropped rather than emitted as
    a phantom hit. Order preserved, so the model's own confidence ranking
    survives into the RRF blend below."""
    try:
        known = set(_store(repo).skeleton_table)
    except Exception:
        known = set()
    out: List[str] = []
    for cand in loop:
        c = cand.replace("\\", "/").strip()
        if c in known:
            out.append(c); continue
        tail = c.split("::")[-1].split(".")[-1]
        match = [k_ for k_ in known
                 if k_.split("::")[-1].split(".")[-1] == tail
                 or k_.replace("\\", "/").startswith(c)]
        if match:
            out.append(match[0])
    seen, dedup = set(), []
    for f in out:
        if f not in seen:
            seen.add(f); dedup.append(f)
    return dedup


def _merge(loop: List[str], base: List[str], repo: Path, k: int) -> List[str]:
    """RRF-blend the loop's (resolved) picks WITH the floor, instead of the
    loop unconditionally overriding it.

    Measured directly (3 models, 3 runs) that a straight "loop's picks first,
    then base" override makes the never-worse floor false in practice: a
    single wrong `commit` can demote a floor rank-1 that was already correct
    (pennylane: fusion MRR 1.0 -> sg-understand MRR 0.125 under the old
    override). At the same time the loop DOES find real fixes fusion missed
    (pypsa: 0.33 -> 1.0) — so the fix isn't to trust the floor unconditionally
    either, it's to stop letting either side unilaterally out-vote the other.
    Equal-weight RRF (k=60, matching retrieve_fusion's own convention): the
    loop's answer only OUTRANKS the floor's rank-1 if it ALSO appears at
    fusion's own top ranks or the floor doesn't already have it there — a
    single bad guess is one vote against the floor's existing signal, not a
    wholesale replacement of it.
    """
    resolved = _resolve_loop_candidates(loop, repo)
    if not resolved:
        return base[:k]
    scores: Dict[str, float] = {}
    for lst in (base, resolved):
        for rank_i, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60 + rank_i + 1)
    # Stable tie-break toward the floor's own order (base built first above,
    # so equal scores keep base's relative order under Python's stable sort).
    order = {f: i for i, f in enumerate(base)}
    ranked = sorted(scores, key=lambda f: (-scores[f], order.get(f, len(base))))
    return ranked[:k]


def retrieve(query: str, repo_path: Path, top_n: int) -> List[str]:
    _STATS["total"] += 1
    _LAST_CALL.clear()
    _LAST_CALL.update(ran=False, turns=0, tokens_in=0, tokens_out=0,
                      cost_usd=0.0, outcome="gated_confident", elapsed_s=0.0)
    repo = Path(repo_path)
    base = _fusion(query, repo, top_n)          # the floor, always computed
    if not _ALWAYS and _is_confident(query, repo, base):
        _trace("gated_confident")
        return base                              # entity-anchored: fusion is right, zero localizer cost
    try:
        loop = _run_loop(query, repo, top_n)     # populates _LAST_CALL itself
    except Exception as e:
        print(f"[sg-understand] loop_error (outer): {e}", file=sys.stderr)
        _trace("loop_error")
        _LAST_CALL["outcome"] = "loop_error"
        loop = None
    if not loop:
        return base                              # never-worse floor
    return _merge(loop, base, repo, top_n)


if __name__ == "__main__":                       # smoke: python -m backends.localizer <repo> "<issue>"
    r = Path(sys.argv[1]); q = sys.argv[2] if len(sys.argv) > 2 else "bug"
    print("OUTLINE:\n", sg_outline(r)[:800], "\n")
    print("RESULT:", *retrieve(q, r, 10), sep="\n  ")
