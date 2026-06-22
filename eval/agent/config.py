"""Central configuration for the agentic eval harness.

Everything tunable lives here: arms, models, the staged run plan, the price
sheet used to impute cost, and paths. Nothing else should hard-code these.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Silence HuggingFace / sentence-transformers model-load progress bars
# ("Loading weights: 100%|...") that interleave with the eval console output.
# Must be set before any transformers/huggingface_hub import; config is imported
# first by every agent module, so this covers the whole harness.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── paths ──────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent.parent          # .../eval
REPO_ROOT = EVAL_DIR.parent                                 # repo root
DATASET = EVAL_DIR / "datasets" / "stage0.jsonl"            # built by make_dataset.py
# Results folder tag — isolates results by model + benchmark so different runs
# never overwrite each other (run_id always uses "main" regardless of model).
# Set this env var BEFORE running run_stage / run_singleshot / aggregate:
#   CMD:    set SG_EVAL_RUN_TAG=qwen7b_swebench
#   bash:   export SG_EVAL_RUN_TAG=qwen32b_swebench
# Default (empty) = eval/results/agent/ — backward-compatible with legacy runs.
_RUN_TAG = os.environ.get("SG_EVAL_RUN_TAG", "")
RUNS_DIR = (EVAL_DIR / "results" / "agent" / _RUN_TAG
            if _RUN_TAG else EVAL_DIR / "results" / "agent")  # per-run trajectories

# ── Heavy IO root (clones + per-run workspaces) ─────────────────────────────
# The base repos and the transient per-run checkouts are GBs and get copied on
# every run. Keep them OUT of the SG repo (cleaner git status, no space bloat,
# clear isolation) by pointing SG_EVAL_DATA_ROOT at a sibling dir, e.g.:
#   $env:SG_EVAL_DATA_ROOT = "C:\Users\ASUS\Desktop\CS\Projects\swebench-data"
# Default (unset) = eval/datasets (legacy, in-repo). make_dataset.py honors the
# same env so clones land there and the jsonl repo_path points there too.
_DATA_ROOT = Path(os.environ["SG_EVAL_DATA_ROOT"]) if os.environ.get("SG_EVAL_DATA_ROOT") \
    else EVAL_DIR / "datasets"
# Workspace root is ALSO tag-namespaced. run_id uses model="main" for every
# model, so two runs (e.g. 7B + NIM-70B) on the same (task, arm) would otherwise
# share one checkout dir and clobber each other if run concurrently. Tagging the
# root lets different SG_EVAL_RUN_TAG runs execute fully in parallel.
WORKSPACE_ROOT = (_DATA_ROOT / "_agent_work" / _RUN_TAG
                  if _RUN_TAG else _DATA_ROOT / "_agent_work")  # isolated per-run checkouts


# ── model endpoint ─────────────────────────────────────────────────────────
# The harness talks to ONE OpenAI-compatible endpoint (vLLM). Swap models by
# changing MODEL_NAME + restarting vLLM with the matching --model.

API_BASE = os.environ.get("SG_EVAL_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("SG_EVAL_API_KEY", "EMPTY")        # vLLM ignores it
MODEL_NAME = os.environ.get("SG_EVAL_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

# ── Multi-account NIM key rotation ─────────────────────────────────────────
# When running against NIM (nvidia inference microservices) with multiple
# accounts, set SG_EVAL_API_KEYS to a comma-separated list of API keys.
# run_stage.py assigns one key per job (round-robin by job index) and sets it
# as a thread-local before dispatching. _client() in react.py reads this
# thread-local, so each concurrent worker uses a different NIM account and
# therefore a separate per-account rate limit.
#
# Usage (CMD):
#   set SG_EVAL_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3,nvapi-key4
#   python -m eval.agent.run_stage --stage baseline --workers 8
#
# Usage (bash/AMD):
#   export SG_EVAL_API_KEYS="nvapi-key1,nvapi-key2,nvapi-key3,nvapi-key4"
#   python -m eval.agent.run_stage --stage 1a-workshop --workers 16
#
# If SG_EVAL_API_KEYS is not set, falls back to API_KEY (single-account mode).

_NIM_KEYS: list = [
    k.strip()
    for k in os.environ.get("SG_EVAL_API_KEYS", "").split(",")
    if k.strip()
]

# Thread-local storage: run_stage sets this per-job so each worker uses its
# assigned key for the duration of the run_one() call.
_thread_api_key: threading.local = threading.local()


def get_api_key() -> str:
    """Return the API key for the current worker thread.

    If multi-key rotation is active (SG_EVAL_API_KEYS set), returns the key
    assigned to this thread by run_stage. Otherwise returns the global API_KEY.
    """
    return getattr(_thread_api_key, "key", None) or API_KEY


def set_thread_api_key(key: Optional[str]) -> None:
    """Called by run_stage before dispatching each job to assign its NIM key."""
    _thread_api_key.key = key

TEMPERATURE = 0.0          # deterministic — pin this, never change mid-study
SEED = 42
# Disable chain-of-thought/thinking for models that support the toggle (Qwen3,
# DeepSeek, etc.). Thinking mode buffers the full reasoning trace before sending
# content chunks, making streaming as slow as non-streaming for our harness.
# Set SG_EVAL_DISABLE_THINKING=0 to re-enable (e.g. to study reasoning models).
DISABLE_THINKING: bool = os.environ.get("SG_EVAL_DISABLE_THINKING", "1") != "0"
MAX_TURNS = int(os.environ.get("SG_EVAL_MAX_TURNS", "40"))  # ReAct step ceiling.
# Override via SG_EVAL_MAX_TURNS for a BUDGET SWEEP: at a tight budget the agent
# cannot brute-force localization, so precise (rank-1) retrieval should separate
# from none/noisy on pass@1 — the test for "does retrieval matter under pressure".
# Applied uniformly to every arm → unbiased (a cost-controlled comparison).
REQUEST_TIMEOUT = 300      # seconds per model call

# ── bounded context (matches how real IDE/agent loops manage history) ────────
# The transcript is re-sent every turn. Without bounds, old read_file/search
# dumps accumulate and inflate input tokens linearly with turns (observed: 200K+
# tails). Real agents (SWE-agent/OpenHands history processors, Aider, Cursor)
# keep only recent observations verbatim and elide older ones. We keep the last
# N tool-result messages in full and stub the rest (the action trace + task are
# always kept). Identical for every arm → no bias. Set 0 to disable (legacy).
# This MATTERS MOST for SG: the agent should lean on the structural summary
# instead of re-reading, so eliding stale raw dumps should widen SG's lead.
CONTEXT_KEEP_LAST_TOOL_OUTPUTS = 5   # tool results kept verbatim; older → stub
CONTEXT_STUB_OVER_CHARS = 600        # only stub outputs longer than this


# ONE model for the whole study — it is a fixed control, not the contribution.
MODELS: Dict[str, Dict[str, str]] = {
    # MAIN — Qwen2.5-Coder-32B-Instruct: the recognised open research workhorse
    # for SWE-bench-style work. Dense 32B (~64 GB BF16) → trivial on MI300X, no
    # FP8/MoE serving risk; strong on SWE-bench Verified; many published
    # baselines, so SG's delta is cleanly interpretable. (To use a more current
    # model accept FP8/MoE setup on ROCm and set SG_EVAL_MODEL to
    # Qwen/Qwen3-Coder-Next — one env var, no code change.)
    "main": {"hf": "Qwen/Qwen2.5-Coder-32B-Instruct", "role": "main"},
    # DRY-RUN model — small + stable, for the RunPod harness validation run and
    # the optional smaller-model point in stage 3-scale.
    "qwen-7b": {"hf": "Qwen/Qwen2.5-Coder-7B-Instruct", "role": "small"},
}


# ── price sheet ────────────────────────────────────────────────────────────
# Used ONLY to impute a $ cost per task (Axis 5) — tokens on self-hosted vLLM
# are "free", but the paper needs a comparable cost number. Fixed published
# reference rates ($/million tokens). Update once, never per-run.

PRICE_INPUT_PER_M = 0.27
PRICE_OUTPUT_PER_M = 1.10
PRICE_CACHED_INPUT_PER_M = 0.07


def impute_cost(input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Reference-priced cost for one run. cached_tokens are billed at the cheap rate."""
    fresh_in = max(0, input_tokens - cached_tokens)
    return round(
        fresh_in / 1e6 * PRICE_INPUT_PER_M
        + cached_tokens / 1e6 * PRICE_CACHED_INPUT_PER_M
        + output_tokens / 1e6 * PRICE_OUTPUT_PER_M,
        5,
    )


# ── graph-EXTRACTION model price (the index-build cost SG never pays) ────────
# graphify and cbmem build their knowledge graph with an LLM, once per repo.
# That cost is real and must be attributed to those arms (a finding: structural
# competitors need an LLM to even construct their index; SG's tree-sitter index
# is zero-LLM). The extraction model differs from the AGENT model — on NIM it is
# Llama-3.3-70B (Qwen-32B-Coder was deprecated on NIM); on AMD it is the local
# 32B. Default = a published Llama-3.3-70B reference rate; override per provider.
PRICE_EXTRACT_INPUT_PER_M = 0.23
PRICE_EXTRACT_OUTPUT_PER_M = 0.40


def impute_extract_cost(input_tokens: float, output_tokens: float) -> float:
    """Reference-priced cost of the one-time LLM graph build (graphify/cbmem)."""
    return round(
        input_tokens / 1e6 * PRICE_EXTRACT_INPUT_PER_M
        + output_tokens / 1e6 * PRICE_EXTRACT_OUTPUT_PER_M,
        5,
    )


# ── arms ───────────────────────────────────────────────────────────────────
# An arm = the retrieval backend the agent's `search_code` tool dispatches to.
# Every arm gets the IDENTICAL tool set {search_code, list_files, read_file,
# edit_file, submit}; only search_code's implementation differs. That parity is
# what isolates SG's contribution.

@dataclass
class Arm:
    name: str
    backend: str          # key dispatched in tools.search_code
    label: str            # human label for tables
    strong: bool = False  # True for the strong-baseline tier


ARMS: Dict[str, Arm] = {
    # `sg` is now the LEAN product default: structural core (gated graph +
    # centrality rerank + BM25 weak-entity fallback), summaries + embeddings
    # OFF. See tools._sg_config / _LEAN_SG for the rationale (embeddings are
    # retrieval-inert in the heuristic path; summaries are agent previews).
    "sg":     Arm("sg",     "sg",     "SkeletonGraph (lean core)"),
    "bm25":   Arm("bm25",   "bm25",   "Flat BM25"),
    "grep":   Arm("grep",   "grep",   "Ripgrep-style lexical"),
    # NOTE: `none` returns NO search results — the agent navigates blind via
    # list_files/read_file. It is NOT a long-context dump (we never paste whole
    # files/repo into the prompt). Labeled accordingly so the paper isn't wrong.
    "none":   Arm("none",   "none",   "No retrieval (blind file navigation)"),
    "hybrid": Arm("hybrid", "hybrid", "Hybrid RAG (BM25+dense+rerank)", strong=True),
    "cbmem":  Arm("cbmem",  "cbmem",  "Codebase-Memory (MCP graph)", strong=True),
    # ── Ablations AROUND the lean default — each isolates ONE decision.
    # These are the FINAL-run ablation set (compare every row against lean `sg`).
    "sg-full":    Arm("sg-full",    "sg-full",    "SG (lean + summaries + embeddings = old full)"),
    "sg-summary": Arm("sg-summary", "sg-summary", "SG (lean + summaries)"),
    "sg-embed":   Arm("sg-embed",   "sg-embed",   "SG (lean + embeddings)"),
    # SG ablations — same SG retrieval with exactly one component disabled
    # (wired in tools._sg_config). Each isolates a contribution.
    "sg-nograph":   Arm("sg-nograph",   "sg-nograph",   "SG (lean, no graph expansion)"),
    "sg-gatedgraph": Arm("sg-gatedgraph", "sg-gatedgraph", "SG (gated graph expansion)"),
    "sg-fullgraph": Arm("sg-fullgraph", "sg-fullgraph", "SG (eager graph expansion)"),
    "sg-norerank":  Arm("sg-norerank",  "sg-norerank",  "SG (no centrality rerank)"),
    "sg-nosummary": Arm("sg-nosummary", "sg-nosummary", "SG (no summaries)"),
    "sg-noembed":   Arm("sg-noembed",   "sg-noembed",   "SG (no embeddings)"),
    # Recall booster (gated weak-entity fallback, see docs/BLUEPRINT.md §4/§7).
    # Full SG + BM25 seeds ONLY when the entity match is ambiguous, so precise
    # matches keep SG's precision. Tests recovery of semantic-mismatch misses.
    "sg-weakfallback": Arm("sg-weakfallback", "sg-weakfallback",
                           "SG (gated weak-entity recall booster)"),
    # Learned curator (see docs/CURATOR.md): a trained classifier predicts the
    # retrieval mode per query instead of the rule-based router. Same SG index;
    # only the mode selection changes (passed via heuristic_query mode_hint).
    "sg-learned":   Arm("sg-learned",   "sg-learned",   "SG (learned curator)"),

    # ── Experimental / trial arms (synthesized from the ablation findings) ──
    # The v3 ablations suggested SG is OVER-engineered for the agent loop:
    # tier-2 summaries and centrality rerank in the search path appear to cost
    # more than they return (sg-nosummary / sg-norerank trended ABOVE plain sg),
    # while lazy graph expansion and embeddings earn their keep. These three
    # arms operationalize three different responses to that finding. All share
    # the same "lean" defaults (no summaries, no centrality rerank surfaced to
    # the agent) and run in sg-env alongside `baseline`.
    #
    #   sg-lean   — STATIC trim: SG minus summaries minus centrality rerank.
    #               Embeddings on, lazy (gated) graph on. The ship-simpler
    #               product candidate: "what the ablations say is optimal."
    #   sg-router — ADAPTIVE: a deterministic per-query router reads the query
    #               SHAPE (precise symbol lookup vs relational vs conceptual)
    #               and spends retrieval effort only where it helps — graph OFF
    #               for symbol lookups, graph ALWAYS for relational queries,
    #               gated for conceptual. Conditional computation; lean defaults.
    #   sg-fusion — ENSEMBLE: reciprocal-rank fusion (RRF) of SG-structural +
    #               flat BM25, no learned reranker, no summaries. "Combine
    #               cheaply" instead of selecting one retriever — a different
    #               philosophy from routing, robust to either signal missing.
    #   sg-chain  — PATH-AWARE: full-body BM25 supplies issue-text recall, lean
    #               SG supplies structural precision, then short graph paths and
    #               consensus files select the minimal evidence chain. This is
    #               the transcript-style "summaries navigate, raw code proves"
    #               candidate without paying for an LLM reranker.
    "sg-chain":  Arm("sg-chain",  "sg-chain",  "SG-chain (path-aware SG+BM25, current best)"),
    # sg-rerank: bm25 recall pool REORDERED by SG structural confirmation (generate-
    # then-rerank, NOT RRF, NOT per-file capped). Aims to keep bm25's recall (0.36)
    # AND inherit sg-chain's rank (1.5) — neither current arm has both.
    "sg-rerank": Arm("sg-rerank", "sg-rerank", "SG-rerank (bm25 recall + SG rerank)",
                     strong=True),
    # fusion: 3-way reciprocal-rank fusion of lexical (BM25) + semantic (dense,
    # code-search model via SG_DENSE_MODEL) + structural (SG-rerank). The
    # natural-language-adaptable retriever — validated on SWE-bench Pro retrieval
    # (recall@1 +50%, recall@10 +12%, MRR +26% over BM25; every language up).
    # No single signal wins NL localization; their RRF beats each. Set
    # SG_DENSE_MODEL=jinaai/jina-embeddings-v2-base-code for the run.
    "fusion": Arm("fusion", "fusion",
                  "3-way RRF (BM25 + Dense + SG; NL-adaptive)", strong=True),
    # New concepts (native harness): semantic dense rerank of the pool, and
    # SG seeded on the issue's tracebacks/code symbols.
    "sg-hybrid-fusion": Arm("sg-hybrid-fusion", "sg-hybrid-fusion",
                            "SG-hybrid-fusion (RRF merge of BM25 and Dense)"),
    "sg-dense-rerank": Arm("sg-dense-rerank", "sg-dense-rerank",
                           "SG-dense-rerank (BM25 recall + Dense rescore)", strong=True),
    "sg-keyword-dense": Arm("sg-keyword-dense", "sg-keyword-dense",
                            "SG-keyword-dense (Extracted keywords -> Dense)", strong=True),
    "sg-seed":  Arm("sg-seed",  "sg-seed",
                    "SG-seed (SG seeded on issue tracebacks/symbols)"),

    # ── Summary-search: rank by function SUMMARIES (purpose), not raw code ───
    # The "a developer recalls what a function DOES, then fetches it" idea. Same
    # tree-sitter chunking as `bm25` (fair), and results are presented identically
    # (bare FQNs, no summary preview) so the ONLY variable is the ranking SOURCE:
    #   summary-bm25  vs `bm25`         → search summaries vs search code (matcher
    #                                     held lexical) — does the summary
    #                                     representation bridge the prose↔identifier gap?
    #   summary-dense vs summary-bm25   → add semantic matching (the point of summaries).
    # Summaries are LOCAL/deterministic in the agent loop (the async LLM worker
    # never runs in an isolated workspace). The local-vs-Ollama summary-quality
    # delta is measured cheaply by the retrieval-only probe, NOT by an agent arm.
    # Run in sg-env (summary-dense needs sentence-transformers, like hybrid/sg-embed).
    "summary-dense": Arm("summary-dense", "summary-dense",
                         "Summary-search (dense over summaries)", strong=True),
    "summary-bm25": Arm("summary-bm25", "summary-bm25",
                         "Summary-search (BM25 over local summaries)"),
    "summary-llm-dense": Arm("summary-llm-dense", "summary-llm-dense",
                         "Summary-search (dense over LLM summaries)", strong=True),
    "summary-llm-bm25": Arm("summary-llm-bm25", "summary-llm-bm25",
                         "Summary-search (BM25 over LLM summaries)"),

    # Single-shot SG (no agent loop): retrieve once → one generation → patch.
    # This is the "is the agent worth it" measure (agent vs no-agent) — which is
    # also where SG's internal query routing would matter, since with no agent
    # there is nothing else doing the adaptive handling. Run via
    # run_singleshot.py, not run_stage; recorded as this arm so it lands in the
    # same aggregate/plots tables next to `sg`.
    "sg-noagent":   Arm("sg-noagent",   "sg",           "SG single-shot (no agent)"),
    # NOTE: graphify (graphifyy v0.8.17, https://github.com/safishamsi/graphify)
    # was considered but DROPPED — it is packaged as a CLI-only AI-coding-assistant
    # skill with 0 public python symbols, not a programmable retrieval library.
    # Wrapping its CLI as a controlled search_code backend would require fragile
    # subcommand stitching that isn't directly comparable to library-form
    # retrievers. The graph-competitor slot is filled by cbmem.

    # Aider RepoMap — tree-sitter + PageRank, the design philosophy closest
    # to SG. Aider is the most-used CLI coding agent on PyPI. Beating its
    # repo-map is the strongest "we beat the prior art" claim available.
    # pip install aider-chat
    "aider":   Arm("aider",   "aider",   "Aider RepoMap (tree-sitter+PageRank)",
                   strong=True),
    # Graphify — knowledge-graph competitor with a "70x token reduction" claim,
    # tested END-TO-END via its native graph tools (graphify_search/explain).
    "graphify": Arm("graphify", "graphify",
                    "Graphify (knowledge-graph, 70x-token claim)", strong=True),
}


# ── staged run plan ────────────────────────────────────────────────────────
# Each stage is a self-contained, publishable-at-its-level result. Stages add;
# they never replace. If credits run out after any stage you still have a
# coherent paper at that tier. See STAGES.md for the rationale.

@dataclass
class Stage:
    name: str
    arms: List[str]
    n_tasks: int
    benchmark: str                 # "swebench" | "contextbench" | ...
    note: str
    repeats: int = 1               # >1 → variance runs
    models: List[str] = field(default_factory=lambda: ["main"])


# SWE-bench task count — FIXED across all stages. 150 stratified tasks is
# statistically sufficient for paired significance tests against strong
# baselines. Compute is spent on BREADTH (baselines/ablations/2nd benchmark),
# not on more SWE-bench tasks. See STAGES.md for the rationale.
SWEBENCH_N = 150

STAGES: Dict[str, Stage] = {
    # Pre-AMD assessment — the single most informative cheap run. Answers:
    # does SG's retrieval edge over lexical (bm25) AND dense-RAG (hybrid) hold
    # beyond the 5-task smoke? Retrieval/consolidation metrics only — no
    # verify.py / pass@1 needed (that requires the SWE-bench Docker harness).
    # Run on whatever model is served via SG_EVAL_MODEL; the "main" label is
    # cosmetic for the run-id. 3 arms × 15 tasks = 45 runs.
    "0-assess": Stage(
        "0-assess", ["sg", "bm25", "hybrid"], 15, "swebench",
        "Pre-AMD assessment — SG vs lexical-RAG (bm25) vs dense-RAG (hybrid). "
        "Retrieval + consolidation metrics; confirms the smoke precision gap "
        "holds at larger N before committing AMD budget.",
    ),
    # Local full comparison (7B, then 14B by swapping SG_EVAL_MODEL). These five
    # arms ALL require the sentence-transformers stack (SG embeddings + hybrid
    # dense), so they share one env. 5 arms × 30 tasks = 150 runs. Retrieval +
    # efficiency + (weak) consolidation; pass@1 from the SWE-bench Docker run.
    "0-full": Stage(
        "0-full", ["sg", "bm25", "grep", "none", "hybrid"], 30, "swebench",
        "Local comparison (7B/14B) — SG vs lexical (bm25/grep) vs no-retrieval "
        "(none) vs dense-RAG (hybrid) × 30 tasks. pass@1 deferred to the AMD run.",
    ),
    # SG component ablations — isolate which part of SG carries the gain. Run on
    # the SAME 30 tasks; compare against the `sg` arm from 0-full. sg-nosummary
    # is the C2 ablation (search results drop the tier-2 summaries); sg-nograph
    # / sg-norerank toggle graph expansion / centrality rerank.
    "0-ablation": Stage(
        "0-ablation",
        ["sg-fullgraph", "sg-nograph", "sg-norerank", "sg-nosummary", "sg-noembed"],
        30, "swebench",
        "SG ablations (local) — eager graph / no graph / centrality / summary "
        "/ embeddings contributions, vs the default gated `sg` arm. "
        "(Agent-vs-no-agent is measured separately by sg-noagent.)",
    ),
    # Codebase-Memory MCP baseline — the closest published competitor. Wrapped
    # via its CLI binary (subprocess), so no Python-env conflict; runs from the
    # main env. Merges with the others via `aggregate` (no --stage).
    "0-cbmem": Stage(
        "0-cbmem", ["cbmem"], 30, "swebench",
        "Codebase-Memory (MCP graph) baseline — needs the binary on PATH "
        "(CBMEM_BIN). Closest published competitor to SG.",
    ),
    "0-learned": Stage(
        "0-learned", ["sg-learned"], 30, "swebench",
        "Test just the learned curator against the 0-full baselines."
    ),
    "0-weakfallback": Stage(
        "0-weakfallback", ["sg-weakfallback"], 30, "swebench",
        "Test the gated weak-entity recall booster vs the `sg` baseline — does it "
        "recover semantic-mismatch misses WITHOUT diluting precision? (default off)"
    ),
    # ──────────────────────────────────────────────────────────────────────
    # ── PAPER-FACING STAGES (canonical names, consistent arm sets) ────────
    # ──────────────────────────────────────────────────────────────────────
    #
    # The seven main arms span four retrieval families plus the closest prior
    # art and the closest published competitor:
    #
    #   sg        — SkeletonGraph (ours)
    #   bm25      — Flat BM25 (lexical, function-level)
    #   grep      — Ripgrep-style lexical (file-level)
    #   hybrid    — Dense+Rerank (BM25 ∪ SBert + cross-encoder rerank, file-level)
    #   none      — No-Retrieval (control)
    #   cbmem     — Codebase-Memory (closest published graph competitor)
    #   aider     — Aider RepoMap (closest prior art: tree-sitter+PageRank)
    #
    # The five SG ablations isolate which component carries the gain:
    #
    #   sg-fullgraph  — eager graph expansion (no gating)
    #   sg-nograph    — graph expansion fully off
    #   sg-norerank   — no centrality rerank
    #   sg-nosummary  — no tier-2 summaries surfaced
    #   sg-noembed    — no semantic embeddings (lexical+graph only)
    #
    # sg-noagent (single-shot, no agent loop) is its own ablation but uses
    # run_singleshot.py instead of run_stage.py — handled outside the stage
    # system so it can't be passed as --stage. See docstring of that script.

    # IMPORTANT — env split. Arms are grouped by what env they need so a run
    # never silently degrades because a backend's dependency is missing:
    #   • baseline / ablation / trial / singleshot → sg-env (sentence-
    #     transformers + skeletongraph). Pure-Python, no external binaries.
    #   • comparators → each needs its OWN env: cbmem needs the codebase-memory
    #     binary on PATH (CBMEM_BIN); aider needs aider-chat installed (its
    #     huggingface_hub pin conflicts with ours — use a SEPARATE venv).
    # Mixing cbmem/aider into `baseline` is exactly what bricked them on
    # ContextBench (binary/lib missing → silent recall=0). Keep them separate.

    "baseline": Stage(
        "baseline",
        ["sg", "bm25", "grep", "hybrid", "none"],
        30, "swebench",
        "BASELINE — 5-arm core comparison: SG vs 4 retrieval families "
        "(BM25, grep, Dense+Rerank, No-Retrieval). All run in the SAME env "
        "(sg-env); no external binaries. The paper's main table. Run the "
        "closest published systems via the `comparators` stage.",
    ),

    "comparators": Stage(
        "comparators",
        ["cbmem", "aider"],
        30, "swebench",
        "COMPARATORS — closest published systems. Run SEPARATELY because each "
        "needs its own env: cbmem needs the binary on PATH (set $env:CBMEM_BIN); "
        "aider needs aider-chat installed (separate venv — its huggingface_hub "
        "pin conflicts with ours). Compare against `sg` from `baseline`. "
        "Preflight each backend (selftest) BEFORE launching or you get silent "
        "recall=0.",
    ),

    "ablation": Stage(
        "ablation",
        ["sg-fullgraph", "sg-nograph", "sg-norerank", "sg-nosummary", "sg-noembed"],
        30, "swebench",
        "SG ABLATION — five SG variants with exactly one component disabled. "
        "Compare against the `sg` arm in `baseline`. Single-shot (no agent "
        "loop) is run separately via `run_singleshot.py --all` (arm sg-noagent).",
    ),

    # ── FINAL run plan (70B NIM @ 100 tasks; later Qwen-32B) ────────────────
    # `sg` is the LEAN product default. baseline = the definitive headline
    # table; ablation isolates each design choice around lean `sg` (run at
    # --limit 30 to save cost — recall/precision are stable there, only pass@1
    # needs 100); comparators run LATER in their own envs. Use SG_EVAL_STRICT=1
    # so any arm that wants embeddings (sg-full/sg-embed) aborts rather than
    # silently degrading.
    "final": Stage(
        "final",
        ["sg", "bm25", "grep", "hybrid", "none", "sg-chain"],
        100, "swebench",
        "FINAL BASELINE — the definitive 100-task headline: lean SG vs 4 "
        "retrieval families, plus sg-chain (path-aware SG+BM25, current best). "
        "Shard across 4 terminals/keys with --shard k/4. Compare vs "
        "`final-comparators` (cbmem/aider) separately.",
    ),
    "final-ablation": Stage(
        "final-ablation",
        ["sg-nograph", "sg-norerank"],
        100, "swebench",
        "FINAL ABLATION — the two ablations kept for further tests: −graph "
        "(does gated graph earn its keep on recall?) and −rerank (PageRank's "
        "marginal value). Compare vs lean `sg`. (sg-full/summary/embed/fullgraph "
        "retired — their question is settled; defs kept for old-tag aggregation.)",
    ),
    "final-summary": Stage(
        "final-summary",
        ["summary-bm25", "summary-dense", "summary-llm-bm25", "summary-llm-dense"],
        100, "swebench",
        "FINAL SUMMARY-SEARCH — the new contribution: rank by function SUMMARIES "
        "(purpose) instead of code. summary-bm25 isolates the summary representation "
        "vs the existing code-`bm25`; summary-dense adds semantic matching. Runs "
        "into the SAME SG_EVAL_RUN_TAG as `final` so aggregate folds both into one "
        "table — 100 tasks, fair n=100 vs n=100. Shard with --shard k/4 across "
        "terminals/keys. Summaries are LOCAL/deterministic in-loop (Ollama quality "
        "delta = the retrieval probe). Run in sg-env (summary-dense needs SBert).",
    ),
    "final-rerank": Stage(
        "final-rerank",
        ["sg-rerank"],
        100, "swebench",
        "FINAL RERANK — sg-rerank (bm25 recall pool reordered by SG structural "
        "confirmation). Tests whether generate-then-rerank gets bm25's recall AND "
        "sg-chain's rank. Runs into the same tag as `final`; shard with --shard k/4.",
    ),
    "final-v2": Stage(
        "final-v2",
        ["sg", "bm25", "grep", "hybrid", "none", "cbmem", "aider"],
        100, "swebench",
        "FINAL v2 — SYSTEMS comparison: each pipeline uses its NATIVE tools "
        "(sg: search_code + read_symbol + expand; cbmem: cbmem_search/trace/snippet/"
        "arch; aider: PageRank repo-map INJECTED into the prompt, no search tool). "
        "Baselines bm25/grep/hybrid/none keep the standard 5 tools. Common substrate "
        "= model + tasks + edit/submit/verify. Split by env: sg-env runs "
        "sg/bm25/grep/hybrid/none/cbmem (set CBMEM_BIN); aider-env runs aider. "
        "Per-arm tool surface in profiles.py. Fresh tag (e.g. nemotron_v2).",
    ),
    "sg-concepts": Stage(
        "sg-concepts",
        ["sg-chain", "sg-rerank", "summary-dense", "sg-hybrid-fusion",
         "sg-dense-rerank", "sg-keyword-dense", "sg-seed", "sg-embed"],
        100, "swebench",
        "SG CONCEPTS — re-test the best retrieval CONCEPTS in the NATIVE harness "
        "(each gets SG's read_symbol/expand): sg-chain (structural+lexical fusion + "
        "graph-path evidence), sg-rerank (bm25 recall pool -> SG structural rerank), "
        "summary-dense (intent/purpose-layer search), sg-hybrid-fusion / sg-dense-rerank "
        "/ sg-keyword-dense (Jina dense variants), sg-seed (issue-traceback anchors), "
        "sg-embed (structural pool + dense rerank). Run into the SAME tag as final-v2 "
        "to compare against `sg` + baselines."
    ),
    "final-comparators": Stage(
        "final-comparators",
        ["cbmem", "aider", "graphify"],
        100, "swebench",
        "FINAL COMPARATORS — cbmem (CBMEM_BIN on PATH) + aider (separate venv). "
        "Run LATER, own envs, after preflight --selftest. ALSO verify their "
        "recall/precision extraction — pass@1 looks fine but retrieval may be "
        "under-counted by the production-pipeline parser.",
    ),

    # ── Universal stage — every registered arm, 100 tasks ───────────────
    # Use --only-arms to pick exactly what you want to run; --skip-arms to
    # exclude env-incompatible arms for this terminal.  Examples:
    #
    #   python -m eval.agent.run_stage --stage v --only-arms sg sg-rerank
    #   python -m eval.agent.run_stage --stage v --only-arms cbmem aider graphify
    #   python -m eval.agent.run_stage --stage v --skip-arms cbmem aider graphify
    #   python -m eval.agent.run_stage --stage v --only-arms sg bm25 grep none hybrid --shard 1/6
    #
    # Env split reminder (cbmem/aider/graphify need their own envs — run those
    # in a separate terminal with only those arms selected):
    #   sg-env  : sg sg-rerank sg-chain sg-seed sg-nograph sg-norerank sg-full
    #             sg-embed sg-summary sg-hybrid-fusion sg-dense-rerank
    #             sg-keyword-dense summary-dense summary-bm25
    #             summary-llm-dense summary-llm-bm25
    #             bm25 grep none hybrid
    #   cbmem-env : cbmem  (needs CBMEM_BIN on PATH)
    #   aider-env : aider  (needs aider-chat, separate venv)
    #   graphify-env: graphify  (needs graphifyy, GRAPHIFY_BIN)
    "v": Stage(
        "v",
        [
            # ── baselines ──────────────────────────────────────────────────
            "sg", "bm25", "grep", "none", "hybrid",
            # ── sg headline operating points ───────────────────────────────
            "sg-rerank", "sg-chain", "sg-seed", "fusion",
            # ── sg ablations ───────────────────────────────────────────────
            "sg-nograph", "sg-norerank", "sg-full", "sg-embed", "sg-summary",
            "sg-fullgraph", "sg-nosummary", "sg-noembed", "sg-weakfallback",
            # ── dense variants ─────────────────────────────────────────────
            "sg-hybrid-fusion", "sg-dense-rerank", "sg-keyword-dense",
            # ── summary-search ─────────────────────────────────────────────
            "summary-bm25", "summary-dense",
            "summary-llm-bm25", "summary-llm-dense",
            # ── external competitors (own envs — use --only-arms) ──────────
            "cbmem", "aider", "graphify",
        ],
        100, "swebench",
        "UNIVERSAL — every arm in one stage. Use --only-arms to target a "
        "subset and --skip-arms to exclude env-incompatible arms. "
        "--shard k/N splits tasks across terminals for multi-key runs. "
        "Env split: sg/baselines in sg-env; cbmem/aider/graphify each need "
        "their own env — run those separately with --only-arms.",
    ),

    "smoke": Stage(
        "smoke",
        ["sg", "bm25", "grep", "hybrid", "none"],
        SWEBENCH_N, "swebench",
        "SMOKE GATE — same 5 arms as `baseline`, scoped to first N tasks via "
        "`--limit 10`. Run BEFORE any big AMD spend; abort + fix if any arm is "
        "wedged. ~$1 on MI300X. (Comparators have their own preflight.)",
    ),

    "contextbench": Stage(
        "contextbench",
        ["sg", "bm25", "grep", "hybrid", "none"],
        60, "contextbench",
        "CONTEXTBENCH — 2nd benchmark, same 5 baseline arms × 60 Python tasks. "
        "Run with `--dataset eval/datasets/contextbench.jsonl`. Confirms the "
        "SG win isn't SWE-bench-specific. Run cbmem/aider here via the "
        "`comparators` stage + the same `--dataset` flag, in their own envs.",
    ),

    "variance": Stage(
        "variance",
        ["sg", "bm25", "hybrid"],
        20, "swebench",
        "VARIANCE APPENDIX — 3 sg-env arms × 20 tasks × 3 seeds. Quantifies "
        "vLLM/NIM serving non-determinism for the paper's noise-floor "
        "appendix. (cbmem/aider excluded — own env; add them manually if "
        "needed.) Spend only after baseline + ablation land.",
        repeats=3,
    ),

    # ──────────────────────────────────────────────────────────────────────
    # ── TARGETED AMD MI300X BUDGET STAGES ─────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────

    "amd-workshop-300": Stage(
        "amd-workshop-300",
        ["sg", "bm25", "cbmem", "graphify", "summary-llm-bm25"],
        300, "swebench",
        "WORKSHOP TARGET — 300 tasks (equivalent to SWE-bench Lite). "
        "Tests core SG, flat BM25, graph/MCP baselines, and our LLM summaries.",
    ),
    "amd-conference-500": Stage(
        "amd-conference-500",
        ["sg", "bm25", "cbmem", "graphify", "summary-llm-bm25", "hybrid"],
        500, "swebench",
        "CONFERENCE TARGET — Full 500 tasks (SWE-bench Verified). "
        "Adds hybrid-dense to the workshop mix for statistical significance.",
    ),
    "amd-pro-100": Stage(
        "amd-pro-100",
        ["sg", "bm25"],
        100, "swebench_pro",
        "CONFERENCE SCALING TARGET — 100 SWE-bench Pro tasks. "
        "Massive repositories to prove architecture scales where BM25 breaks.",
    ),
    "amd-nim-150": Stage(
        "amd-nim-150",
        ["sg", "bm25", "cbmem", "graphify", "summary-llm-bm25"],
        150, "swebench",
        "NIM FALLBACK — 150 tasks (SWE-bench Verified). "
        "Reduced subset for a 1:1 comparison against 120B NIM model without "
        "hitting API rate limits or budget caps.",
    ),

    # ──────────────────────────────────────────────────────────────────────
    # ── LEGACY ALIASES (kept for backward compatibility with old commands) ─
    # ──────────────────────────────────────────────────────────────────────
    # Old runbooks reference these names. New work should use the canonical
    # stages above.

    "0-smoke":        None,   # alias → smoke   (set below)
    "0-full":         None,   # alias → baseline
    "0-assess":       None,   # alias → baseline with --limit 15
    "0-ablation":     None,   # alias → ablation
    "0-cbmem":        None,   # narrow cbmem-only runs
    "1a-workshop":    None,   # alias → baseline at full N
    "1b-conference":  None,   # alias → ablation at full N
    "2-competitor":   None,   # alias → cbmem-only at full N

    # Legacy specialized arms (rarely used)
    "0-learned": Stage(
        "0-learned", ["sg-learned"], 30, "swebench",
        "Learned curator ablation. Needs eval/curator/curator_model.pkl."
    ),
    "0-weakfallback": Stage(
        "0-weakfallback", ["sg-weakfallback"], 30, "swebench",
        "Gated weak-entity recall booster (default off in `sg`)."
    ),
    "3-further": Stage(
        "3-further", ["sg", "bm25", "hybrid", "sg-learned"], 60, "swebench",
        "Top-tier 3x variance + learned curator. Spend only if 1-2 warrant.",
        repeats=3,
    ),
}

# Resolve legacy aliases to canonical stages.
STAGES["0-smoke"]        = STAGES["smoke"]
STAGES["0-full"]         = STAGES["baseline"]
STAGES["0-assess"]       = STAGES["baseline"]
STAGES["0-ablation"]     = STAGES["ablation"]
STAGES["1a-workshop"]    = STAGES["baseline"]
STAGES["1b-conference"]  = STAGES["ablation"]
STAGES["0-cbmem"] = Stage(
    "0-cbmem", ["cbmem"], 30, "swebench",
    "cbmem-only — useful when the cbmem binary changes and you want to "
    "refresh just this arm without re-running the others.",
)
STAGES["2-competitor"] = STAGES["0-cbmem"]
