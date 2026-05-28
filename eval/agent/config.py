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
# Workspace root is ALSO tag-namespaced. run_id uses model="main" for every
# model, so two runs (e.g. 7B + NIM-70B) on the same (task, arm) would otherwise
# share one checkout dir and clobber each other if run concurrently. Tagging the
# root lets different SG_EVAL_RUN_TAG runs execute fully in parallel.
WORKSPACE_ROOT = (EVAL_DIR / "datasets" / "_agent_work" / _RUN_TAG
                  if _RUN_TAG else EVAL_DIR / "datasets" / "_agent_work")  # isolated per-run checkouts


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
MAX_TURNS = 40             # ReAct step ceiling; tasks hitting it = a failure mode
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
    "sg":     Arm("sg",     "sg",     "SkeletonGraph (structural)"),
    "bm25":   Arm("bm25",   "bm25",   "Flat BM25"),
    "grep":   Arm("grep",   "grep",   "Ripgrep-style lexical"),
    "none":   Arm("none",   "none",   "No retrieval (long-context)"),
    "hybrid": Arm("hybrid", "hybrid", "Hybrid RAG (BM25+dense+rerank)", strong=True),
    "cbmem":  Arm("cbmem",  "cbmem",  "Codebase-Memory (MCP graph)", strong=True),
    # SG ablations — same SG retrieval with exactly one component disabled
    # (wired in tools._sg_config). Each isolates a contribution.
    "sg-nograph":   Arm("sg-nograph",   "sg-nograph",   "SG (no graph expansion)"),
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
    # ── Named tiers (paper-facing names for tables/CLI) ──────────────────────
    # `baseline` = the five core comparison arms. Functionally identical to
    # 0-full; the old name is preserved for backward compat with existing results.
    "baseline": Stage(
        "baseline", ["sg", "bm25", "grep", "none", "hybrid"], 30, "swebench",
        "Baseline tier — SG vs lexical (bm25/grep) vs no-retrieval (none) vs "
        "dense-RAG (hybrid). Same arms as 0-full; cleaner name for paper tables.",
    ),
    # `full` = baseline + Graphify (knowledge-graph RAG) + cbmem (MCP graph).
    # Three different graph strategies side-by-side with lexical/dense floors.
    # Requires both external tools to be set up first (see their backends).
    "full": Stage(
        "full",
        ["sg", "bm25", "grep", "none", "hybrid", "cbmem"],
        30, "swebench",
        "Full comparison — 5 baseline arms + Graphify knowledge-graph RAG "
        "+ cbmem (Codebase-Memory MCP). External tools must be set up first.",
    ),
    # ── AMD staged plan (the real spend) ────────────────────────────────────
    # Stage 1 = 1a + 1b, run in PARALLEL on the MI300X (192 GB → many isolated
    # tasks concurrently; --workers 16-32). pass@1 via verify.py (Docker) after.
    # Full rationale, budget, and decisions: docs/PLAN.md.  Stage 0 (local
    # 7B/14B) is done; it gives the retrieval/efficiency baseline without pass@1.
    "1a-workshop": Stage(
        "1a-workshop", ["sg", "bm25", "grep", "none", "hybrid"], SWEBENCH_N,
        "swebench",
        "STAGE 1a (WORKSHOP) — SG vs floors (bm25/grep/none) + dense-RAG "
        "(hybrid). pass@1 + retrieval + efficiency. The defensible workshop core.",
    ),
    "1b-conference": Stage(
        "1b-conference",
        ["sg-fullgraph", "sg-nograph", "sg-norerank", "sg-nosummary", "sg-noembed"],
        SWEBENCH_N, "swebench",
        "STAGE 1b (CONFERENCE) — SG component ablations (eager-graph; C2 "
        "summaries; C3 graph/rerank/embed) vs the default gated `sg`. Run in "
        "PARALLEL with 1a. Agent-vs-no-agent (sg-noagent) via run_singleshot.",
    ),
    "2-competitor": Stage(
        "2-competitor", ["cbmem"], SWEBENCH_N, "swebench",
        "STAGE 2 — closest published graph competitor (cbmem / CodeCompass). "
        "Plus ContextBench: re-run 1a arms with "
        "--dataset eval/datasets/contextbench.jsonl (2nd benchmark).",
    ),
    "3-further": Stage(
        "3-further", ["sg", "bm25", "hybrid", "sg-learned"], 60, "swebench",
        "STAGE 3 (FURTHER / TOP-TIER) — 3x variance (mean±std + McNemar/CIs), "
        "the learned curator (sg-learned, see docs/CURATOR.md), and a 2nd "
        "language. Spend only if 1-2 warrant the push.",
        repeats=3,
    ),
    # ── AMD smoke pre-check (single most important pre-stage) ────────────────
    # Same 7 arms as `full`, but ONLY the first N tasks of the 150-task pool.
    # Run this FIRST on the MI300X; if 1a-arms look sane and verify works,
    # proceed to the real stages; if not, abort before burning budget.
    # Use --limit 10 on the run_stage call to scope to 10 tasks; the stage
    # itself stays open-ended so you can re-run with --limit 25 if needed.
    "0-smoke": Stage(
        "0-smoke",
        ["sg", "bm25", "grep", "none", "hybrid", "cbmem"],
        SWEBENCH_N, "swebench",
        "AMD SMOKE — first N tasks across ALL arms (baseline + cbmem + "
        "graphify). Use `run_stage --stage 0-smoke --limit 10` to scope.",
    ),
    # ── ContextBench stage with graph competitors included ───────────────────
    # The ContextBench pass becomes the 2nd benchmark in the paper. We include
    # cbmem and graphify here too so the graph-competitor comparison spans
    # BOTH SWE-bench and ContextBench (the reviewer ask: "does the result
    # generalise?"). Dataset is set via --dataset eval/datasets/contextbench.jsonl
    # at the run_stage call site.
    "contextbench": Stage(
        "contextbench",
        ["sg", "bm25", "grep", "none", "hybrid", "cbmem"],
        60, "contextbench",
        "ContextBench (2nd benchmark) — baseline + cbmem + graphify. "
        "Run with --dataset eval/datasets/contextbench.jsonl.",
    ),
    # ── 3-seed variance appendix ─────────────────────────────────────────────
    # 4 core arms × 20 tasks × 3 seeds = 240 runs. Quantifies serving-level
    # non-determinism (NIM/vLLM kernel-reduction non-bitwise reproducibility).
    # The reviewer-defensible "we know how noisy our numbers are" appendix.
    "variance": Stage(
        "variance", ["sg", "bm25", "hybrid", "cbmem"], 20, "swebench",
        "VARIANCE APPENDIX — 4 arms × 20 tasks × 3 seeds. Single most "
        "compact way to quantify NIM/vLLM non-determinism for the paper.",
        repeats=3,
    ),
}
