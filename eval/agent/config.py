"""Central configuration for the agentic eval harness.

Everything tunable lives here: arms, models, the staged run plan, the price
sheet used to impute cost, and paths. Nothing else should hard-code these.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
RUNS_DIR = EVAL_DIR / "results" / "agent"                   # per-run trajectories
WORKSPACE_ROOT = EVAL_DIR / "datasets" / "_agent_work"      # isolated per-run checkouts


# ── model endpoint ─────────────────────────────────────────────────────────
# The harness talks to ONE OpenAI-compatible endpoint (vLLM). Swap models by
# changing MODEL_NAME + restarting vLLM with the matching --model.

API_BASE = os.environ.get("SG_EVAL_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("SG_EVAL_API_KEY", "EMPTY")        # vLLM ignores it
MODEL_NAME = os.environ.get("SG_EVAL_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

TEMPERATURE = 0.0          # deterministic — pin this, never change mid-study
SEED = 42
MAX_TURNS = 40             # ReAct step ceiling; tasks hitting it = a failure mode
REQUEST_TIMEOUT = 300      # seconds per model call


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
    "grep":   Arm("grep",   "grep",   "Keyword grep (naive)"),
    "none":   Arm("none",   "none",   "No retrieval (long-context)"),
    "hybrid": Arm("hybrid", "hybrid", "Hybrid RAG (BM25+dense+rerank)", strong=True),
    "aider":  Arm("aider",  "aider",  "Aider repo-map", strong=True),
    # SG ablations — same SG retrieval with one component disabled. The backend
    # toggle is not yet wired (tools._retrieve raises NotImplementedError); do
    # that before running stage 2-ablation.
    "sg-nograph":   Arm("sg-nograph",   "sg-nograph",   "SG (no graph expansion)"),
    "sg-norerank":  Arm("sg-norerank",  "sg-norerank",  "SG (no centrality rerank)"),
    "sg-nosummary": Arm("sg-nosummary", "sg-nosummary", "SG (no summaries)"),
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
    # dense), so they share one env. aider-chat hard-pins huggingface-hub==1.4.1
    # which is incompatible with that stack — it runs separately as `0-aider`
    # from its own venv. 5 arms × 30 tasks = 150 runs. Retrieval + efficiency +
    # (weak) consolidation; pass@1 deferred to the SWE-bench Docker run on AMD.
    "0-full": Stage(
        "0-full", ["sg", "bm25", "grep", "none", "hybrid"], 30, "swebench",
        "Local comparison (7B/14B) — SG vs lexical (bm25/grep) vs no-retrieval "
        "(none) vs dense-RAG (hybrid) × 30 tasks. pass@1 deferred to the AMD run.",
    ),
    # aider repo-map baseline — run from an ISOLATED venv (aider-chat's
    # huggingface-hub==1.4.1 pin conflicts with the sentence-transformers stack).
    # Writes into the same RUNS_DIR; combine with `aggregate` (no --stage) for the
    # full 6-arm table. The aider arm needs no sentence-transformers.
    "0-aider": Stage(
        "0-aider", ["aider"], 30, "swebench",
        "Aider repo-map baseline (isolated venv) — merges with 0-full for the "
        "complete 6-arm comparison.",
    ),
    # Stage 0 is free/CPU and done (eval/run_stage0.py + the IDE smoke).
    "1-core": Stage(
        "1-core", ["sg", "bm25", "grep", "none"], SWEBENCH_N, "swebench",
        "WORKSHOP tier — controlled agentic result vs floor baselines. "
        "Yields Axes 1/2/3/5.",
    ),
    "2-strong": Stage(
        "2-strong", ["sg", "hybrid", "aider"], SWEBENCH_N, "swebench",
        "CONFERENCE tier (a) — vs strong deployed baselines "
        "(hybrid RAG, Aider repo-map).",
    ),
    "2-ablation": Stage(
        "2-ablation", ["sg-nograph", "sg-norerank", "sg-nosummary"],
        SWEBENCH_N, "swebench",
        "CONFERENCE tier (b) — which SG component carries the gain. "
        "Ablation backends need wiring (see tools._retrieve).",
    ),
    "2-variance": Stage(
        "2-variance", ["sg", "bm25", "hybrid"], 60, "swebench",
        "CONFERENCE tier (c) — 3x repeats on a subset → mean±std + McNemar/CIs.",
        repeats=3,
    ),
    "3-benchmark": Stage(
        "3-benchmark", ["sg", "bm25", "hybrid"], SWEBENCH_N, "contextbench",
        "TOP-TIER (a) — second benchmark; generalisation beyond SWE-bench. "
        "Needs the contextbench loader wired.",
    ),
    "3-scale": Stage(
        "3-scale", ["sg", "bm25"], 60, "swebench",
        "TOP-TIER (b) — does SG's gain hold on a smaller model. The 7B leaves "
        "the MI300X mostly idle, so co-host it and run this alongside stage 1/2.",
        models=["qwen-7b"],
    ),
}
