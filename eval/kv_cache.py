"""Analytical KV-cache footprint + inference serving-density calculator.

This is the systems backbone of the SkeletonGraph paper. It needs NO GPU —
the KV-cache size of a transformer request is fully determined by the model
config and the number of tokens in context:

    KV_bytes(per request) = n_tokens
                          * n_layers
                          * 2                  # K and V
                          * n_kv_heads          # GQA: far fewer than Q heads
                          * head_dim
                          * dtype_bytes

Serving density on one GPU (the metric reviewers care about):

    concurrent_requests = (VRAM_total - weights_bytes) / KV_bytes_per_request

Run:
    python eval/kv_cache.py                       # full table
    python eval/kv_cache.py --model llama3-70b --tokens 8000 120000
    python eval/kv_cache.py --csv results/kv_cache.csv

The point the paper makes: SkeletonGraph cuts `n_tokens` from ~100k (naive
long-context stuffing) to ~10-15k (knowledge-aware structural retrieval) at
equal task success — and that linearly multiplies how many requests fit on a
GPU. This file produces those numbers deterministically and reproducibly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ── Model configs ────────────────────────────────────────────────────────
# Values from public model configs (config.json). GQA models have far fewer
# KV heads than attention heads — that is what makes KV cache tractable.


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params_billion: float
    n_layers: int
    n_kv_heads: int       # GQA key/value heads (NOT attention heads)
    head_dim: int
    weight_dtype_bytes: float = 2.0   # fp16/bf16 = 2; fp8 = 1; int4 ≈ 0.5
    kv_dtype_bytes: float = 2.0       # KV cache usually fp16; fp8 KV = 1

    def weights_bytes(self) -> float:
        return self.params_billion * 1e9 * self.weight_dtype_bytes

    def kv_bytes_per_token(self) -> float:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_dtype_bytes

    def kv_bytes(self, n_tokens: int) -> float:
        return self.kv_bytes_per_token() * n_tokens


MODELS: Dict[str, ModelConfig] = {
    # Llama 3.1 family — GQA, 8 KV heads, head_dim 128
    "llama3-8b":  ModelConfig("Llama-3.1-8B",  8.0,  32, 8, 128),
    "llama3-70b": ModelConfig("Llama-3.1-70B", 70.0, 80, 8, 128),
    # Qwen2.5-Coder family — GQA
    "qwen-coder-7b":  ModelConfig("Qwen2.5-Coder-7B",  7.6,  28, 4, 128),
    "qwen-coder-32b": ModelConfig("Qwen2.5-Coder-32B", 32.0, 64, 8, 128),
    # A large dense model for the "long context hurts" worst case
    "llama3-405b": ModelConfig("Llama-3.1-405B", 405.0, 126, 8, 128),
}

# Common datacenter GPUs (usable VRAM in bytes — leave ~5% headroom)
GPUS: Dict[str, float] = {
    "A100-40GB": 40e9 * 0.95,
    "A100-80GB": 80e9 * 0.95,
    "H100-80GB": 80e9 * 0.95,
    "H200-141GB": 141e9 * 0.95,
}

# Context-length scenarios the paper compares
SCENARIOS: Dict[str, int] = {
    "SG (structural retrieval)": 12_000,
    "flat RAG":                  35_000,
    "long-context stuffing":     128_000,
}


# ── Core computation ──────────────────────────────────────────────────────


def serving_density(model: ModelConfig, gpu_vram: float, n_tokens: int,
                     tensor_parallel: int = 1) -> float:
    """Concurrent requests that fit on `tensor_parallel` GPUs at n_tokens context.

    Returns 0.0 if the weights don't even fit (need more GPUs).
    """
    total_vram = gpu_vram * tensor_parallel
    free_for_kv = total_vram - model.weights_bytes()
    if free_for_kv <= 0:
        return 0.0
    kv = model.kv_bytes(n_tokens)
    if kv <= 0:
        return 0.0
    return free_for_kv / kv


def min_gpus_for_weights(model: ModelConfig, gpu_vram: float) -> int:
    """Minimum GPU count just to hold the weights."""
    import math
    return max(1, math.ceil(model.weights_bytes() / gpu_vram))


# ── Reporting ─────────────────────────────────────────────────────────────


def _human_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"


def print_table(model_key: str, gpu_key: str = "A100-80GB") -> List[dict]:
    """Print the headline serving-density table for one model + GPU."""
    model = MODELS[model_key]
    gpu_vram = GPUS[gpu_key]
    tp = min_gpus_for_weights(model, gpu_vram)

    print(f"\n{'=' * 72}")
    print(f"Model: {model.name}   ({model.params_billion}B params)")
    print(f"GPU:   {gpu_key} x{tp}  (tensor-parallel to hold weights)")
    print(f"  weights:           {_human_bytes(model.weights_bytes())}")
    print(f"  KV cache per token: {_human_bytes(model.kv_bytes_per_token())}")
    print(f"{'=' * 72}")
    print(f"{'scenario':<28}{'ctx tokens':>12}{'KV/request':>14}{'req/GPU-set':>14}")
    print("-" * 72)

    rows = []
    baseline_density = None
    for label, ntok in SCENARIOS.items():
        kv = model.kv_bytes(ntok)
        density = serving_density(model, gpu_vram, ntok, tensor_parallel=tp)
        if "stuffing" in label:
            baseline_density = density
        print(f"{label:<28}{ntok:>12,}{_human_bytes(kv):>14}{density:>14.1f}")
        rows.append({
            "model": model.name, "gpu": f"{gpu_key}x{tp}",
            "scenario": label, "ctx_tokens": ntok,
            "kv_bytes_per_request": int(kv),
            "concurrent_requests": round(density, 2),
        })

    # The money number: how much denser SG is vs naive stuffing
    if baseline_density and baseline_density > 0:
        print("-" * 72)
        for r in rows:
            r["density_vs_stuffing"] = round(r["concurrent_requests"] / baseline_density, 2)
        sg_row = next((r for r in rows if "SG" in r["scenario"]), None)
        if sg_row:
            print(f"  -> SkeletonGraph serves {sg_row['density_vs_stuffing']:.1f}x "
                  f"more concurrent requests per GPU than long-context stuffing,")
            print(f"     at equal task success (see SWE-bench / ContextBench results).")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="KV-cache + serving-density calculator")
    ap.add_argument("--model", choices=list(MODELS), default=None,
                    help="Single model (default: all)")
    ap.add_argument("--gpu", choices=list(GPUS), default="A100-80GB")
    ap.add_argument("--tokens", type=int, nargs="*", default=None,
                    help="Custom context lengths to evaluate")
    ap.add_argument("--csv", type=str, default=None, help="Write rows to CSV")
    args = ap.parse_args()

    if args.tokens:
        SCENARIOS.clear()
        for t in args.tokens:
            SCENARIOS[f"{t:,} tokens"] = t

    model_keys = [args.model] if args.model else list(MODELS)
    all_rows: List[dict] = []
    for mk in model_keys:
        all_rows.extend(print_table(mk, args.gpu))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {out}")


if __name__ == "__main__":
    main()
