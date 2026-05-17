"""Axis 4 — measured serving density.

Turns the analytical kv_cache.py estimate into a real measurement: fires N
concurrent requests at the live vLLM endpoint for each context-length scenario
and records achieved throughput and how many requests fit before saturation.

    # restart vLLM at --max-model-len 16384 first, then:
    python -m eval.agent.vllm_bench --scenario sg
    python -m eval.agent.vllm_bench --scenario flat_rag
    python -m eval.agent.vllm_bench --scenario stuffing

Output: appends a row to eval/results/agent/serving_density.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from pathlib import Path

from . import config

# context sizes the paper compares (mirror kv_cache.py SCENARIOS)
SCENARIOS = {
    "sg":       12_000,
    "flat_rag": 35_000,
    "stuffing": 128_000,
}

OUT = config.RUNS_DIR / "serving_density.csv"


def _one_request(prompt_tokens: int, gen_tokens: int, results: list,
                 idx: int) -> None:
    from openai import OpenAI
    client = OpenAI(base_url=config.API_BASE, api_key=config.API_KEY,
                    timeout=config.REQUEST_TIMEOUT)
    # a cheap synthetic prompt of ~prompt_tokens (≈ 0.75 words/token)
    filler = ("def f(): pass  # context line\n" * (prompt_tokens // 8))
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user",
                       "content": filler + "\nSummarise the code above."}],
            max_tokens=gen_tokens,
            temperature=0.0,
        )
        dt = time.time() - t0
        out = resp.usage.completion_tokens if resp.usage else gen_tokens
        results[idx] = {"ok": True, "latency": dt, "out_tokens": out}
    except Exception as e:
        results[idx] = {"ok": False, "error": str(e)}


def sweep(scenario: str, concurrencies: list, gen_tokens: int = 256) -> None:
    ctx = SCENARIOS[scenario]
    print(f"Scenario '{scenario}': context≈{ctx} tokens")
    print("NOTE: start vLLM with --max-model-len >= this context first.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["scenario", "context_tokens", "concurrency",
                        "ok", "failed", "wall_s", "agg_tok_per_s",
                        "p50_latency_s", "p95_latency_s"])

        for c in concurrencies:
            results = [None] * c
            threads = [threading.Thread(target=_one_request,
                                        args=(ctx, gen_tokens, results, i))
                       for i in range(c)]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.time() - t0

            ok = [r for r in results if r and r.get("ok")]
            failed = c - len(ok)
            lat = sorted(r["latency"] for r in ok) or [0]
            total_out = sum(r["out_tokens"] for r in ok)
            agg = round(total_out / wall, 1) if wall else 0.0
            p50 = lat[len(lat) // 2]
            p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
            w.writerow([scenario, ctx, c, len(ok), failed, round(wall, 1),
                        agg, round(p50, 2), round(p95, 2)])
            print(f"  concurrency={c:3d}  ok={len(ok):3d} fail={failed:2d}  "
                  f"agg={agg:8.1f} tok/s  p50={p50:.1f}s  p95={p95:.1f}s")

    print(f"\nAppended to {OUT}")
    print("Throughput stops rising once the KV cache saturates — that knee is "
          "the measured max concurrent requests for this context size.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--concurrencies", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 24, 32, 48])
    ap.add_argument("--gen-tokens", type=int, default=256)
    args = ap.parse_args()
    sweep(args.scenario, args.concurrencies, args.gen_tokens)


if __name__ == "__main__":
    main()
