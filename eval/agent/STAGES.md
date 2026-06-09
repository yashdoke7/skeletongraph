# Staged Evaluation Plan

Budget: AMD Developer Cloud credit — **$100 ≈ 50 hours** on one MI300X. The
T&C wording ("roughly equivalent to 50 hours") confirms billing is **time-based**,
not compute-based: every wall-clock hour the box is up costs the same.

## Compute Math (AMD MI300X)

*   **VRAM / Concurrency:** An MI300X has 192GB of VRAM. A 32B model in fp16 takes ~64GB for weights. This leaves ~128GB for KV Cache. With vLLM, you can easily sustain **16–20 parallel workers** continuously across multiple `tmux` windows.
*   **Agent Task Duration:** At 16–20 concurrency, vLLM throughput is incredibly high. Tasks finish in 60-110 seconds. You can process **~700 tasks per GPU-hour**. 
*   **Indexing Time:** `graphify` and `summary-llm` take roughly 1-5 minutes to index a massive repository. Both Verified and Pro have relatively few unique repositories (~40 total). The total one-time LLM indexing cost across both benchmarks is **~2 hours**.
*   **Docker Execution:** SWE-bench docker verification is CPU-bound and runs outside the GPU budget in a background window.

## Evaluation Strategy

We have divided the execution into four targeted subsets. Because MI300X throughput is so high, both the Workshop and Conference targets fit entirely within ~15 GPU-hours, leaving massive headroom for variance and scaling runs!

| Stage Name | Target | Tasks | Arms | Est. Compute |
|---|---|---|---|---|
| **amd-workshop-300** | Workshop (Lite Equiv) | 300 | 5 | ~2.5 hours |
| **amd-conference-500** | Conference (Full Stats) | 500 | 6 | ~4.5 hours |
| **amd-pro-100** | Conference (Scaling) | 100 | 2 | ~0.5 hours |
| **amd-nim-150** | API Rate-Limit Safe | 150 | 5 | ~8.0 hours* |

*\*NIM throughput is limited by API rate limits (e.g. 40 RPM), not raw GPU compute.*

---

### 1. Workshop Target (`amd-workshop-300`)
For a workshop, you need solid baseline comparisons on a recognized benchmark subset. SWE-bench "Lite" is exactly 300 tasks, making it the perfect target.

*   **Benchmark:** SWE-bench Verified (300 subset)
*   **Arms (5):** `sg`, `bm25`, `cbmem`, `graphify`, `summary-llm-bm25`
*   **Compute:** 1,500 task-evals @ 700/hr = **~2.5 hours**.

### 2. Conference Target (`amd-conference-500` + `amd-pro-100`)
For Top-Tier (ICLR / ICSE), reviewers expect the full benchmark (500 tasks) to prove statistical significance, plus an ablation or scaling experiment (SWE-bench Pro) to prove it handles massive codebases.

*   **Benchmark 1:** SWE-bench Verified (Full 500)
*   **Arms (6):** `sg`, `bm25`, `cbmem`, `graphify`, `summary-llm-bm25`, `hybrid`
*   **Compute:** 3,000 task-evals @ 700/hr = **~4.5 hours**.

*   **Benchmark 2:** SWE-bench Pro (100 tasks)
*   **Arms (2):** `sg`, `bm25`
*   **Compute:** 200 task-evals (Pro tasks take longer) = **~0.5 hours**.

*Conclusion:* The entire evaluation suite runs in less than **10 GPU-hours**, meaning you have **40 hours of budget left** to run variance repeats (3-seeds) or test a 7B model for a scaling curve!

## Parallelism & Execution (tmux batches)

Because vLLM handles continuous batching perfectly, you should run different arms in completely parallel `tmux` windows so that CPU-bound arms (like `cbmem` indexing or `graphify` fetching) don't bottleneck the GPU.

```bash
# Window 0: Start vLLM Server
bash eval/agent/serve_model.sh

# Window 1: Headline Arms
python -m eval.agent.run_stage --stage amd-workshop-300 --only-arms sg,bm25,summary-llm-bm25 --workers 10

# Window 2: CBMEM
python -m eval.agent.run_stage --stage amd-workshop-300 --only-arms cbmem --workers 4

# Window 3: Graphify
python -m eval.agent.run_stage --stage amd-workshop-300 --only-arms graphify --workers 4

# Window 4: Rolling Docker Verify (Runs in background)
python -m eval.agent.verify --stage amd-workshop-300
```
