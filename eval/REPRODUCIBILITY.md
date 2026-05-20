# Eval Reproducibility

Version pins and setup instructions for every arm in the agentic eval.

---

## Environment

```
Python 3.11.x
CUDA / ROCm: AMD MI300X (or equivalent — 7B runs on any 24 GB GPU)
OS: Linux (Ubuntu 22.04 recommended) or Windows 11
```

## Inference server (local model)

```bash
# Qwen2.5-Coder-32B (main eval model)
pip install vllm
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --port 8000 \
  --max-model-len 32768 \
  --dtype bfloat16

# Qwen2.5-Coder-7B (smoke / scale ablation)
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 \
  --max-model-len 32768
```

## Python dependencies

```bash
# Core SG + eval harness
pip install -e ".[eval,embeddings]"

# Strong baselines (aider + hybrid-RAG)
pip install -e ".[eval-strong]"
```

## Arm version pins

| Arm | Package | Version | Notes |
|-----|---------|---------|-------|
| `sg` | skeletongraph | this repo | Commit hash in paper |
| `bm25` | — | — | Hand-rolled Okapi BM25 in `eval/backends/bm25_flat.py` |
| `grep` | — | — | `re`-based token grep in `eval/backends/grep_sim.py` |
| `none` | — | — | No retrieval; long-context only |
| `hybrid` | sentence-transformers | ≥2.2 | all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `aider` | aider-chat | ≥0.82 | RepoMap class; pin exact version before paper submission |
| `sg-nograph` | skeletongraph | this repo | `enable_graph_expansion=False` |
| `sg-norerank` | skeletongraph | this repo | `enable_centrality_rerank=False` |
| `sg-nosummary` | skeletongraph | this repo | `enable_summaries=False` |

**Before paper submission**: record the exact `aider-chat` version in use:
```bash
python -c "import aider; print(aider.__version__)"
```

## Dataset

```bash
# Build the SWE-bench task dataset (stage0.jsonl)
python -m eval.make_dataset --stage stage0 --n 150

# Clone task repos (one-time; cached under eval/datasets/_repo_cache/)
python -m eval.agent.run_stage --stage 1-core --dry-run
```

## Running a stage

```bash
# Full stage 1 (150 tasks × 4 arms)
SG_EVAL_API_BASE=http://localhost:8000/v1 \
  python -m eval.agent.run_stage --stage 1-core

# Quick smoke (5 tasks × all arms, local 7B)
SG_EVAL_API_BASE=http://localhost:8000/v1 \
SG_EVAL_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
  python -m eval.agent.run_stage --stage 1-core --n 5

# Re-run single task for debugging
python -m eval.agent.run_agent \
  --task-id astropy__astropy-8707 \
  --arm sg \
  --keep-workspace
```

## Aggregate and verify

```bash
# Compute all metrics (writes eval/results/agent/SUMMARY.md)
python -m eval.agent.aggregate

# Run SWE-bench verify (writes pass/fail verdicts into each run JSON)
python -m eval.agent.verify

# Re-aggregate with verdicts
python -m eval.agent.aggregate
```

## Notes on determinism

- `TEMPERATURE=0.0` in `eval/agent/config.py` — never change.
- `SEED=42` for bootstrap CIs in `aggregate.py` — never change.
- vLLM uses `--seed 42` by default; pin it explicitly if your vLLM version differs.
- Per-task repo isolation: each run gets a fresh `git worktree` under
  `eval/datasets/_agent_work/`. The worktree is cleaned after each run.
