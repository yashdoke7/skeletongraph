# WSL — recover the NIM-70B v2 pass@1 numbers (hybrid bug + cbmem skipped)

The harness fix in this commit (workspace `.gitignore` + hybrid cache moved
outside the repo + `_SG_ARTIFACTS` extension) prevents the bug going
forward. The two things below recover the NIM-70B-v2 numbers that already
exist, without re-running any agent task.

## 1. Clean the polluted hybrid predictions (retroactive fix)

```bash
source ~/sg-env/bin/activate
cd /mnt/c/Users/ASUS/Desktop/CS/Projects/skeletongraph

python -m eval.scripts.clean_hybrid_predictions \
    --results-dir eval/results/agent/nim70b_swebench_v2 \
    --arm hybrid
```

Expected output: ~29/30 hybrid run JSONs cleaned, ~29/30 rows cleaned in
`_predictions_hybrid.jsonl`. Idempotent — running it twice is a no-op.

## 2. Run SWE-bench Verified pass@1 on the cleaned predictions + cbmem

```bash
# Docker must be running (WSL2 + Docker Desktop integration)
docker --version
docker ps

# Pin the same tag as the original run so RUNS_DIR resolves correctly
export SG_EVAL_RUN_TAG=nim70b_swebench_v2

# Re-verify hybrid (now that the binary cache hunks are stripped) + cbmem
# (which was never verified). Idempotent — submit/max_turns runs are kept.
python -m eval.agent.verify --stage baseline    --run-tag nim70b_v2
python -m eval.agent.verify --stage 0-cbmem     --run-tag nim70b_v2

# Aggregate — SUMMARY.md will now have pass@1 for ALL six arms
python -m eval.agent.aggregate
cat eval/results/agent/nim70b_swebench_v2/SUMMARY.md
```

## 3. Push the corrected results

```bash
cd /mnt/c/Users/ASUS/Desktop/CS/Projects/skeletongraph
git add eval/results/agent/nim70b_swebench_v2/
git status
git commit -m "results(nim70b_v2): re-verify hybrid (post-isolation-fix) + cbmem pass@1"
git push origin main
```

## 4. Finish the in-flight ContextBench (the run paused at 72/150 of baseline)

```bash
# Same env you've been using for the in-flight run
export SG_EVAL_RUN_TAG=nim70b_contextbench
export SG_EVAL_API_BASE="https://integrate.api.nvidia.com/v1"
export SG_EVAL_MODEL="meta/llama-3.1-70b-instruct"
export SG_EVAL_API_KEYS="nvapi-KEY1,nvapi-KEY2,nvapi-KEY3,nvapi-KEY4"

# Resume — `run_stage` skips completed jobs. Use the contextbench stage that
# now includes cbmem + graphify; if you only want the original baseline,
# pass --only-arms.
python -m eval.agent.run_stage \
    --stage contextbench \
    --dataset eval/datasets/contextbench.jsonl \
    --only-arms sg,bm25,grep,none,hybrid \
    --workers 8

# Add cbmem + graphify on the same 60 tasks (CPU-bound, lower workers)
python -m eval.backends.cbmem    --selftest eval/datasets/repos/django__django-14725
python -m eval.backends.graphify --selftest eval/datasets/repos/django__django-14725
python -m eval.agent.run_stage \
    --stage contextbench \
    --dataset eval/datasets/contextbench.jsonl \
    --only-arms cbmem,graphify \
    --workers 4

# Offline retrieval eval (no LLM cost, instant)
python eval/retrieval_eval.py --dataset eval/datasets/contextbench.jsonl

# Aggregate
python -m eval.agent.aggregate
cat eval/results/agent/nim70b_contextbench/SUMMARY.md
```

ContextBench pass@1 verification is harness-dependent — the SWE-bench Docker
harness only scores tasks on its own HF dataset. If your contextbench.jsonl
tasks aren't a subset of `princeton-nlp/SWE-bench_Verified`, the SUMMARY.md
will be retrieval/edited-gold-only for that benchmark (which is still the
correct comparable, just no pass@1 column). That's expected; the offline
retrieval-eval line above produces the recall@k figures.
