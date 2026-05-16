#!/usr/bin/env bash
# Stage 0 — one-shot GO/NO-GO eval. Pure CPU, no API, no GPU.
#
#   bash eval/stage0.sh smoke      # 2 tasks, ~10 min — verify the pipeline works
#   bash eval/stage0.sh            # 30 tasks, ~2-3 h  — the real run
#
# Safe to background:  nohup bash eval/stage0.sh > eval/logs/stage0.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-full}"
mkdir -p eval/logs

echo "=== Stage 0 :: $MODE ==="
echo "[1/4] dependencies"
python -m pip install -q datasets 2>&1 | tail -1 || true

echo "[2/4] build dataset"
if [ "$MODE" = "smoke" ]; then
  python eval/make_dataset.py --smoke
else
  python eval/make_dataset.py --n "${SG_STAGE0_N:-30}"
fi

echo "[3/4] retrieval + context-token eval"
python eval/run_stage0.py

echo "[4/4] analytical KV-cache / serving density"
python eval/kv_cache.py --csv eval/results/stage0/kv_cache.csv

echo
echo "=== Stage 0 done ==="
echo "Verdict: eval/results/stage0/SUMMARY.md"
cat eval/results/stage0/SUMMARY.md
