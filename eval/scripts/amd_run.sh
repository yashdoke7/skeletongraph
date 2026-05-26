#!/bin/bash
set -euo pipefail
# ==================================================================================
# SkeletonGraph AMD Evaluation Script (v1)
# GPU: AMD MI300X 192 GB VRAM  |  Rate: ~$1.99/hr
# Model: Qwen/Qwen2.5-Coder-32B-Instruct (BF16, ~64 GB VRAM, fits single GPU)
# ==================================================================================
#
# STAGED RUN PLAN (build one paper tier at a time; stop after any stage):
#
#   Stage 0-assess  (smoke, 3 arms × 15 tasks):     ~0.3 GPU-hrs  (free sanity check)
#   Stage 1a        (workshop,  5 arms × 150 tasks): ~7-10 GPU-hrs  ★★★ required
#   Stage 1b        (conference, 5 arms × 150 tasks): ~7-10 GPU-hrs  ★★ parallel with 1a
#   Single-shot     (sg-noagent × 150 tasks):          ~0.5 GPU-hrs  part of conference
#   Stage 2         (competitor cbmem × 150 tasks):    ~3-5 GPU-hrs  ★
#   ContextBench    (1a arms on 2nd benchmark):         ~5-7 GPU-hrs  ★
#   Stage 3-further (variance + sg-learned, 60 tasks): ~5-8 GPU-hrs  top-tier only
#
#   WORKSHOP bundle (1a):           ~8-10 GPU-hrs  ≈ $16-20   at $1.99/hr
#   CONFERENCE bundle (1a + 1b):    ~15-20 GPU-hrs ≈ $30-40   (1a+1b run in PARALLEL)
#   FULL plan (all stages):         ~25-35 GPU-hrs ≈ $50-70
#
# WORKERS: 192 GB VRAM — all for vLLM. SG workers are CPU/RAM, not GPU.
#   --workers 16 to start; bump to --workers 24 if GPU < 80% utilization.
# ==================================================================================
#
# QUICK OPERATION GUIDE (READ ONCE)
# ─────────────────────────────────
# Connect from Windows PowerShell:
#   ssh root@<AMD_IP_OR_RUNPOD_HOST> -p <PORT>  (or use RunPod SSH button)
#
# Start tmux on AMD (do this ONCE per server session):
#   tmux new-session -s sg
#
# Detach (laptop can be closed after this — jobs keep running on AMD):
#   Ctrl+B, then D
#
# Reconnect later:
#   ssh root@<AMD_HOST> -p <PORT>
#   tmux attach -t sg && ensure_vllm
#
# Open a second window (for running 1a and 1b in PARALLEL):
#   Ctrl+B c
# Switch between windows:
#   Ctrl+B 0  (window 0 — vLLM server or 1a)
#   Ctrl+B 1  (window 1 — 1b or monitoring)
# Kill session when all done:
#   tmux kill-session -t sg
# ==================================================================================

REPO_DIR="${WORKSPACE_DIR:-/workspace}/skeletongraph"
MODELS_DIR="${WORKSPACE_DIR:-/workspace}/models"
GITHUB_USER="yashdoke7"
GITHUB_REPO="skeletongraph"
# Replace YOUR_PAT with a GitHub personal access token (Settings → Developer → Tokens)
# Minimum scope: repo (read + write). Create at: https://github.com/settings/tokens
GITHUB_PAT="${GITHUB_PAT:-YOUR_PAT_HERE}"
GITHUB_URL="https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"

PY="${PYTHON:-python3}"

# ── helper functions — paste ALL at once (persistent in tmux session) ─────────

ensure_vllm() {
  curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && {
    echo "vLLM OK: $(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); [print("  model:", m["id"]) for m in d.get("data",[])]' 2>/dev/null)"
    return 0
  }
  echo "vLLM not running — start it in tmux window 0 (see BLOCK 1)."
  return 1
}

push_results() {
  local msg="${1:-checkpoint}"
  echo "=== Pushing results: $msg ==="
  cd "$REPO_DIR"
  git add -f eval/results/ 2>/dev/null || true
  git diff --cached --quiet && { echo "Nothing new to commit."; return 0; }
  git commit -m "results: $msg [$(date '+%Y-%m-%d %H:%M')]"
  # Pull first in case Windows pushed code changes while we were running:
  git pull --rebase origin main 2>/dev/null || echo "(pull warning — continuing)"
  git push origin main
  echo "=== Pushed: $msg ==="
}

show_progress() {
  local run_dir="${1:-$REPO_DIR/eval/results/agent}"
  echo "=== Progress: $run_dir ==="
  python3 - "$run_dir" << 'PY'
import sys, json, os, collections
d = sys.argv[1]
if not os.path.isdir(d):
    print("  (no results yet)")
    sys.exit(0)
by_arm = collections.defaultdict(lambda: {"total": 0, "done": 0, "errors": 0})
for f in os.listdir(d):
    if not f.endswith(".json") or f.startswith("_"): continue
    try: r = json.loads(open(os.path.join(d, f)).read())
    except: continue
    arm = r.get("arm", "?")
    by_arm[arm]["total"] += 1
    if r.get("stopped") in ("submit", "max_turns"): by_arm[arm]["done"] += 1
    if r.get("stopped") == "error": by_arm[arm]["errors"] += 1
for arm, s in sorted(by_arm.items()):
    print(f"  {arm:25} {s['done']:3d}/{s['total']:3d} done  {s['errors']} errors")
PY
}

set_run_tag() {
  export SG_EVAL_RUN_TAG="$1"
  export SG_EVAL_API_BASE="http://127.0.0.1:8000/v1"
  export SG_EVAL_API_KEY="EMPTY"
  export SG_EVAL_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
  echo "Run tag: $SG_EVAL_RUN_TAG  |  model: $SG_EVAL_MODEL"
}

estimate_remaining() {
  local run_dir="${1:-$REPO_DIR/eval/results/agent/$SG_EVAL_RUN_TAG}"
  local total_expected="${2:-750}"   # 5 arms × 150 tasks default
  python3 - "$run_dir" "$total_expected" << 'PY'
import sys, json, os, time
d, total = sys.argv[1], int(sys.argv[2])
done = sum(1 for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")) if os.path.isdir(d) else 0
pct = done / total * 100 if total else 0
print(f"  {done}/{total} runs complete ({pct:.1f}%)")
PY
}


# ==================================================================================
# BLOCK 0 — ONE-TIME SETUP (~15-30 min total)
# Run each step manually; verify before proceeding to the next.
# ==================================================================================

# 0.1  Verify storage (run this first — DO NOT proceed if storage is < 200 GB)
df -h | head -5

# 0.2  System packages
apt-get update -qq && apt-get install -y git tmux htop curl python3-pip python3-venv

# 0.3  Clone the repo (first time only)
mkdir -p "$(dirname "$REPO_DIR")"
git clone "$GITHUB_URL" "$REPO_DIR"
cd "$REPO_DIR"

# OR if repo already exists: pull latest code
# cd "$REPO_DIR" && git pull origin main

# 0.4  Git identity (needed for commits from AMD)
git config user.email "yashdoke215@gmail.com"
git config user.name "Yash Doke"

# 0.5  Python dependencies
pip3 install -r requirements.txt -q
pip3 install huggingface-hub sentence-transformers scikit-learn -q

# 0.6  Install vLLM with ROCm (AMD GPU) support
# Option A — standard pip (vLLM v0.8+ has ROCm built in):
pip3 install vllm -q

# Option B — if Option A fails (ROCm version mismatch), use AMD's prebuilt wheel:
# Visit https://docs.vllm.ai/en/latest/getting_started/amd-installation.html
# pip3 install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2

# 0.7  Download model to network volume (once — ~65 GB; takes 15-20 min)
mkdir -p "$MODELS_DIR"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-Coder-32B-Instruct',
    local_dir='$MODELS_DIR/Qwen2.5-Coder-32B-Instruct',
    local_dir_use_symlinks=False
)
print('Model downloaded.')
"
# Verify:
ls "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct/"

# 0.8  Build stage0.jsonl with 150 stratified tasks (AMD run uses 150, not 30)
# NOTE: This clones ~150 task repos — takes ~30-60 min depending on network.
cd "$REPO_DIR"
pip3 install datasets -q
$PY eval/make_dataset.py --n 150 --seed 42
# Verify:
$PY -c "import json; ts=list(map(json.loads,open('eval/datasets/stage0.jsonl').readlines())); print(len(ts),'tasks')"

# 0.9  Smoke test — minimal retrieval check on 2 tasks (no LLM needed)
$PY eval/make_dataset.py --smoke --out /tmp/smoke_test.jsonl
$PY eval/retrieval_eval.py --dataset /tmp/smoke_test.jsonl
echo "Setup complete. Proceed to BLOCK 1."


# ==================================================================================
# BLOCK 1 — START vLLM SERVER (dedicated tmux window — keeps running throughout)
# Open a NEW tmux window: Ctrl+B c
# Run these lines in that window; keep it open.
# ==================================================================================

cd "$REPO_DIR"
# Verify AMD GPU is visible:
rocm-smi 2>/dev/null || python3 -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

# Start vLLM (leave this window running — do NOT close or Ctrl+C mid-eval):
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct" \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --port 8000 \
    > ~/vllm.log 2>&1 &
echo "vLLM starting (log: ~/vllm.log) ... waiting 60s"
sleep 60 && ensure_vllm

# Switch back to your main eval window: Ctrl+B 0


# ==================================================================================
# BLOCK 2 — STAGE 0-ASSESS (smoke test — run FIRST to verify pipeline on AMD)
# 3 arms × 15 tasks = 45 runs; ~20-30 min; confirms vLLM + SG integration works.
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"   # all AMD swebench results go here
echo "=== 0-ASSESS START $(date) ==="
$PY -m eval.agent.run_stage --stage 0-assess --workers 4
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench
echo "=== 0-ASSESS DONE $(date) ==="
push_results "0-assess complete"

# ── STOP HERE. Check the SUMMARY.md / summary.json. ─────────────────────────
# If retrieval-hit rates look sensible (SG ~0.6, BM25 ~0.7, none ~0.0),
# proceed to BLOCK 3. If vLLM errors / OOM / wrong results, investigate first.
# ─────────────────────────────────────────────────────────────────────────────


# ==================================================================================
# BLOCK 3a — STAGE 1a-WORKSHOP (tmux window 0 — run while 3b runs in window 1)
# 5 arms × 150 tasks = 750 runs; ~7-10 GPU-hrs (wall-clock, sharing GPU with 3b).
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"
echo "=== 1a-WORKSHOP START $(date) ==="
$PY -m eval.agent.run_stage --stage 1a-workshop --workers 16
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench
echo "=== 1a-WORKSHOP DONE $(date) ==="
push_results "stage 1a-workshop complete ($(show_progress eval/results/agent/qwen32b_swebench 2>/dev/null | wc -l) arms)"


# ==================================================================================
# BLOCK 3b — STAGE 1b-CONFERENCE (tmux window 1 — START SAME TIME AS 3a)
# 5 arms × 150 tasks = 750 runs; ~7-10 GPU-hrs (parallel with 3a).
# Switch to window 1: Ctrl+B c (or Ctrl+B 1 if window 1 already exists)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"   # same tag — run_ids differ (different arms)
echo "=== 1b-CONFERENCE START $(date) ==="
$PY -m eval.agent.run_stage --stage 1b-conference --workers 16
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench
echo "=== 1b-CONFERENCE DONE $(date) ==="
push_results "stage 1b-conference complete"


# ==================================================================================
# BLOCK 3c — SINGLE-SHOT sg-noagent (run after 3b, or in a 3rd window)
# 1 LLM call per task × 150 tasks = 150 runs; ~20-30 min; cheap.
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"
echo "=== sg-noagent single-shot START $(date) ==="
$PY -m eval.agent.run_singleshot --all
echo "=== sg-noagent single-shot DONE $(date) ==="
push_results "sg-noagent single-shot complete"

# After 3a + 3b + 3c: run full aggregate + push
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench
push_results "stage 1 all arms + single-shot complete — WORKSHOP/CONFERENCE DONE"


# ==================================================================================
# BLOCK 4 — PASS@1 VERIFICATION (needs SWE-bench Docker harness)
# This is the REAL accuracy metric. Local eval only gives retrieval + efficiency.
# Requires: Docker installed + SWE-bench eval Docker images.
# verify.py runs the harness ONCE PER ARM (one predictions file per arm) so each
# task's verdict is attributed to the correct arm, then writes `resolved` back.
# ==================================================================================

pip3 install swebench -q

# Per-arm pass@1 for each stage's arms (writes `resolved` into each run JSON):
$PY -m eval.agent.verify --stage 1a-workshop   --run-tag qwen32b
$PY -m eval.agent.verify --stage 1b-conference --run-tag qwen32b
# aggregate now fills the pass@1 column + McNemar (sg vs each baseline):
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench
push_results "post-verify pass@1 + McNemar written"


# ==================================================================================
# BLOCK 5 — STAGE 2-COMPETITOR (cbmem / CodeCompass)
# Requires either cbmem binary (CBMEM_BIN) or CodeCompass running.
# Skip if competitor tooling is not set up; come back after stage 3.
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"

# Option A — cbmem (set CBMEM_BIN to the binary path):
# export CBMEM_BIN=/workspace/codebase-memory-mcp
# $PY -m eval.agent.run_stage --stage 0-cbmem --workers 4
# push_results "cbmem baseline complete"

# Option B — CodeCompass (preferred, open-source Neo4j MCP):
# npm install -g @codecompass/cli && codecompass --help
# (wire into cbmem arm backend or add codecompass backend in tools.py)
echo "Stage 2 competitor — set up cbmem or CodeCompass first, then uncomment above."


# ==================================================================================
# BLOCK 6 — CONTEXTBENCH (second benchmark — separate results tag)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm

# Extract ContextBench dataset (run once):
$PY -m eval.scripts.extract_contextbench \
    --out eval/datasets/contextbench.jsonl \
    --data-files data/SWEContextBench_Experience.parquet
$PY -c "import json; ts=list(map(json.loads,open('eval/datasets/contextbench.jsonl').readlines())); print(len(ts),'ContextBench tasks')"

# Run 1a arms on ContextBench (separate tag so results don't mix with SWE-bench):
set_run_tag "qwen32b_contextbench"
echo "=== ContextBench 1a START $(date) ==="
$PY -m eval.agent.run_stage --stage 1a-workshop \
    --dataset eval/datasets/contextbench.jsonl \
    --workers 16
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_contextbench
echo "=== ContextBench 1a DONE $(date) ==="
push_results "contextbench 1a-workshop complete"

# Offline retrieval eval on ContextBench (no LLM, instant):
$PY eval/retrieval_eval.py --dataset eval/datasets/contextbench.jsonl


# ==================================================================================
# BLOCK 7 — STAGE 3-FURTHER (top-tier only — run if 1+2 show strong results)
# 4 arms × 60 tasks × 3 repeats = 720 runs; ~8-12 GPU-hrs.
# Includes sg-learned (needs curator_model.pkl — build it first locally).
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm

# Only run if curator_model.pkl exists (else sg-learned == sg):
if [ -f eval/curator/curator_model.pkl ]; then
  echo "Curator model found — sg-learned is active."
else
  echo "WARNING: no curator_model.pkl — sg-learned will fall back to rule router (== sg)."
fi

set_run_tag "qwen32b_swebench_stage3"
echo "=== 3-FURTHER START $(date) ==="
$PY -m eval.agent.run_stage --stage 3-further --workers 16
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench_stage3
echo "=== 3-FURTHER DONE $(date) ==="
push_results "stage 3-further complete"


# ==================================================================================
# BLOCK 8 — FINAL ANALYSIS + DOWNLOAD
# ==================================================================================

cd "$REPO_DIR"

# Full aggregate across all SWE-bench runs:
$PY -m eval.agent.aggregate --run-dir eval/results/agent/qwen32b_swebench

# Figures:
$PY -m eval.agent.plots

# Push everything:
git add -f eval/results/ eval/figures/ 2>/dev/null || true
git commit -m "results: all stages complete — final push [$(date '+%Y-%m-%d')]" 2>/dev/null || true
git push origin main

# Zip for direct download (backup):
zip -r ~/sg_results_$(date +%Y%m%d).zip eval/results/ eval/figures/
echo "Download from: ~/sg_results_$(date +%Y%m%d).zip"


# ==================================================================================
# MONITORING (run in a separate tmux window while eval is running)
# ==================================================================================

# GPU utilization + VRAM:
watch -n 10 'rocm-smi 2>/dev/null || nvidia-smi'

# Progress per arm (updates every 30s):
watch -n 30 'python3 -c "
import json, os, collections
d = \"eval/results/agent/$SG_EVAL_RUN_TAG\" if \"$SG_EVAL_RUN_TAG\" else \"eval/results/agent\"
by_arm = collections.defaultdict(lambda: {\"n\": 0, \"done\": 0, \"errors\": 0})
if os.path.isdir(d):
    for f in os.listdir(d):
        if not f.endswith(\".json\") or f.startswith(\"_\"): continue
        try:
            r = json.loads(open(os.path.join(d, f)).read())
            arm = r.get(\"arm\", \"?\")
            by_arm[arm][\"n\"] += 1
            if r.get(\"stopped\") in (\"submit\", \"max_turns\"): by_arm[arm][\"done\"] += 1
            if r.get(\"stopped\") == \"error\": by_arm[arm][\"errors\"] += 1
        except: pass
for arm, s in sorted(by_arm.items()):
    print(f\"  {arm:25} {s[chr(100)]:3d}/{s[chr(110)]:3d}  errors={s[chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(115)]}\")
else:
    print(\"  no results yet\")
"'

# vLLM request rate (from its log):
tail -f ~/vllm.log | grep -E "Avg prompt|Avg generation|Running|Pending"
