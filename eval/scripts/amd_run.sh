#!/bin/bash
set -euo pipefail
# ==================================================================================
# SkeletonGraph — AMD Developer Cloud MI300X Runbook (v3 — SMOKE-FIRST)
# GPU: AMD MI300X 192 GB VRAM   |  Credits: AMD Developer Cloud $100 grant
# Model: Qwen/Qwen2.5-Coder-32B-Instruct (BF16, ~64 GB VRAM, fits single GPU)
#
# This file is COPY-PASTE ONLY — you (the user) read it block by block and paste
# into a tmux session on AMD. It is NOT meant to run end-to-end as a script.
# ==================================================================================
#
# STAGED RUN PLAN (smoke first; only spend the budget if smoke passes):
#
#   BLOCK 0   Setup (apt, pip, clone, dataset prep, dependencies)       ~$1-2
#   BLOCK 1   Start vLLM server (background; runs throughout)            free
#   BLOCK 2   SMOKE 7 arms × 10 tasks (THE GATE — pass before scaling)  ~$1
#   BLOCK 3   verify SMOKE pass@1 + inspect SUMMARY.md before continuing
#   BLOCK 4a  1a-workshop 5 arms × 150 tasks (in tmux window 0)         ~$10-14
#   BLOCK 4b  1b-conference 5 ablations × 150 tasks (window 1, parallel) ~$10-14
#   BLOCK 4c  sg-noagent single-shot × 150                              ~$0.80
#   BLOCK 5   cbmem competitor × 150                                    ~$3
#   BLOCK 6   ContextBench (7 arms × 60 tasks)                          ~$4-6
#   BLOCK 7   3-seed variance appendix (4 arms × 20 tasks × 3)          ~$3
#   BLOCK 8   graphify × 150 (LAST — only if smoke + 1a both clean)     ~$3
#   BLOCK 9   verify ALL via Docker harness                             ~$4-8
#   BLOCK 10  final aggregate + figures + push
#
#   EXPECTED TOTAL: $40-55 of the $100 grant. Headroom: $45-60.
#
# WORKERS — 192 GB VRAM is overkill for 32B BF16; spend it on concurrency:
#   --workers 16 to start; bump to 24 if `rocm-smi` shows GPU < 80% busy.
#   cbmem/graphify are CPU/disk-bound — use --workers 4 for those.
#
# ROCm vLLM caveat:
#   If `pip install vllm` errors on AMD wheels, follow Option B in 0.6 below
#   (AMD's prebuilt ROCm wheel). Test on a small instance FIRST if unsure —
#   ~$0.50 of MI300X time to verify vLLM boots is much cheaper than $20 wasted.
# ==================================================================================
#
# QUICK OPERATION GUIDE (READ ONCE)
# ─────────────────────────────────
# Connect from Windows PowerShell:
#   ssh root@<AMD_DC_HOST> -p <PORT>
#
# Start tmux on AMD (do this ONCE per server session):
#   tmux new-session -s sg
#
# Detach (laptop can be closed; jobs keep running on AMD):
#   Ctrl+B, then D
#
# Reconnect later:
#   ssh root@<AMD_DC_HOST> -p <PORT>
#   tmux attach -t sg
#
# Paste the helper-function block (BLOCK A) into your session ONCE on attach.
# Then run `ensure_vllm` to confirm the server is up before resuming any block.
#
# Open a second window (for running 1a and 1b in PARALLEL):
#   Ctrl+B c          — new window
#   Ctrl+B 0 / 1      — switch window 0 / window 1
#   Ctrl+B [          — scroll mode (q to exit)
#
# Kill session when all done:
#   tmux kill-session -t sg
# ==================================================================================

REPO_DIR="${WORKSPACE_DIR:-/workspace}/skeletongraph"
MODELS_DIR="${WORKSPACE_DIR:-/workspace}/models"
GITHUB_USER="yashdoke7"
GITHUB_REPO="skeletongraph"
GITHUB_PAT="${GITHUB_PAT:-YOUR_PAT_HERE}"
GITHUB_URL="https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"

PY="${PYTHON:-python3}"


# ==================================================================================
# BLOCK A — HELPER FUNCTIONS (paste once per tmux attach)
# ==================================================================================

ensure_vllm() {
  curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && {
    echo "vLLM OK: $(curl -s http://127.0.0.1:8000/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); [print("  model:", m["id"]) for m in d.get("data",[])]' 2>/dev/null)"
    return 0
  }
  echo "vLLM not running — start it in tmux window 0 (BLOCK 1)."
  return 1
}

push_results() {
  local msg="${1:-checkpoint}"
  echo "=== Pushing results: $msg ==="
  cd "$REPO_DIR"
  git add -f eval/results/ 2>/dev/null || true
  git diff --cached --quiet && { echo "Nothing new to commit."; return 0; }
  git commit -m "results: $msg [$(date '+%Y-%m-%d %H:%M')]"
  git pull --rebase origin main 2>/dev/null || echo "(pull warning — continuing)"
  git push origin main
  echo "=== Pushed: $msg ==="
}

show_progress() {
  local run_dir="${1:-$REPO_DIR/eval/results/agent/$SG_EVAL_RUN_TAG}"
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


# ==================================================================================
# BLOCK 0 — ONE-TIME SETUP (~15-30 min total)
# Paste each step manually; verify before proceeding.
# ==================================================================================

# 0.1  Verify storage (must be ≥ 200 GB free)
df -h | head -5

# 0.2  System packages
apt-get update -qq && apt-get install -y git tmux htop curl python3-pip python3-venv

# 0.3  Clone the repo (first time only)
mkdir -p "$(dirname "$REPO_DIR")"
git clone "$GITHUB_URL" "$REPO_DIR"
cd "$REPO_DIR"
# OR if repo already exists:
# cd "$REPO_DIR" && git pull origin main

# 0.4  Git identity
git config user.email "yashdoke215@gmail.com"
git config user.name "Yash Doke"

# 0.5  Python dependencies
pip3 install -r requirements.txt -q
pip3 install huggingface-hub sentence-transformers scikit-learn datasets -q

# 0.6  Install vLLM with ROCm (AMD GPU) support
# Option A — standard pip (vLLM v0.8+ ships ROCm wheels):
pip3 install vllm -q

# Option B — if Option A fails (ROCm version mismatch), use AMD's prebuilt:
# pip3 install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2

# 0.7  Download the model to local storage (one-time, ~65 GB, 15-20 min)
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
ls "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct/"

# 0.8  Build the 150-task SWE-bench dataset (clones ~150 repos; ~30-60 min)
cd "$REPO_DIR"
$PY eval/make_dataset.py --n 150 --seed 42
$PY -c "import json; ts=list(map(json.loads,open('eval/datasets/stage0.jsonl').readlines())); print(len(ts),'tasks')"

# 0.9  Build the 60-task ContextBench dataset (HF pull; ~10-30 min)
$PY -m eval.scripts.extract_contextbench --inspect
$PY -m eval.scripts.extract_contextbench --n 60 --lang python \
    --out eval/datasets/contextbench.jsonl
$PY -c "import json; ts=list(map(json.loads,open('eval/datasets/contextbench.jsonl').readlines())); print(len(ts),'ContextBench tasks')"

# 0.10 Install cbmem binary (Linux, AMD DC)
#   Option A — pre-built Linux x86_64 binary:
curl -L https://github.com/seawinde/codebase-memory-mcp/releases/latest/download/cbmem-linux-amd64 \
    -o /usr/local/bin/codebase-memory-mcp
chmod +x /usr/local/bin/codebase-memory-mcp
export CBMEM_BIN="/usr/local/bin/codebase-memory-mcp"
"$CBMEM_BIN" --version  # confirm

# 0.11 Install graphify (Python module path; binary path is fallback)
pip3 install graphifyy -q || echo "(graphifyy install failed — backend will use CLI path if GRAPHIFY_BIN is set)"

# 0.12 Selftest both external backends BEFORE running the smoke
$PY -m eval.backends.cbmem    --selftest eval/datasets/repos/django__django-14725 || echo "cbmem selftest FAILED — fix before stage 5/6"
$PY -m eval.backends.graphify --selftest eval/datasets/repos/django__django-14725 || echo "graphify selftest FAILED — fix or skip arm; safe to proceed without"

echo "Setup complete. Proceed to BLOCK 1."


# ==================================================================================
# BLOCK 1 — START vLLM SERVER (dedicated tmux window — keeps running throughout)
# Open a NEW tmux window: Ctrl+B c.  Run these in that window; leave it open.
# ==================================================================================

cd "$REPO_DIR"
rocm-smi 2>/dev/null || python3 -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct" \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8000 \
    > ~/vllm.log 2>&1 &
echo "vLLM starting (log: ~/vllm.log) ... waiting 90s"
sleep 90 && ensure_vllm

# Switch back to main eval window: Ctrl+B 0


# ==================================================================================
# BLOCK 2 — SMOKE: 7 arms × 10 tasks  (THE GATE before any big spend)
# ~1-1.5 hours wall time at --workers 16; ≈ $1 of GPU.
# If smoke arms look healthy AND verify works, proceed to BLOCK 4. If anything
# is broken (errored arms, empty patches, vLLM crashes), STOP and fix.
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"

echo "=== 0-SMOKE START $(date) ==="
# 7 arms × first 10 tasks. cbmem/graphify CPU-bound — separate command below.
$PY -m eval.agent.run_stage --stage 0-smoke --limit 10 --workers 16 \
    --skip-arms cbmem,graphify
# cbmem + graphify on the same 10 tasks (lower workers — they hit disk/CPU)
$PY -m eval.agent.run_stage --stage 0-smoke --limit 10 --workers 4 \
    --only-arms cbmem,graphify

show_progress
push_results "0-smoke complete (10 tasks × 7 arms)"


# ==================================================================================
# BLOCK 3 — SMOKE VERIFY (pass@1 via SWE-bench Docker; the GATE check)
# Requires Docker. If Docker isn't on this box, run BLOCK 3 on a separate
# CPU pod against the cloned predictions JSONLs.
# ==================================================================================

pip3 install swebench -q
$PY -m eval.agent.verify --stage 0-smoke --run-tag qwen32b_smoke
$PY -m eval.agent.aggregate --stage 0-smoke
push_results "smoke verify done"

# READ THE SUMMARY.MD. If pass@1 is plausible (e.g. sg ≥ bm25 on edited-gold,
# none < everyone else, hybrid completes without 90%+ errors), continue.
# If anything is wildly off, fix BEFORE BLOCK 4 — that's the whole point.


# ==================================================================================
# BLOCK 4a — STAGE 1a-WORKSHOP (tmux window 0)
# 5 arms × 150 tasks = 750 runs; ~3-4 GPU-hrs at --workers 16.
# Chunks of 50 with per-chunk push (results survive crashes).
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"

echo "=== 1a-WORKSHOP START $(date) ==="
$PY -m eval.agent.run_stage --stage 1a-workshop --limit 50  --workers 16
show_progress && push_results "1a-workshop 50/150 done"
$PY -m eval.agent.run_stage --stage 1a-workshop --limit 100 --workers 16
show_progress && push_results "1a-workshop 100/150 done"
$PY -m eval.agent.run_stage --stage 1a-workshop --limit 150 --workers 16
$PY -m eval.agent.aggregate
echo "=== 1a-WORKSHOP DONE $(date) ==="
push_results "1a-workshop 150/150 COMPLETE"


# ==================================================================================
# BLOCK 4b — STAGE 1b-CONFERENCE (tmux window 1 — START SAME TIME AS 4a)
# 5 ablation arms × 150 tasks = 750 runs; ~3-4 GPU-hrs (parallel with 4a).
# Switch to window 1: Ctrl+B c (or Ctrl+B 1 if already open).
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"

echo "=== 1b-CONFERENCE START $(date) ==="
$PY -m eval.agent.run_stage --stage 1b-conference --limit 50  --workers 16
push_results "1b 50/150 done"
$PY -m eval.agent.run_stage --stage 1b-conference --limit 100 --workers 16
push_results "1b 100/150 done"
$PY -m eval.agent.run_stage --stage 1b-conference --limit 150 --workers 16
$PY -m eval.agent.aggregate
echo "=== 1b-CONFERENCE DONE $(date) ==="
push_results "1b-conference 150/150 COMPLETE"


# ==================================================================================
# BLOCK 4c — SINGLE-SHOT sg-noagent (after 4a+4b finish; cheap, parallelisable)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"
echo "=== sg-noagent START $(date) ==="
$PY -m eval.agent.run_singleshot --all
echo "=== sg-noagent DONE $(date) ==="
push_results "sg-noagent single-shot done"


# ==================================================================================
# BLOCK 5 — cbmem competitor on 150 tasks (CPU/disk-bound, --workers 4)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"
export CBMEM_BIN="/usr/local/bin/codebase-memory-mcp"

echo "=== 2-competitor cbmem START $(date) ==="
$PY -m eval.agent.run_stage --stage 2-competitor --limit 50  --workers 4
push_results "cbmem 50/150 done"
$PY -m eval.agent.run_stage --stage 2-competitor --limit 100 --workers 4
push_results "cbmem 100/150 done"
$PY -m eval.agent.run_stage --stage 2-competitor --limit 150 --workers 4
$PY -m eval.agent.aggregate
echo "=== 2-competitor cbmem DONE $(date) ==="
push_results "cbmem 150/150 COMPLETE"


# ==================================================================================
# BLOCK 6 — ContextBench (7 arms × 60 tasks, separate run tag)
# Includes cbmem + graphify so the graph-competitor comparison spans both benchmarks.
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_contextbench"

echo "=== ContextBench START $(date) ==="
# Baseline arms + SG: GPU-bound, --workers 16
$PY -m eval.agent.run_stage --stage contextbench --limit 20 --workers 16 \
    --dataset eval/datasets/contextbench.jsonl --skip-arms cbmem,graphify
$PY -m eval.agent.run_stage --stage contextbench --limit 40 --workers 16 \
    --dataset eval/datasets/contextbench.jsonl --skip-arms cbmem,graphify
$PY -m eval.agent.run_stage --stage contextbench --limit 60 --workers 16 \
    --dataset eval/datasets/contextbench.jsonl --skip-arms cbmem,graphify
push_results "contextbench (5 GPU arms) 60/60 done"

# cbmem + graphify on the same 60 tasks (CPU-bound, --workers 4)
$PY -m eval.agent.run_stage --stage contextbench --limit 60 --workers 4 \
    --dataset eval/datasets/contextbench.jsonl --only-arms cbmem,graphify
push_results "contextbench (cbmem+graphify) 60/60 done"

$PY -m eval.agent.aggregate
echo "=== ContextBench DONE $(date) ==="

# Offline retrieval eval (no LLM, instant)
$PY eval/retrieval_eval.py --dataset eval/datasets/contextbench.jsonl


# ==================================================================================
# BLOCK 7 — VARIANCE APPENDIX (4 arms × 20 tasks × 3 seeds = 240 runs)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench_variance"

echo "=== variance appendix START $(date) ==="
$PY -m eval.agent.run_stage --stage variance --workers 16
$PY -m eval.agent.aggregate --stage variance
echo "=== variance DONE $(date) ==="
push_results "variance appendix COMPLETE"


# ==================================================================================
# BLOCK 8 — GRAPHIFY × 150  (LAST — only after smoke + 1a are clean)
# ==================================================================================

cd "$REPO_DIR" && ensure_vllm
set_run_tag "qwen32b_swebench"

# Selftest one more time after smoke to confirm graphify is stable
$PY -m eval.backends.graphify --selftest eval/datasets/repos/django__django-14725

echo "=== graphify START $(date) ==="
$PY -m eval.agent.run_stage --stage 2-competitor --limit 50  --workers 4 --only-arms graphify
push_results "graphify 50/150 done"
$PY -m eval.agent.run_stage --stage 2-competitor --limit 100 --workers 4 --only-arms graphify
push_results "graphify 100/150 done"
$PY -m eval.agent.run_stage --stage 2-competitor --limit 150 --workers 4 --only-arms graphify
$PY -m eval.agent.aggregate
echo "=== graphify DONE $(date) ==="
push_results "graphify 150/150 COMPLETE"


# ==================================================================================
# BLOCK 9 — PASS@1 VERIFY for everything (SWE-bench Docker harness)
# ==================================================================================

pip3 install swebench -q

# SWE-bench arms
$PY -m eval.agent.verify --stage 1a-workshop    --run-tag qwen32b_swebench
$PY -m eval.agent.verify --stage 1b-conference  --run-tag qwen32b_swebench
$PY -m eval.agent.verify --stage 2-competitor   --run-tag qwen32b_swebench  # cbmem + graphify
$PY -m eval.agent.verify --stage variance       --run-tag qwen32b_swebench_variance

# ContextBench arms — verify.py's --dataset flag is an HF dataset name, not a
# local jsonl. ContextBench tasks may or may not be on SWE-bench's HF Verified
# split; if not, score them offline against the gold patches in
# contextbench.jsonl rather than via the SWE-bench Docker harness:
#   $PY eval/retrieval_eval.py --dataset eval/datasets/contextbench.jsonl
# Once a HF SWE-bench-compatible mirror of ContextBench is identified, replace
# the line below with `--dataset <hf-name>`.
# $PY -m eval.agent.verify --stage contextbench --run-tag qwen32b_contextbench

# Final aggregate with pass@1 filled in
$PY -m eval.agent.aggregate
push_results "all verified — pass@1 + McNemar in SUMMARY.md"


# ==================================================================================
# BLOCK 10 — FIGURES + DOWNLOAD + FINAL PUSH
# ==================================================================================

cd "$REPO_DIR"
$PY -m eval.agent.plots

git add -f eval/results/ eval/figures/ 2>/dev/null || true
git commit -m "results: all stages complete [$(date '+%Y-%m-%d')]" 2>/dev/null || true
git push origin main

zip -r ~/sg_results_$(date +%Y%m%d).zip eval/results/ eval/figures/
echo "Download from: ~/sg_results_$(date +%Y%m%d).zip"


# ==================================================================================
# MONITORING (run in a separate tmux window while eval is running)
# ==================================================================================

# GPU utilisation + VRAM (ROCm):
watch -n 10 'rocm-smi'

# Progress per arm (updates every 30s):
watch -n 30 'show_progress'

# vLLM request rate (from its log):
tail -f ~/vllm.log | grep -E "Avg prompt|Avg generation|Running|Pending"
