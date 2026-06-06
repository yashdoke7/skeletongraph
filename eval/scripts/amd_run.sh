#!/bin/bash
set -euo pipefail
# ==================================================================================
# SkeletonGraph — AMD MI300X Runbook (v5 — BATCHED, WINDOW-PRECISE)
# GPU: AMD MI300X 192 GB | Budget: $100 ≈ 50 GPU-h | Agent model: Qwen2.5-Coder-32B
#
# 120B is NOT served here (done via NIM = tag `nemotron_v2`). AMD serves 32B only.
# Mirrors docs/EVAL_PLAN_FINAL.md §7/§9. Copy-paste BLOCK by BLOCK into tmux windows.
#
# BATCHES (each a tmux window/session):
#   SETUP  → pull, deps, model, datasets, comparator envs, PREBUILD graphify graphs
#   SMOKE  → 10 tasks (a slice of workshop) = THE GATE
#   WORKSHOP → baselines+method (win1) ∥ cbmem (win2) ∥ graphify (win3) ∥ aider (win4),
#              THEN ablations (win5). Both benchmarks, 100 tasks.
#   CONFERENCE → same windows, --n 300 Verified / 150 Pro, headline+comparators; + Claude Code
#   VERIFY (rolling) → after each arm-set, verify it in win6 while the next set runs.
#
# WINDOW LAYOUT (total in-flight ≈ 16–20; vLLM is the shared bottleneck):
#   win0  vLLM 32B server (leave running)
#   win1  headline+method  sg-env        --workers 8
#   win2  cbmem            sg-env+.exe    --workers 4
#   win3  graphify         graphify-venv  --workers 4   (graphs prebuilt → fast)
#   win4  aider            aider-venv     --workers 4
#   win5  ablations (after win1–4 free)  sg-env  --workers 12–16
#   win6  rolling verify (Docker/CPU) + monitor
#
# Why baselines+comparators CONCURRENT then ablations ALONE: comparators are CPU/disk
# heavy (cbmem index, aider context, graphify queries) so they interleave with GPU
# generations and keep vLLM saturated; ablations are pure-GPU → run them after, alone.
# ==================================================================================

REPO_DIR="${WORKSPACE_DIR:-/workspace}/skeletongraph"
MODELS_DIR="${WORKSPACE_DIR:-/workspace}/models"
GITHUB_USER="yashdoke7"; GITHUB_REPO="skeletongraph"
GITHUB_PAT="${GITHUB_PAT:-YOUR_PAT_HERE}"
GITHUB_URL="https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
PY="${PYTHON:-python3}"
HEADLINE="sg,sg-rerank,bm25,grep,hybrid,none"
ABLATIONS="sg-chain,sg-embed,sg-seed,summary-dense,sg-nograph,sg-norerank"

# ==================================================================================
# BLOCK A — HELPERS (paste once per tmux attach)
# ==================================================================================
ensure_vllm(){ curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "vLLM OK"; return 0; }; echo "start vLLM (win0, BLOCK 1)"; return 1; }
set_run_tag(){ export SG_EVAL_RUN_TAG="$1" SG_EVAL_API_BASE="http://127.0.0.1:8000/v1" SG_EVAL_API_KEY="EMPTY" SG_EVAL_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"; echo "tag=$SG_EVAL_RUN_TAG dataset=${DS:-<unset>}"; }
push_results(){ cd "$REPO_DIR"; git add -f eval/results/ 2>/dev/null||true; git diff --cached --quiet && { echo "nothing new"; return 0; }; git commit -m "results($SG_EVAL_RUN_TAG): ${1:-checkpoint} [$(date '+%m-%d %H:%M')]"; git pull --rebase origin main 2>/dev/null||true; git push origin main && echo "pushed: ${1:-checkpoint}"; }
show_progress(){ $PY - "${1:-$REPO_DIR/eval/results/agent/$SG_EVAL_RUN_TAG}" <<'PY'
import sys,json,os,collections
d=sys.argv[1]
if not os.path.isdir(d): print("  (none yet)"); sys.exit()
b=collections.defaultdict(lambda:{"n":0,"done":0,"err":0})
for f in os.listdir(d):
    if not f.endswith(".json") or f.startswith("_"): continue
    try:r=json.loads(open(os.path.join(d,f)).read())
    except:continue
    a=r.get("arm","?");b[a]["n"]+=1
    if r.get("stopped") in("submit","max_turns"):b[a]["done"]+=1
    if r.get("stopped")=="error":b[a]["err"]+=1
for a,s in sorted(b.items()):print(f"  {a:14}{s['done']:3d}/{s['n']:3d}  {s['err']}err")
PY
}
# Prebuild graphify graphs ONCE per unique repo in a dataset, on the LOCAL 32B vLLM.
# Times + counts the extraction cost (counted against graphify in the paper).
graphify_prebuild(){ cd "$REPO_DIR"; source .venv-graphify/bin/activate
  export OLLAMA_BASE_URL="http://127.0.0.1:8000/v1" OLLAMA_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct" OLLAMA_API_KEY="EMPTY" GRAPHIFY_OLLAMA_PARALLEL=1
  $PY -m eval.scripts.graphify_prebuild "$1"; deactivate; }

# ==================================================================================
# BLOCK 0 — SETUP (can split across 2 terminals: T1 model+deps, T2 datasets+envs)
# ==================================================================================
# T1:
df -h | head -5                                           # need >= 250 GB free
apt-get update -qq && apt-get install -y git tmux htop curl python3-pip python3-venv
mkdir -p "$(dirname "$REPO_DIR")"; git clone "$GITHUB_URL" "$REPO_DIR" || (cd "$REPO_DIR" && git pull origin main)
cd "$REPO_DIR"; git config user.email "yashdoke215@gmail.com"; git config user.name "Yash Doke"
pip3 install -r requirements.txt swebench huggingface-hub sentence-transformers scikit-learn datasets -q
pip3 install vllm -q     # ROCm mismatch → --extra-index-url https://download.pytorch.org/whl/rocm6.2
mkdir -p "$MODELS_DIR"
$PY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-Coder-32B-Instruct', local_dir='$MODELS_DIR/Qwen2.5-Coder-32B-Instruct', local_dir_use_symlinks=False)"
# T2 (parallel): comparator envs
curl -L https://github.com/seawinde/codebase-memory-mcp/releases/latest/download/cbmem-linux-amd64 -o /usr/local/bin/codebase-memory-mcp && chmod +x /usr/local/bin/codebase-memory-mcp
python3 -m venv .venv-aider    && .venv-aider/bin/pip install aider-chat -q
python3 -m venv .venv-graphify && .venv-graphify/bin/pip install graphifyy openai -q
echo "Setup done → BLOCK 1."

# ==================================================================================
# BLOCK 1 — vLLM 32B (win0; leave running)
# ==================================================================================
cd "$REPO_DIR"; rocm-smi 2>/dev/null || $PY -c "import torch;print('ROCm:',torch.cuda.is_available())"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct" --served-model-name "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --tensor-parallel-size 1 --max-model-len 131072 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000 > ~/vllm.log 2>&1 &
sleep 90 && ensure_vllm

# ==================================================================================
# BLOCK 2 — DATASETS + PREBUILDS + background verify-image prefetch (win5 early)
# ==================================================================================
cd "$REPO_DIR"
$PY eval/make_dataset.py --n 300 --seed 42 --out eval/datasets/swebench_verified.jsonl   # 300 superset; --limit slices it
$PY eval/make_dataset.py --split ScaleAI/SWE-bench_Pro --n 150 --seed 42 --out eval/datasets/swebench_pro.jsonl  # KeyError→add field adapter
# Prebuild graphify graphs ONCE per unique repo (local 32B, fast, no RPD). Reused by 32B AND 120B.
graphify_prebuild eval/datasets/swebench_verified.jsonl
graphify_prebuild eval/datasets/swebench_pro.jsonl
# (Optional) tar the graphs to reuse on the NIM/120B machine — avoids re-extracting there:
#   tar czf ~/graphify_graphs.tgz $(find eval/datasets -type d -name graphify-out)
# Background: prefetch SWE-bench verify Docker images (huge; do it now, not at verify time)
nohup $PY -m swebench.harness.prepare_images --dataset_name princeton-nlp/SWE-bench_Verified --split test --max_workers 8 > ~/prefetch_verified.log 2>&1 &

# ==================================================================================
# BLOCK 3 — SMOKE GATE (10 tasks; ~$0.20). Abort + fix if any arm wedges.
# ==================================================================================
cd "$REPO_DIR" && ensure_vllm; export DS=eval/datasets/swebench_verified.jsonl
set_run_tag "amd_32b_verified"
$PY -m eval.agent.run_stage --stage final-v2 --dataset $DS --limit 10 --workers 16 --only-arms $HEADLINE
show_progress    # sanity: none < others, no arm 90%+ errors

# ==================================================================================
# BLOCK 4 — WORKSHOP (100 tasks, BOTH benchmarks). Run per benchmark.
#   Phase 1: win1 headline+method  ∥  win2 cbmem  ∥  win3 graphify  ∥  win4 aider
#   Phase 2: win5 ablations (after win1–4 done)
#   Set BENCH/DS/TAG, then paste the matching window commands.
# ==================================================================================
# --- Verified ---
export DS=eval/datasets/swebench_verified.jsonl; set_run_tag "amd_32b_verified"; LIM="--limit 100"
# win1: $PY -m eval.agent.run_stage --stage final-v2 --dataset $DS $LIM --workers 8 --only-arms $HEADLINE
# win2: $PY -m eval.agent.run_stage --stage final-comparators --dataset $DS $LIM --workers 4 --only-arms cbmem
# win3: source .venv-graphify/bin/activate; OLLAMA_BASE_URL=http://127.0.0.1:8000/v1 OLLAMA_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct OLLAMA_API_KEY=EMPTY GRAPHIFY_OLLAMA_PARALLEL=1 \
#       SG_EVAL_RUN_TAG=amd_32b_verified SG_EVAL_API_BASE=http://127.0.0.1:8000/v1 SG_EVAL_API_KEY=EMPTY SG_EVAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
#       $PY -m eval.agent.run_stage --stage final-comparators --dataset $DS $LIM --workers 4 --only-arms graphify; deactivate
# win4: source .venv-aider/bin/activate; SG_EVAL_RUN_TAG=amd_32b_verified SG_EVAL_API_BASE=http://127.0.0.1:8000/v1 SG_EVAL_API_KEY=EMPTY SG_EVAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
#       $PY -m eval.agent.run_stage --stage final-comparators --dataset $DS $LIM --workers 4 --only-arms aider; deactivate
#   ↳ when win1–4 done: aggregate + verify (BLOCK 5), then ablations:
# win5: $PY -m eval.agent.run_stage --stage sg-concepts --dataset $DS $LIM --workers 14 --only-arms $ABLATIONS
$PY -m eval.agent.aggregate && push_results "verified workshop done"
# --- Pro --- (repeat the same 5 windows with this DS/TAG)
export DS=eval/datasets/swebench_pro.jsonl; set_run_tag "amd_32b_pro"; LIM="--limit 100"
# ... same win1–win5 commands with $DS/$SG_EVAL_RUN_TAG=amd_32b_pro ...
$PY -m eval.agent.aggregate && push_results "pro workshop done"

# ==================================================================================
# BLOCK 5 — ROLLING VERIFY (win6; CPU/Docker — runs WHILE the next arm-set runs)
# Kick after each arm-set completes; images prefetched in BLOCK 2.
# ==================================================================================
cd "$REPO_DIR"
for tag in amd_32b_verified amd_32b_pro; do
  $PY -m eval.agent.verify --stage final-v2          --run-tag $tag
  $PY -m eval.agent.verify --stage final-comparators --run-tag $tag
  $PY -m eval.agent.verify --stage sg-concepts       --run-tag $tag
done
$PY -m eval.agent.aggregate && push_results "workshop verified — pass@1 + McNemar"

# ==================================================================================
# BLOCK 6 — CONFERENCE (same windows, increased tasks; ablations stay at 100)
#   Verified → --limit 300 ; Pro → --limit 150 ; headline+comparators only.
# ==================================================================================
# Verified 300:
export DS=eval/datasets/swebench_verified.jsonl; set_run_tag "amd_32b_verified"; LIM="--limit 300"
# win1: headline $HEADLINE --workers 8 ; win2 cbmem ; win3 graphify ; win4 aider (as BLOCK 4)
# Pro 150:
export DS=eval/datasets/swebench_pro.jsonl; set_run_tag "amd_32b_pro"; LIM="--limit 150"
# win1–4 as above with this DS/TAG. Then verify (BLOCK 5) + aggregate.
$PY -m eval.agent.aggregate && push_results "conference scale done"
# Claude Code ± SG on 30–50 Pro tasks (frontier agent; external API, ~0 GPU):
# bash eval/scripts/run_claude_code.sh --dataset $DS --limit 50   # MCP wrapper (pending)

# ==================================================================================
# BLOCK 7 — HEADROOM (spend the ~30 GPU-h surplus, step by step — see EVAL_PLAN §7d)
#   7B scaling point, Verified-500, 3-seed variance appendix, more Claude Code.
# Example 7B point (swap the served model in win0 to 7B first):
#   set_run_tag "amd_7b_verified"; $PY -m eval.agent.run_stage --stage final-v2 --dataset $DS --limit 100 --workers 24 --only-arms $HEADLINE
# ==================================================================================

# ==================================================================================
# BLOCK 8 — FIGURES + FINAL PUSH
# ==================================================================================
cd "$REPO_DIR"
for tag in amd_32b_verified amd_32b_pro; do $PY -m eval.scripts.make_figures --tag $tag; done
push_results "figures + final"; zip -r ~/sg_results_$(date +%Y%m%d).zip eval/results/

# MONITORING (win6): watch -n 10 rocm-smi ; watch -n 30 'show_progress' ; tail -f ~/prefetch_verified.log
