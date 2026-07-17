#!/bin/bash
set -euo pipefail
# ==================================================================================
# SkeletonGraph — AMD MI300X Runbook (v7 — corrected 2026-07 after re-checking the
# actual `fusion` implementation. v6's claim that react-loop `fusion` has no dense
# leg was WRONG — see the correction below. v5 (final-v2/sg-concepts stage names)
# is doubly stale — do not resurrect either old version.
#
# GPU: AMD MI300X 192 GB | Budget: $100 ~= 50 GPU-h | Model: Qwen2.5-Coder-32B (bf16)
# AMD serves the 32B react-loop model AND the dense-embedding model (both share
# the GPU fine — the embedder is ~161M params vs the LLM's 32B). The 120B (NIM
# API) and Sonnet (Claude Code, separate account) are NOT served here.
#
# *** CORRECTION (found by re-checking eval/agent/tools.py directly) ***
# arm == "fusion" (what EVERY historical run — nim_fusion_30, nemotron_v4, etc. —
# actually used) dispatches to `_retrieve_fusion3`: a REAL 3-way RRF of BM25 +
# Dense (jina-embeddings-v2-base-code, via eval/backends/dense.py) + SG-structural
# rerank. It DOES need a dense-embedding prewarm (eval/agent/prewarm_fusion_dense.py
# already exists for exactly this). The 2-way (structural+BM25 only, no dense) I
# found last time is a DIFFERENT, separate arm called "sg-fusion" — don't confuse
# the two arm names; "fusion" is the one the paper's react-loop numbers use.
#
# SECOND CORRECTION: the dense cache is keyed PER TASK CHECKOUT DIRECTORY
# (eval/datasets/repos/<task_id>/.skeletongraph/dense_cache/), not per unique
# repo — SWE-bench gives each task its own git-worktree checkout at a distinct
# base_commit, even for repeated repos (10 django tasks = 10 separate django
# checkouts). So prewarm cost scales with N TASKS, not the 12 unique repo names.
#
# What this means for planning: dense prewarm is real GPU work, roughly
# proportional to n. The exact seconds/checkout on MI300X is NOT KNOWN YET —
# CPU cold-encode was measured at ~11 min/repo in an earlier session; GPU with
# batch_size raised via SG_DENSE_BATCH_SIZE (patched in this session — was
# hardcoded at 8, CPU-safe-margin) should be dramatically faster, but "should be"
# is not a number to commit 50 hours against. BLOCK 3 below is a MANDATORY
# PROBE — encode a handful of real checkouts, measure actual seconds, THEN
# decide the final n. Do not skip it (this is the same discipline the old v5
# runbook's "6.0 probe is mandatory" got right — restoring it here).
#
# THREE BENCHMARKING AXES THIS SCRIPT COVERS:
#   Axis A — Retrieval-only (model-independent): grep/bm25/hybrid/fusion/sg-rerank
#            over full SWE-bench Verified (500) + Pro. No LLM calls; cheap; still
#            needs the SAME dense prewarm per checkout for its dense/hybrid arms.
#   Axis B — React loop, 32B Qwen-Coder (the GPU-serving-bound part):
#            arms = fusion, cbmem, graphify, none(baseline). n DECIDED AFTER
#            BLOCK 3's probe — target as large as the probe shows affordable
#            (aim higher than 100; see Block 3/4 for the actual math).
#   Axis C — Claude Code deployment (sg-fusion vs native) — NOT run here; runs
#            on a separate account, see run_claude_code.py. n=50-100 (real $/task,
#            keep modest relative to AMD's free compute).
#
# WINDOW LAYOUT (tmux):
#   win0  vLLM 32B server (leave running for the whole session)
#   win1  react loop: fusion + none        (sg-env)            --workers 8
#   win2  react loop: cbmem                (sg-env, CBMEM_BIN) --workers 4
#   win3  react loop: graphify             (graphify-venv)     --workers 4
#   win4  retrieval-only axis (CPU/GPU-light, can run concurrently once win0-3
#         aren't saturating the GPU — dense encode is small vs LLM serving)
#   win5  rolling verify (Docker/CPU) + monitor
# ==================================================================================

REPO_DIR="${WORKSPACE_DIR:-/workspace}/skeletongraph"
MODELS_DIR="${WORKSPACE_DIR:-/workspace}/models"
GITHUB_USER="yashdoke7"; GITHUB_REPO="skeletongraph"
GITHUB_PAT="${GITHUB_PAT:-YOUR_PAT_HERE}"
GITHUB_URL="https://${GITHUB_USER}:${GITHUB_PAT}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
PY="${PYTHON:-python3}"
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
REACT_ARMS="fusion,cbmem,graphify,none"     # Axis B — the 4 arms decided 2026-07
DS="eval/datasets/sg_final_100.jsonl"       # frozen n=99 seed=42 (12 unique repos).
                                             # Re-freeze bigger AFTER Block 3's probe
                                             # if the probe shows headroom for n>100
                                             # (`$PY eval/make_dataset.py --n <N> --seed 42 --out ...`)
export SG_DENSE_BATCH_SIZE=64                # was hardcoded 8 (CPU-safe); GPU can take
                                             # much more — MI300X has huge VRAM headroom
                                             # for a 161M-param embedder. Bump further
                                             # (128/256) if Block 3's probe shows no OOM.

# ==================================================================================
# BLOCK A — HELPERS (paste once per tmux attach)
# ==================================================================================
ensure_vllm(){ curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "vLLM OK"; return 0; }; echo "start vLLM (win0, BLOCK 1)"; return 1; }
set_run_tag(){ export SG_EVAL_RUN_TAG="$1" SG_EVAL_API_BASE="http://127.0.0.1:8000/v1" SG_EVAL_API_KEY="EMPTY" SG_EVAL_MODEL="$MODEL_NAME"; echo "tag=$SG_EVAL_RUN_TAG dataset=${DS:-<unset>}"; }
# Results are gitignored (by design — see feedback_confirm_before_delete memory).
# -f is REQUIRED or git add silently no-ops on them.
push_results(){ cd "$REPO_DIR"; git add -f eval/results/ eval/datasets/*.jsonl 2>/dev/null||true; git diff --cached --quiet && { echo "nothing new"; return 0; }; git commit -m "results($SG_EVAL_RUN_TAG): ${1:-checkpoint} [$(date '+%m-%d %H:%M')]"; git pull --rebase origin main 2>/dev/null||true; git push origin main && echo "pushed: ${1:-checkpoint}"; }
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
# Prebuild graphify graphs ONCE per unique repo NAME (graphify's own extraction
# is repo-level, unlike the dense cache — check eval/scripts/graphify_prebuild.py
# if this needs confirming per-checkout vs per-reponame before a big run).
graphify_prebuild(){ cd "$REPO_DIR"; source .venv-graphify/bin/activate
  export OLLAMA_BASE_URL="http://127.0.0.1:8000/v1" OLLAMA_MODEL="$MODEL_NAME" OLLAMA_API_KEY="EMPTY" GRAPHIFY_OLLAMA_PARALLEL=1
  $PY -m eval.scripts.graphify_prebuild "$1"; deactivate; }

# ==================================================================================
# BLOCK 0 — SETUP  (do ALL of this BEFORE the 50h clock matters — CPU/network work)
# ==================================================================================
df -h | head -5                                           # need >= 250 GB free
apt-get update -qq && apt-get install -y git tmux htop curl python3-pip python3-venv
mkdir -p "$(dirname "$REPO_DIR")"; git clone "$GITHUB_URL" "$REPO_DIR" || (cd "$REPO_DIR" && git pull origin main)
cd "$REPO_DIR"; git config user.email "yashdoke215@gmail.com"; git config user.name "Yash Doke"
pip3 install -e ".[all]" -q                              # SG + eval extras (pyproject.toml, not requirements.txt)
pip3 install vllm swebench huggingface-hub sentence-transformers scikit-learn datasets -q
mkdir -p "$MODELS_DIR"
$PY -c "from huggingface_hub import snapshot_download as d; d('$MODEL_NAME', local_dir='$MODELS_DIR/Qwen2.5-Coder-32B-Instruct', local_dir_use_symlinks=False)"
# Pre-download the embedder too — don't burn GPU clock on an HF fetch mid-probe:
$PY -c "from sentence_transformers import SentenceTransformer as S; S('jinaai/jina-embeddings-v2-base-code', trust_remote_code=True)"

# --- Competitor arms: NOT yet set up for Linux — this is real pre-flight work ---
# cbmem: Windows .exe was used locally; AMD is Linux — use the official setup script
# (see eval/backends/cbmem.py header) to fetch the Linux binary, then point CBMEM_BIN:
curl -fsSL https://raw.githubusercontent.com/DeusData/codebase-memory-mcp/main/scripts/setup.sh | bash
export CBMEM_BIN="$(command -v codebase-memory-mcp || echo /usr/local/bin/codebase-memory-mcp)"
# graphify: pip package `graphifyy`, its own venv (huggingface_hub pin conflicts with ours)
python3 -m venv .venv-graphify && .venv-graphify/bin/pip install graphifyy openai -q
export GRAPHIFY_BIN="$REPO_DIR/.venv-graphify/bin/graphify"

# --- Task list: already frozen locally at n=99/seed=42 (12 unique repos:
# astropy/django/matplotlib/xarray/pylint/scikit-learn/sphinx/sympy/pytest/
# requests/seaborn/flask) and pushed to the repo — pull it, don't regenerate,
# so the Claude Code / NIM / AMD arms all compare on the IDENTICAL frozen set: ---
test -f "$DS" || $PY eval/make_dataset.py --n 100 --seed 42 --out "$DS"
sha256sum "$DS"     # record in run_manifest.json — this is the frozen list (until Block 3 decides to re-freeze bigger)

echo "Setup done -> BLOCK 1."

# ==================================================================================
# BLOCK 1 — vLLM 32B (win0; leave running the whole session)
# ==================================================================================
cd "$REPO_DIR"; rocm-smi 2>/dev/null || $PY -c "import torch;print('ROCm:',torch.cuda.is_available())"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODELS_DIR/Qwen2.5-Coder-32B-Instruct" --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000 > ~/vllm.log 2>&1 &
sleep 90 && ensure_vllm
# NOTE: gpu-memory-utilization dropped to 0.85 (from 0.90) to deliberately leave
# ~15-20 GB headroom for the embedding model + its batch activations to run
# CONCURRENTLY on the same GPU (Block 2's prewarm can overlap Block 4's react
# loop once both are running — the embedder is tiny next to the 32B LLM's KV
# cache; watch `rocm-smi` VRAM during Block 3 to confirm real headroom before
# relying on this in Block 4/5).

# ==================================================================================
# BLOCK 2 — DENSE PREWARM + graphify PREBUILD (real GPU work, NOT skippable —
# see the correction at the top of this file). Run for the CURRENT $DS only.
# ==================================================================================
cd "$REPO_DIR" && ensure_vllm
time $PY -m eval.agent.prewarm_fusion_dense --dataset "$DS"     # per-task-checkout dense cache
# graphify_prebuild.py FIXED (2026-07): now extracts ONCE per unique repo NAME
# (12 for this dataset) and copies the graph into sibling checkouts, restoring
# the original ~150-calls/repo, ~2-GPU-h-total economics (docs/EVAL_PLAN_FINAL.md
# §8) instead of the ~8-10x-worse per-checkout blowup it would otherwise do.
time graphify_prebuild "$DS"
# cbmem + SG's own structural index build happen automatically per-task inside
# run_stage.py — no separate step needed (mirrors run_claude_code.py's prepare_repo).
# Background: prefetch SWE-bench verify Docker images (big; do now, not at verify time)
nohup $PY -m swebench.harness.prepare_images --dataset_name princeton-nlp/SWE-bench_Verified --split test --max_workers 8 > ~/prefetch_verified.log 2>&1 &

# ==================================================================================
# BLOCK 3 — MANDATORY PROBE (~30-45 min, do NOT skip): converts every timing
# estimate above into a REAL number before committing 50 hours to a specific n.
# Measures: (a) dense-encode seconds/checkout on THIS gpu at SG_DENSE_BATCH_SIZE,
# (b) real react-loop wall-time/task at n=5 for all 4 arms, (c) whether Block 2's
# prewarm and a react run can share the GPU without VRAM contention.
# ==================================================================================
cd "$REPO_DIR" && ensure_vllm
# (a) already timed by Block 2's `time` prefix above — note the per-checkout
#     seconds it printed; multiply by target n for the real prewarm-phase cost.
# (b) 5-task smoke, all 4 arms, TIMED:
set_run_tag "amd_react_probe"
time $PY -m eval.agent.run_stage --stage v --dataset "$DS" --limit 5 --workers 8 --only-arms $REACT_ARMS
show_progress    # sanity: no arm 100% errors, none isn't beating fusion by a mile
# (c) watch VRAM while (b) runs: `rocm-smi` in another pane — confirm no OOM,
#     note peak usage, and how much headroom is left for concurrent prewarm work.
#
# >>> DECISION POINT: take (b)'s wall-clock / 5 tasks, x4 arms, and project the
#     affordable n within the remaining budget (50h minus Block 0-2's real elapsed
#     time minus a ~20% buffer). Re-freeze $DS bigger here if the probe supports it:
#     $PY eval/make_dataset.py --n <TARGET_N> --seed 42 --out eval/datasets/sg_final_<TARGET_N>.jsonl
#     then re-run Block 2's prewarm for ONLY the new tasks beyond the current 99
#     (prewarm_fusion_dense.py skips already-warm checkouts, so this is additive,
#     not a redo) before proceeding to Block 4.

# ==================================================================================
# BLOCK 4 — AXIS B: REACT LOOP, 32B — the main GPU-timed run, AT THE N BLOCK 3
# DECIDED. 4 arms, 3 windows (fusion+none share sg-env; cbmem needs CBMEM_BIN;
# graphify needs its own venv).
# ==================================================================================
set_run_tag "amd_react_32b"
# win1 (sg-env):
$PY -m eval.agent.run_stage --stage v --dataset "$DS" --workers 8 --only-arms fusion,none
# win2 (sg-env, cbmem needs CBMEM_BIN set above):
$PY -m eval.agent.run_stage --stage v --dataset "$DS" --workers 4 --only-arms cbmem
# win3 (graphify-venv — activate it first):
#   source .venv-graphify/bin/activate
#   OLLAMA_BASE_URL=http://127.0.0.1:8000/v1 OLLAMA_MODEL=$MODEL_NAME OLLAMA_API_KEY=EMPTY GRAPHIFY_OLLAMA_PARALLEL=1 \
#   SG_EVAL_RUN_TAG=amd_react_32b SG_EVAL_API_BASE=http://127.0.0.1:8000/v1 SG_EVAL_API_KEY=EMPTY SG_EVAL_MODEL=$MODEL_NAME \
#   python3 -m eval.agent.run_stage --stage v --dataset "$DS" --workers 4 --only-arms graphify
#   deactivate
$PY -m eval.agent.aggregate && push_results "react 32b done"

# ==================================================================================
# BLOCK 5 — AXIS A: RETRIEVAL-ONLY, on the SAME $DS as Block 4 (deliberately —
# reuses the exact checkouts + dense prewarm + graphify graphs already built in
# Blocks 2/3/4, no second clone/encode pass). No LLM calls; mostly CPU + light-GPU
# (dense/hybrid lookups against an already-warm cache) — safe to run concurrently
# with Block 4 once win0's VRAM headroom is confirmed in Block 3(c), or right after.
# ==================================================================================
set_run_tag "amd_retrieval_only_verified"
$PY -m eval.agent.run_stage --stage baseline --dataset "$DS" --workers 16 --only-arms sg,bm25,grep,hybrid,none
$PY -m eval.agent.run_stage --stage final-rerank --dataset "$DS" --workers 16
# Pro (multi-language) — separate axis, own dataset/repos, own prewarm+graphify pass:
$PY -m eval.agent.prewarm_fusion_dense --dataset eval/datasets/swebench_pro.jsonl
graphify_prebuild eval/datasets/swebench_pro.jsonl
set_run_tag "amd_retrieval_only_pro"
$PY -m eval.agent.run_stage --stage baseline --dataset eval/datasets/swebench_pro.jsonl --workers 16 --only-arms sg,bm25,grep,hybrid,none
$PY -m eval.agent.aggregate && push_results "retrieval-only axis done"

# ==================================================================================
# BLOCK 6 — ROLLING VERIFY (win5; CPU/Docker — runs WHILE the next block runs)
# ==================================================================================
cd "$REPO_DIR"
for tag in amd_react_32b amd_retrieval_only_verified amd_retrieval_only_pro; do
  $PY -m eval.agent.verify --all --run-tag $tag
done
$PY -m eval.agent.aggregate && push_results "verified — pass@1 + McNemar"

# ==================================================================================
# BLOCK 7 — HEADROOM (spend surplus hours here, IF Block 3's probe confirms real
# headroom remains — do not assume it, check the actual elapsed-vs-budget math
# after Block 6). Priority order:
#   7a. Model-scale ablation — swap win0's served model to 7B, then 72B, re-run
#       Block 4 on a 60-task subset (fusion+cbmem only). "SG helps across scale"
#       is worth more to reviewers than a bigger single-model n.
#   7b. Bump react n further if 7a isn't needed / time remains.
#   7c. Variance appendix — repeat a 30-task subset x3 seeds for a noise-floor CI.
# ==================================================================================
# 7a example (7B):
#   kill vLLM (win0), restart with --model Qwen2.5-Coder-7B-Instruct, then:
#   set_run_tag "amd_react_7b"; $PY -m eval.agent.run_stage --stage v --dataset "$DS" --limit 60 --workers 16 --only-arms fusion,cbmem

# ==================================================================================
# BLOCK 8 — FIGURES + FINAL PUSH + SHUTDOWN
# ==================================================================================
cd "$REPO_DIR"
for tag in amd_react_32b amd_retrieval_only_verified amd_retrieval_only_pro; do $PY -m eval.scripts.make_figures --tag $tag 2>/dev/null || true; done
push_results "figures + final"
tar czf ~/sg_results_$(date +%Y%m%d).zip eval/results/ eval/datasets/sg_final_*.jsonl run_manifest.json 2>/dev/null || true
echo "Pull the tarball to your laptop BEFORE stopping the instance, then STOP it — an idle GPU still bills."

# MONITORING (win5): watch -n 10 rocm-smi ; watch -n 30 'show_progress' ; tail -f ~/prefetch_verified.log
