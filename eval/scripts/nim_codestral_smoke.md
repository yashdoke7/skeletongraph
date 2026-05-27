# NIM Codestral-22B smoke run — commands

Smoke test that validates the SkeletonGraph pipeline on a coder-tuned,
non-thinking, low-latency model **before** spending AMD MI300X credit.
The model on NIM is `mistralai/codestral-22b-v0.1` — closest non-thinking
coder model to AMD's planned Qwen2.5-Coder-32B-Instruct.

We DO **NOT** disturb the in-flight contextbench-on-llama-70b run. This uses
a separate `SG_EVAL_RUN_TAG` so results land in their own folder.

> Why this model: coder-tuned (unlike Llama-3.x), non-thinking by default
> (unlike DeepSeek R1 / Qwen3 thinking modes you tried), sized close to the
> AMD target (22B vs 32B), low latency on NIM. Paper-defensible as the
> smoke proxy.

## 0. One-time setup (WSL / native shell)

```bash
# Inside the skeletongraph repo
source ~/sg-env/bin/activate    # or however you activate the venv
cd /mnt/c/Users/ASUS/Desktop/CS/Projects/skeletongraph

# NIM API endpoint for Codestral
export SG_EVAL_API_BASE="https://integrate.api.nvidia.com/v1"
export SG_EVAL_MODEL="mistralai/codestral-22b-v0.1"

# Use the rotation if you have multiple NIM accounts (recommended):
export SG_EVAL_API_KEYS="nvapi-KEY1,nvapi-KEY2,nvapi-KEY3,nvapi-KEY4"
# OR a single key:
# export SG_EVAL_API_KEY="nvapi-..."

# Separate results tag so we don't clobber nim70b_swebench_v2
export SG_EVAL_RUN_TAG="codestral22b_swebench_smoke"
```

## 1. SWE-bench smoke — 5 baseline arms × 30 tasks (the same dataset as v2)

```bash
# Workers low-ish because Codestral hits NIM's per-account rate limit fast;
# with 4 keys × workers 4 = 16 concurrent requests, well under typical caps.
python -m eval.agent.run_stage \
    --stage baseline \
    --workers 4

# Show progress as it goes
python -c "
import json, os, collections
d = 'eval/results/agent/codestral22b_swebench_smoke'
by_arm = collections.defaultdict(lambda: {'n':0, 'done':0, 'err':0})
for f in os.listdir(d):
    if not f.endswith('.json') or f.startswith('_'): continue
    try: r = json.loads(open(os.path.join(d, f)).read())
    except: continue
    s = by_arm[r.get('arm','?')]
    s['n'] += 1
    if r.get('stopped') in ('submit','max_turns'): s['done'] += 1
    if r.get('stopped') == 'error': s['err'] += 1
for arm, s in sorted(by_arm.items()):
    print(f'  {arm:8} {s[\"done\"]:3d}/{s[\"n\"]:3d}  errors={s[\"err\"]}')
"
```

## 2. Aggregate retrieval/efficiency table

```bash
python -m eval.agent.aggregate
# Inspect the new SUMMARY.md:
cat eval/results/agent/codestral22b_swebench_smoke/SUMMARY.md
```

## 3. cbmem on the same 30 tasks

```bash
# Linux WSL: cbmem binary
curl -L https://github.com/seawinde/codebase-memory-mcp/releases/latest/download/cbmem-linux-amd64 \
    -o ~/codebase-memory-mcp
chmod +x ~/codebase-memory-mcp
export CBMEM_BIN="$HOME/codebase-memory-mcp"
"$CBMEM_BIN" --version

# Validate before running the full stage
python -m eval.backends.cbmem --selftest eval/datasets/repos/django__django-14725

# Stage
python -m eval.agent.run_stage --stage 0-cbmem --workers 4
```

## 4. graphify on the same 30 tasks (optional — only if selftest passes)

```bash
pip install graphifyy -q
python -m eval.backends.graphify --selftest eval/datasets/repos/django__django-14725
# If selftest fails, skip graphify — DO NOT run the stage with a broken backend.

# Stage (note: only-arms because the smoke stage covers 7 arms; we already did 5)
python -m eval.agent.run_stage --stage 0-smoke --only-arms graphify --workers 4
```

## 5. SWE-bench Docker pass@1 verify (the gate metric)

```bash
# Docker must be running in WSL (Docker Desktop with WSL2 integration)
docker --version
docker ps   # should not error

pip install swebench -q

# Per-arm verify (writes resolved=True/False into each run JSON)
python -m eval.agent.verify --stage baseline    --run-tag codestral22b_smoke
python -m eval.agent.verify --stage 0-cbmem     --run-tag codestral22b_smoke
# (graphify uses stage 2-competitor or 0-smoke depending on how you ran it)
python -m eval.agent.verify --stage 0-smoke     --run-tag codestral22b_smoke

# Final aggregate with pass@1 + McNemar filled in
python -m eval.agent.aggregate
cat eval/results/agent/codestral22b_swebench_smoke/SUMMARY.md
```

## 6. Expected runtime + cost

| step | wall time | NIM credit |
|---|---|---|
| Baseline 5 × 30 = 150 runs | 1.5–2.5 hr at 4 workers × 4 keys | included in your NIM allowance |
| cbmem × 30 | 0.5 hr | included |
| graphify × 30 (if run) | 0.5 hr | included |
| Docker verify (≈210 patches) | 2–4 hr first time (image pulls) | local CPU |
| **TOTAL** | **4–7 hr** | **free** |

## 7. What to compare against

After the smoke:

| arm | metric | NIM-70B-v2 (corrected) | Codestral-22B (this run) | AMD Qwen32B (target) |
|---|---|---|---|---|
| sg  | edited-gold |  0.60 | _your number_ | _AMD target ≥ 0.65_ |
| sg  | pass@1      |  0.133 | _your number_ | _AMD target ≥ 0.20_ |
| sg  | in-tok mean |  63K  | _your number_ | _AMD target ≤ 60K_ |

If Codestral-22B numbers are in the same neighborhood as NIM-70B-v2 corrected
(±20% on edited-gold, hybrid not collapsed, cbmem comparable), the AMD pipeline
is validated and you can boot MI300X with confidence.
