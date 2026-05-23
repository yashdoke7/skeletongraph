# SkeletonGraph — Eval Command Reference

Copy-paste reference for every model × arm × benchmark we run. **PowerShell**
syntax shown (you're using PS); for CMD use `set VAR=val` and `%PY%` instead of
`$env:` and `& $PY`.

> ⚠ **Run-id collision across models.** A run is keyed `task__arm__main__r0` —
> the model is a *label* (`main`), not the real model. The real model is
> `SG_EVAL_MODEL`. So a 14B run **overwrites** a 7B run in the live results dir.
> **Workflow: run one model → aggregate + plots → archive the whole dir →
> clear → next model.** Never run two models into the live dir without archiving.

---

## 0. Environment (run once per shell)

```powershell
cd C:\Users\ASUS\Desktop\CS\Projects\skeletongraph
$PY = "C:\Users\ASUS\AppData\Local\Programs\Python\Python311\python.exe"

# one-time env repair (keeps sentence-transformers working)
& $PY -m pip install -U "huggingface-hub>=1.5.0,<2.0"
& $PY -c "import skeletongraph.graph.embeddings as e; print('SG embeddings:', e.is_available())"  # must be True

# embedder: leave default (MiniLM) for local; jina for the strong/AMD run (see §5)
# $env:SG_EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
```

Ollama (local 7B/14B):
```powershell
$env:SG_EVAL_API_BASE = "http://localhost:11434/v1"
$env:SG_EVAL_API_KEY  = "ollama"
```

---

## 1. Archive / restore results (do BETWEEN models)

```powershell
# After a model's run is fully aggregated + plotted, snapshot it:
$tag = "7b"   # or 14b, nim, 32b-amd
New-Item -ItemType Directory -Force "eval\results\agent_$tag" | Out-Null
Copy-Item eval\results\agent\* "eval\results\agent_$tag\" -Recurse -Force

# Then clear the live dir for the next model:
Remove-Item eval\results\agent\*.json, eval\results\agent\_predictions.jsonl, eval\results\agent\SUMMARY.md -ErrorAction SilentlyContinue
Remove-Item eval\results\agent\figures\* -ErrorAction SilentlyContinue
```
To re-aggregate an archived set later, copy it back into `eval\results\agent\`
first (aggregate/plots always read that fixed dir).

---

## 2. 7B (Ollama)  — local validation: retrieval + efficiency (NO pass@1)

5 core arms already done. **Remaining new arms** (different filenames, safe to
add to the live dir):

```powershell
$env:SG_EVAL_MODEL = "qwen2.5-coder:7b"

# SG ablations (graph / centrality / summary) — 3 arms x 30 = 90 runs
& $PY -m eval.agent.run_stage --stage 0-ablation --workers 2

# Codebase-Memory (needs the binary on PATH or CBMEM_BIN set — see §6b)
& $PY -m eval.agent.run_stage --stage 0-cbmem --workers 2

# combined table (all arms in the dir) + figures
& $PY -m eval.agent.aggregate
& $PY -m eval.agent.plots
```
aider (isolated venv — its hf-hub pin breaks sentence-transformers; already done
for 7B, re-run only if needed):
```powershell
& $PY -m venv .venv-aider
.\.venv-aider\Scripts\python.exe -m pip install aider-chat openai
$env:SG_EVAL_MODEL = "qwen2.5-coder:7b"
.\.venv-aider\Scripts\python.exe -m eval.agent.run_stage --stage 0-aider --workers 2
```
Then **archive to agent_7b** (§1).

---

## 3. 14B (Ollama)  — does the story hold at scale

```powershell
ollama pull qwen2.5-coder:14b
# (archive 7B first, then clear live dir — §1)
$env:SG_EVAL_MODEL = "qwen2.5-coder:14b"

# --limit 10 = first 10 tasks (same 10 as 7B → clean comparison). Drop --limit
# for the full 30.
& $PY -m eval.agent.run_stage --stage 0-full     --workers 2 --limit 10   # 5 ST-arms
& $PY -m eval.agent.run_stage --stage 0-ablation --workers 2 --limit 10
& $PY -m eval.agent.run_stage --stage 0-cbmem    --workers 2 --limit 10
# aider venv:
.\.venv-aider\Scripts\python.exe -m eval.agent.run_stage --stage 0-aider --workers 2 --limit 10

& $PY -m eval.agent.aggregate
& $PY -m eval.agent.plots
# archive to agent_14b (§1)
```

---

## 4. NIM (hosted, build.nvidia.com)  — portability + a stronger model

Free tier is rate-limited, so: few tasks, `--workers 1`, fewest arms.

```powershell
$env:SG_EVAL_API_BASE = "https://integrate.api.nvidia.com/v1"
$env:SG_EVAL_API_KEY  = "<YOUR_NVIDIA_API_KEY>"
# Model: pick a strong coder with >=32k context. Verify it's listed at
# build.nvidia.com and note its context window before running.
$env:SG_EVAL_MODEL = "qwen/qwen2.5-coder-32b-instruct"   # or meta/llama-3.1-70b-instruct
# (archive previous model first — §1)

# Keep it cheap: 3 arms (0-assess = sg/bm25/hybrid), workers 1, 10 tasks.
& $PY -m eval.agent.run_stage --stage 0-assess --workers 1 --limit 10
& $PY -m eval.agent.aggregate --stage 0-assess
# archive to agent_nim (§1)
```
> NIM token note: our prompts are small (retrieval, not whole-repo dumps), so
> 32k context is plenty. The risk is **request count** (free-tier RPM), not
> tokens — hence workers 1 + few tasks. If you hit 429s, lower to --probe.

---

## 5. Strong embedder (jina-code)  — offline retrieval only, no model

Controlled comparison: same embedder for SG **and** hybrid. Build cost is
one-time per repo; query time unaffected.

```powershell
& $PY -m pip install einops          # jina needs it
$env:SG_EMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
# clear stale (384-dim) indexes so they rebuild at the new dimension
Get-ChildItem eval\datasets\repos -Directory | ForEach-Object {
  Remove-Item -Recurse -Force (Join-Path $_.FullName ".skeletongraph"),
                              (Join-Path $_.FullName ".hybrid_index") -EA SilentlyContinue }

& $PY -m eval.retrieval_eval --dataset eval\datasets\stage0.jsonl --backend sg     --k 5 10 20 --granularity file --out eval\results\retrieval_sg_jina.json
& $PY -m eval.retrieval_eval --dataset eval\datasets\stage0.jsonl --backend hybrid --k 5 10 20 --granularity file --out eval\results\retrieval_hybrid_jina.json
# compare latency_ms + recall/precision vs the MiniLM runs (retrieval_sg.json etc.)
```
Decision: if jina lifts SG/hybrid recall without crippling build time → use it
for AMD. Otherwise keep MiniLM and report it as a controlled choice.

---

## 6. Prereqs for the two external-binary arms

**6a. aider** — isolated venv (see §2). Never install aider-chat into the main
env (its `huggingface-hub==1.4.1` pin breaks sentence-transformers).

**6b. Codebase-Memory (cbmem)** — single static binary, no Python conflict:
```powershell
# Windows: download + run the installer from
#   https://github.com/DeusData/codebase-memory-mcp/releases
# then point the harness at it (if not on PATH):
$env:CBMEM_BIN = "C:\path\to\codebase-memory-mcp.exe"

# ONE-TIME validation (confirms index verb + JSON schema for the wrapper):
& $PY -m eval.backends.cbmem --selftest eval\datasets\repos\astropy__astropy-8707
# If the parsed-files list is empty, paste me the raw output — the wrapper's
# index command / JSON keys need a small adjustment to match this version.
```

---

## 7. ContextBench (second benchmark)  — loader built

```powershell
# STEP 1 — confirm the real field names (schema unverified offline):
& $PY -m eval.scripts.extract_contextbench --inspect
# STEP 2 — build python subset (add --field-* overrides if --inspect differs):
& $PY -m eval.scripts.extract_contextbench --n 60 --lang python
# STEP 3 — run retrieval offline on it (model-free):
& $PY -m eval.retrieval_eval --dataset eval\datasets\contextbench.jsonl --backend sg --k 5 10 20 --granularity file --out eval\results\retrieval_cb_sg.json
# For the AGENTIC run, point config.DATASET at contextbench.jsonl (or tell me to
# add a --dataset flag to run_stage) and run the stages as usual.
```
> ContextBench ships human-annotated gold *contexts* — its native metric IS the
> consolidation gap, so it's the strongest external validation of C2.
> Source: https://github.com/EuniAI/ContextBench (arXiv 2602.05892).

## 7b. Single-shot SG — the "why agent" ablation (agent vs no-agent)

Retrieve once → one generation → patch. Runs per model (uses SG_EVAL_MODEL);
recorded as arm `sg-noagent` so it sits next to `sg` in aggregate/plots.
```powershell
$env:SG_EVAL_MODEL = "qwen2.5-coder:7b"
& $PY -m eval.agent.run_singleshot --limit 10      # or --all, or --task-id <id>
& $PY -m eval.agent.aggregate                      # sg vs sg-noagent now both present
```

## 7c. SG ablations

`0-ablation` covers the component ablations: sg-nograph, sg-norerank,
sg-nosummary, sg-noembed (4 arms × tasks). Agent-vs-no-agent is measured
separately by §7b (sg-noagent single-shot).
```powershell
& $PY -m eval.agent.run_stage --stage 0-ablation --workers 2   # add --limit 10 for a quick read
```

---

## 8. IDE testing (Claude Code / Copilot + SG MCP)  — manual

Run the **same short SWE-style prompts** (now applied to all 30 tasks, not 5)
through the IDE with the SG MCP server, native vs +SG. This is the
frontier-model decider (models we can't self-host). Capture: did it locate in
1-2 `sg_search` calls, did the patch pass. See `eval/scripts/parse_ide_trace.py`
for logging the traces. (Prompt-set extension to 30: see §IDE in the response /
ask me to wire it.)

---

## 9. pass@1 / verify  — needs Docker + Linux (do on AMD)

```bash
pip install swebench           # Linux, Docker running
python -m eval.agent.verify --stage 0-full   # writes `resolved` into each json
python -m eval.agent.aggregate               # now includes pass@1 + McNemar
python -m eval.agent.plots                   # adds pass1.png
```
This is the only honest source of pass@1 — defer to the AMD box.

---

## 9b. AMD staged plan (1a/1b parallel → 2 → 3)

Stage 1 = 1a + 1b run **in parallel** (MI300X 192 GB → many isolated tasks at
once). Full isolation, no index cache; parallelism absorbs the build cost.
```bash
export SG_EVAL_API_BASE=http://localhost:8000/v1
export SG_EVAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
export SG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code   # strong embedder

# Stage 1 (parallel): workshop + conference arms + single-shot
python -m eval.agent.run_stage --stage 1a-workshop   --workers 24 &
python -m eval.agent.run_stage --stage 1b-conference --workers 24 &
python -m eval.agent.run_singleshot --all &           # sg-noagent
wait
python -m eval.agent.verify --stage 1a-workshop       # pass@1 (Docker)
python -m eval.agent.verify --stage 1b-conference
python -m eval.agent.aggregate ; python -m eval.agent.plots

# Stage 2: graph competitor + ContextBench (2nd benchmark)
python -m eval.agent.run_stage --stage 2-competitor --workers 24      # cbmem/CodeCompass
python -m eval.agent.run_stage --stage 1a-workshop  --workers 24 \
    --dataset eval/datasets/contextbench.jsonl
python -m eval.agent.verify --stage 2-competitor ; python -m eval.agent.aggregate

# Stage 3 (only if 1-2 justify): variance + learned curator + 2nd language
python -m eval.agent.run_stage --stage 3-further --workers 24         # 3x repeats
# learned curator: train first (docs/CURATOR.md), then sg-learned is live
```
Measure the first 5 tasks' wall time to lock the per-stage budget (~20-35 GPU-hrs
for the full plan; $100 = ~50 hrs).

## 10. AMD final (32B coder, the headline)

Clean dedicated venv (avoid the shared-env dependency hell):
```bash
python -m venv .venv-eval && source .venv-eval/bin/activate
pip install -e . sentence-transformers einops openai numpy
# serve the best 32B coder via vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --enable-auto-tool-choice --tool-call-parser hermes --port 8000
export SG_EVAL_API_BASE=http://localhost:8000/v1
export SG_EVAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
export SG_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code   # strong embedder
# run all stages, then verify (§9) for pass@1.
```
