# SWE-bench Pro — pass@1 verification (the real one)

This is the missing piece: actual **resolve rate** on SWE-bench Pro, the same way
`verify.py` gives it for Verified. Pro is **not** the standard SWE-bench harness —
it uses **scaleapi/SWE-bench_Pro-os** with **prebuilt Docker images**
(`jefzda/sweap-images`), so you don't build images. It MUST run on a real Docker
host (AMD/Linux box, or Modal cloud). Our run `task_id` IS the Pro `instance_id`
(verified 99/99), so patches map directly.

> ⚠️ **A no-op run is not a 0%.** If Docker/Modal isn't actually executing, the
> harness writes `eval_results.json` = all-`false` with **empty** per-instance
> `workspace/` dirs and **no** `*_output.json`. That is "**not evaluated**", not a
> real pass@1 of 0. The `check` step below detects this; `writeback` refuses to
> record it. Always run `check` before believing any number.

## 0. Prereqs (on the Docker host — AMD/Linux)
```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os
cd SWE-bench_Pro-os
pip install -r requirements.txt
pip install docker            # needed for --use_local_docker
docker info                   # MUST succeed — confirms the daemon is up
# swe_bench_pro_full.csv ships in the repo / HF dataset (instance metadata +
# dockerhub_tag per instance).
```

## 1. Gather our patches → scaleapi format (on this machine)
```powershell
# one file per arm — the harness keys by instance_id + prefix
python -m eval.agent.verify_pro gather --results eval/results/agent/nemotron_pro --arm fusion    --out eval/results/pro_preds_fusion.json
python -m eval.agent.verify_pro gather --results eval/results/agent/nemotron_pro --arm sg-rerank --out eval/results/pro_preds_sgrerank.json
python -m eval.agent.verify_pro gather --results eval/results/agent/nemotron_pro --arm bm25      --out eval/results/pro_preds_bm25.json
```
Each is a JSON list of `{instance_id, patch, prefix}`. Coverage is printed
(empty patches dropped — they'd fail anyway). Copy these to the Docker host.

## 2a. SMOKE TEST FIRST — one instance, confirm it actually executes
Do **not** launch all 81 until a single instance produces logs. Use local Docker
(direct daemon, no Modal account needed):
```bash
# tiny preds file with ONE ansible instance, then run just it
python - <<'PY'
import json
p = json.load(open("pro_preds_fusion.json"))
json.dump(p[:1], open("pro_preds_smoke.json","w"), indent=2)
print("smoke instance:", p[0]["instance_id"])
PY

python swe_bench_pro_eval.py \
    --raw_sample_path=swe_bench_pro_full.csv \
    --patch_path=pro_preds_smoke.json \
    --output_dir=results_smoke \
    --scripts_dir=run_scripts \
    --num_workers=1 \
    --use_local_docker \
    --dockerhub_username=jefzda
```
Then **audit it** (this is the gate — copy `results_smoke` back to this machine, or
run from the skeletongraph repo dir on the host):
```powershell
python -m eval.agent.verify_pro check --harness-dir <path>/results_smoke --arm fusion
```
- **REAL RUN** (exit 0): you'll see `EXECUTED: 1` and a `*_output.json` /
  `*_stdout.log` in the instance dir. Proceed to step 2b.
- **NULL RUN** (exit 2): `EXECUTED: 0`. Docker didn't run the container. Fix the
  host (`docker info`, image pull, `--use_local_docker`) before wasting time on 81.

## 2b. Run the full arm (on the Docker host)
```bash
python swe_bench_pro_eval.py \
    --raw_sample_path=swe_bench_pro_full.csv \
    --patch_path=pro_preds_fusion.json \
    --output_dir=results_fusion \
    --scripts_dir=run_scripts \
    --num_workers=16 \
    --use_local_docker \
    --dockerhub_username=jefzda
```
It pulls the prebuilt image per instance (`dockerhub_tag`), applies the patch,
runs the fail2pass + pass2pass tests, and writes per-instance `*_output.json`
plus `eval_results.json` in `--output_dir`. **This is the AMD compute step.**
Repeat per arm (`pro_preds_sgrerank.json` → `results_sgrerank`, etc.).
On Modal instead of local Docker: drop `--use_local_docker` and authenticate
`modal token new` first — otherwise it silently no-ops into a NULL RUN.

## 3. Audit, THEN write pass@1 back into our run JSONs
```powershell
# (a) audit — must say "REAL RUN" before you trust anything
python -m eval.agent.verify_pro check --harness-dir <host>/results_fusion --arm fusion

# (b) writeback — execution-gated: only EXECUTED instances get a resolved bool;
#     un-executed ones are recorded as resolved=null / pro_evaluated=false (NOT
#     counted as failures). Refuses entirely on a NULL RUN.
python -m eval.agent.verify_pro writeback --results eval/results/agent/nemotron_pro --arm fusion --harness-dir <host>/results_fusion
```
`writeback` prints `pass@1 among evaluated = resolved/(resolved+unresolved)`.
(The legacy `--harness-report <file>.json` path still exists but needs `--force`
because a bare `{id:bool}` file carries no execution evidence.)

## 4. Read pass@1
```powershell
python -m eval.agent.aggregate --run-dir eval/results/agent/nemotron_pro
```
`aggregate` shows `pass@1` per arm. An arm with **no** real verdict (harness
never ran / unevaluated) shows **`n/a`**, never a fake `0.0%` — empty-patch fails
only count inside an arm that was actually evaluated.

## Notes
- **Coverage matters for the paper.** Report pass@1 as `resolved / evaluated`
  and state coverage (how many of the 99 were executed). `writeback` prints both.
- **Weak model caveat.** nemotron max-turns ~42% on Pro; expect low pass@1 for
  *every* arm. A meaningful pass@1 comparison needs a STRONGER model (Claude
  wrapper / a capable model served on AMD) generating the patches — the harness
  is model-agnostic, so swap the agent, re-gather, re-run.
- **Why this was missing:** we measured Pro *retrieval* only; this closes the
  loop to the actual task outcome — but only once `check` says REAL RUN.
```
