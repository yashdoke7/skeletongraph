# Evaluation harness

Everything needed to reproduce the study: what better code retrieval buys a coding
agent, measured without an agent, in a controlled agent loop, and inside Claude Code
and Codex CLI across releases and benchmarks. The results themselves are summarised
in the top-level [README](../README.md) and recorded, with every caveat, in
[`docs/FINDINGS.md`](../docs/FINDINGS.md).

## Layout

| path | what it is |
|---|---|
| `retrieval_eval.py` | retrieval-only evaluation: one backend, one task file, MRR and recall@k |
| `backends/` | the retrieval backends compared (grep, BM25, dense, hybrid, Aider map, Graphify, codebase-memory-mcp, summary search) |
| `agent/react.py`, `agent/run_stage.py`, `agent/run_agent.py` | the controlled ReAct loop and its stage runner |
| `agent/run_claude_code.py` | Claude Code driver (headless `claude -p`), all Claude arms |
| `agent/run_codex.py` | Codex CLI driver (`codex exec`), all Codex arms |
| `agent/verify.py` | execution-based verification through the official SWE-bench harness |
| `agent/config.py` | stages, arms, run tags, model endpoint |
| `datasets/` | the frozen task sets ([details](datasets/README.md)) |
| `scripts/paper_v2_analysis.py` | every number in the current paper, from run records and transcripts |
| `scripts/make_v2_paper_figures.py`, `scripts/make_paper_figures.py` | the figures |
| `scripts/corpus_audit.py` | an independent census and paired audit of the whole corpus |
| `scripts/` (other) | dataset builders, repository restore, prewarming, and the analyses behind the July 2026 preprint |
| `docs/` | arm definitions and the Claude Code runbook |

## Requirements

- Python 3.11 and `pip install -e ".[all]"` from the repository root.
- Docker, for verification. The SWE-bench harness imports a POSIX-only module, so run
  verification from Linux or WSL, not native Windows Python.
- For the production agents: the Claude Code CLI and Codex CLI (0.155.0 in the study),
  each signed in. Every run records the CLI version it used.
- For the ReAct loop: any OpenAI-compatible endpoint, set with `SG_EVAL_API_BASE`,
  `SG_EVAL_MODEL` and `SG_EVAL_API_KEY`. The study used
  `nvidia/nemotron-3-super-120b-a12b`.
- Results go to `eval/results/agent/<SG_EVAL_RUN_TAG>/`, one JSON per run plus the
  agent transcript. Set `SG_EVAL_RUN_TAG` for every run.

## 1. Task sets and repositories

The committed task files keep the original machine's paths. Write a local copy that
points at a directory of yours, then clone every repository at its base commit:

```bash
python -m eval.scripts.relocate_tasks --dataset eval/datasets/swebench_100.jsonl --root /data/sg
python -m eval.scripts.restore_repos  --dataset eval/datasets/swebench_100.local.jsonl
```

Repeat for `graphify_100`, `swebench_100_prose_stripped`, `swe_rebench_100` and
`swe_rebench_100_prose` as needed; the prose variants share clones with their originals.
Pre-pull the verification images once, so verification can then run offline:

```bash
python -m eval.agent.prefetch --tasks eval/datasets/swebench_100.local.jsonl --workers 8
```

## 2. Retrieval only (no agent)

```bash
for b in grep bm25 sg-rerank bm25-dense bm25-dense-sg; do
  python eval/retrieval_eval.py --dataset eval/datasets/graphify_100.local.jsonl \
    --backend $b --granularity file --k 5 10 20 --out eval/results/paper_verified_${b}_file.json
done
```

## 3. Controlled ReAct loop

Stage `v` defines every arm; choose the ones to run.

```bash
export SG_EVAL_RUN_TAG=nemotron_v4 SG_EVAL_MODEL=nvidia/nemotron-3-super-120b-a12b
python -m eval.agent.run_stage --stage v --dataset eval/datasets/graphify_100.local.jsonl \
  --only-arms none,grep,bm25,fusion,graphify,aider,sg-rerank --workers 4
```

`none` has no search tool but can list and read files; it is not closed-book.

## 4. Claude Code

```bash
export SG_EVAL_RUN_TAG=claude_v8
python -m eval.agent.run_claude_code --dataset eval/datasets/swebench_100.local.jsonl --arm native
python -m eval.agent.run_claude_code --dataset eval/datasets/swebench_100.local.jsonl --arm sg-fusion-plain
```

Arms: `native` (built-in tools only), `sg-fusion` (the retriever with its shipped
integration: rules file, prompt hint, hooks), `sg-fusion-plain` (the same retriever with
no behavioural instructions). `--range A-B` and `--tasks id,id` split a batch across
terminals. The driver runs Claude Code on its own login (an API key in `.env` is
ignored), denies web search and fetch, and records any shell command that reaches for
the network or for unreachable git objects in `leak_flags`.

## 5. Codex CLI

```bash
export SG_EVAL_RUN_TAG=codex_v1
python -m eval.agent.run_codex --dataset eval/datasets/swebench_100.local.jsonl --arm codex-native
python -m eval.agent.run_codex --dataset eval/datasets/swebench_100.local.jsonl --arm codex-sg-plain
python -m eval.agent.run_codex --dataset eval/datasets/swebench_100.local.jsonl --arm codex-sg-fusion
```

Codex runs with a clean configuration directory, web search disabled, and an
unreachable proxy for shell commands.

## 6. Verification

From Linux or WSL:

```bash
python -m eval.agent.verify --run-tag claude_v8 --all
python -m eval.scripts.verify_rebench --tag claude_rebench_v1      # SWE-rebench tags
```

Verification writes `resolved` into each run record. It clears the harness's cached
per-task verdicts for the runs being scored, so a re-run task is never scored against an
old patch (`--reuse-verdicts` opts out), and it keeps per-task Docker images by default so
repeat verification needs no rebuild. Drop `--incremental` after re-running tasks.

## 7. Numbers and figures

```bash
python -m eval.scripts.paper_v2_analysis --json docs/paper/numbers_v2.json
python -m eval.scripts.make_v2_paper_figures        # fig_funnel, fig_settings, fig_release
python -m eval.scripts.corpus_audit                 # census + paired audit -> tmp/corpus_audit.json
```

`docs/paper/numbers_v2.json` is committed, so the figures regenerate without the raw runs.

## Run tags used in the study

| tag | setting | CLI version | arms |
|---|---|---|---|
| `nemotron_v4` | ReAct loop | — | none, grep, bm25, fusion, graphify, aider, sg-rerank |
| `nemotron_v2` | ReAct loop, earlier run | — | 15 retrieval backends |
| `claude_v7` | Claude Code, Verified | 2.1.206–2.1.211 | native, sg-fusion |
| `claude_v7_rep2` | Claude Code, Verified | 2.1.274 (sg-fusion-plain: 2.1.278) | native, sg-fusion, sg-fusion-plain |
| `claude_v8` | Claude Code, Verified | 2.1.278 | native |
| `claude_rebench_v1` | Claude Code, SWE-rebench (first 50) | 2.1.211–2.1.214 | native, sg-fusion |
| `claude_rebench_prose_v1` | Claude Code, SWE-rebench, code removed | 2.1.214 | native, sg-fusion |
| `codex_v1` | Codex CLI, Verified | 0.155.0 | codex-native, codex-sg-fusion, codex-sg-plain |

The run records and transcripts are several gigabytes and are not in git; they are
published with the tagged release. Runs excluded for contamination stay in the
archive, in `<tag>/_quarantine_contaminated_*/`.

## Things that will bite you

- **Record the agent's version.** The same arm on the same tasks behaved very
  differently between Claude Code 2.1.274 and 2.1.278. Compare arms only within one
  release.
- **Report tokens, not dollars.** The provider's prices changed by about a third partway
  through the study.
- **Token fields differ by harness.** Claude Code and Codex report `total_input_tokens`;
  the ReAct loop's `billed_input` already includes cached tokens, so never add
  `cached_input` to it.
- **Single runs are noisy.** At 100 tasks only differences of roughly ten points are
  resolvable; rely on paired comparisons.
- **Confirm each tool was actually used.** Two slow-starting MCP servers were never
  offered to headless Claude Code and recorded zero calls.
