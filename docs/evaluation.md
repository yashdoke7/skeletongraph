# SkeletonGraph Evaluation Plan

Status: canonical evaluation plan.
Last updated: 2026-05-08.

This file defines how SkeletonGraph should be measured. The main architecture
and pipeline blueprint lives in `docs/blueprint.md`.

## 1. Evaluation Philosophy

Never report token savings alone.

Cost reduction only matters if quality remains acceptable. A cheaper failed
patch is not a win.

Every evaluation should pair:

```text
cost or token metric + success or quality metric
```

Good metric pairs:

- input tokens + pass rate
- model cost + tests pass
- file reads + missed target rate
- first response latency + user correction count
- route tier + underpower rate

Bad standalone claims:

- "80% fewer tokens" without pass rate
- "uses SLM more often" without correctness
- "smaller packets" without completeness

## 2. Retrieval Evaluation

Goal:

Prove SG finds the right target and enough supporting context.

Dataset:

- 50 prompt-target tasks in small/medium repos
- 20 large repo tasks
- known target functions/files
- known related tests
- known callers/callees where relevant
- include vague prompts such as "users get logged out" that do not lexically
  match obvious function names

Metrics:

- target recall@1/3/5
- precision@k
- MRR
- packet completeness
- missed-test rate
- missed-caller/callee rate
- packet token count
- expansion suggestion usefulness
- stale-index false confidence rate

Example command:

```bash
python eval/offline_retrieval_eval.py --dataset eval/tasks.json
sg metrics
```

Pass bar before strong retrieval claims:

```text
target recall@5 >= 90% on targeted tasks
MRR improves over lexical baseline
missed related test rate decreases over native grep/read baseline
median targeted packet <= 6k tokens
```

## 3. Classifier Evaluation

Goal:

Prove prompt-to-mode classification is reliable enough because routing,
expansion depth, packet shape, and model tier depend on it.

Dataset:

- at least 50 real prompts
- labels for desired packet shape
- labels for expected tier
- ambiguous examples with acceptable fallback labels

Metrics:

- exact mode accuracy
- acceptable-family accuracy
- over-broad classification rate
- under-scoped classification rate
- ambiguity detection rate
- downstream route error rate

Important cases:

- "fix auth token validation" should be targeted/debug.
- "users get logged out" should be investigation unless a strong target exists.
- "refactor payment module to be more testable" should not blindly become
  architecture; it may be feature/refactor with medium or low confidence.
- "review my changes" should use git diff/blast radius, not lexical prompt
  search alone.

## 4. CLI Routing Evaluation

Goal:

Prove dynamic routing beats static model choice.

Compare:

- SG dynamic routing
- always SLM
- always MLM
- always LLM
- manual user tier

Metrics:

- overkill rate
- underpower rate
- pass rate
- cost per passing task
- average input tokens
- average output tokens
- retries per task
- route override frequency
- time to patch

Target before marketing:

```text
>= 25% cost reduction vs always-LLM
<= 3% pass-rate loss vs always-LLM
>= 95% pass-rate parity vs always-MLM on targeted edits
underpower warnings visible on low-confidence tasks
```

Important:

If always-MLM is nearly as cheap and more reliable, SG should say so. The
product can still win through context quality, but the routing claim must be
honest.

## 5. Wrapper Backend Evaluation

Goal:

Prove SG improves existing tools without breaking their native strengths.

Backends to test:

- Aider
- Claude Code
- Codex
- Gemini CLI
- OpenCode
- Continue
- generic Markdown/web-UI handoff

Baselines:

- backend alone
- SG prepared context + backend
- SG route + backend model override when possible
- SG route + SG context + backend

Metrics:

- task pass rate
- total model cost where observable
- number of backend turns
- number of file reads/searches after SG context
- time to first patch
- user approval prompts
- missed target/test rate
- whether backend ignored SG context
- whether backend preserved its native features
- user preference in blind review

Wrapper-specific success criteria:

```text
SG wrapper should not reduce pass rate.
SG wrapper should reduce exploration turns or file reads.
SG wrapper should preserve backend undo/diff/test/apply behavior.
SG wrapper should make route/cost decisions visible.
```

This evaluation is more important than native SG execution evaluation for the
near-term CLI strategy. If Aider, Claude Code, Codex, Gemini CLI, or OpenCode
plus SG context beats native SG execution, SG should embrace that instead of
competing with it.

## 6. CLI Native Execution Evaluation

Goal:

Prove SG's direct provider/local execution can create useful outputs safely.
This is secondary to wrapper evaluation until patch apply and verify are
mature.

Task set:

- 20 fixture bug fixes
- 20 SWE-bench-lite style tasks
- 10 test-generation tasks
- 10 docs/refactor tasks

Metrics:

- patch applies cleanly
- tests pass
- verify catches failures
- files touched outside packet
- hallucinated file/API rate
- time to patch
- run log completeness
- user approval friction

Required instrumentation:

- route
- selected tier/model
- input/output tokens
- cost
- provider duration
- packet path
- response path
- apply status
- verify status

Commands:

```bash
sg run "task" --dry-run --json-output
sg run "task" --execute --json-output
sg verify
sg runs
```

## 7. IDE Compliance Evaluation

Goal:

Prove IDE agents actually use SG in the intended way.

Measure:

- did the agent call SG before broad exploration?
- number of extra file reads after SG packet
- number of SG expansions
- time to first useful answer
- completion reporting rate
- missed target/test rate
- user correction count

Targets:

```text
>= 80% first SG call compliance
<= 1 average SG expansion call per targeted task
>= 70% targeted tasks solved from first packet
>= 80% completion reporting when rules/hooks are installed
```

If an IDE ignores SG rules frequently, rules must be shorter or integration
must use a stronger hook/extension path.

## 8. Human Friction Evaluation

Goal:

Prove real users can adopt SG without handholding.

Metrics:

- time from install to first packet
- commands to first useful packet
- config failure rate
- API-key confusion rate
- local provider setup success
- docs task completion rate
- model override frequency

Watch especially:

- users thinking CLI always needs an API key
- users not understanding IDE vs CLI model names
- users expecting patch apply before it exists
- users confusing route recommendation with quality guarantee
- users assuming SG is tied to only one backend

## 9. Cost Evaluation

Goal:

Prove SG reduces cost per successful task.

Metrics:

- input token cost
- output token cost
- provider total cost
- turns per task
- tool-call overhead
- cache hit rate if provider supports it
- cost per passing task

Report format:

```text
Task group: targeted bug fixes
Baseline: backend alone or always-LLM native agent
SG route: dynamic
Pass rate: X vs Y
Median cost per passing task: X vs Y
Median turns: X vs Y
Median file reads: X vs Y
```

## 10. Release Gates

Before claiming retrieval quality:

- retrieval dataset exists
- target recall and MRR reported
- packet completeness reported
- stale-index behavior tested

Before claiming routing saves money:

- dynamic route compared to always-MLM and always-LLM
- cost per passing task reported
- underpower/overkill rates reported

Before claiming wrapper value:

- at least two backends tested
- backend-alone baseline exists
- SG+backend does not reduce pass rate
- SG+backend reduces exploration/file reads or improves speed

Before claiming native CLI agent quality:

- safe apply exists
- `sg verify` exists
- real provider and local model smokes pass
- patch pass rate and hallucinated-file rate reported
