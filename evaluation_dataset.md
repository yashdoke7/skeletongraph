# SkeletonGraph Evaluation Datasets

This file defines which datasets prove which part of SkeletonGraph. The core rule: do not use one dataset to claim something it does not measure.

## Proof Matrix

| Dataset / run type | Primary purpose | Measures well | Does not prove by itself |
|:---|:---|:---|:---|
| Smoke/dry-run | Pipeline readiness | Build, MCP tool availability, parser shape, report generation | Product quality or token savings |
| Copilot paired real-agent runs | Product workflow | Native vs SG tool-output tokens, turns, redundant reads, export reliability | General benchmark credibility unless task set is disclosed |
| SWE-bench Verified | Execution-quality credibility | File localization, patch quality, FAIL_TO_PASS/PASS_TO_PASS outcomes | IDE-specific UX unless run through the target agent |
| Large repo stress | Scaling proof | Index size, context budget behavior, retrieval token slope | Bug-fix success unless coupled to real tasks/tests |
| CRG-compatible replay | Competitor-methodology comparison | How SG compares to code-review-graph style static/context benchmarks | Primary claim of agent quality |
| Custom golden prompts | Demo coverage | Qualitative examples and product storytelling | Research-grade evidence unless tasks/tests are public and reproducible |

## Dataset 1: SWE-bench Verified

**Purpose:** Primary credibility dataset for execution quality. SWE-bench Verified contains human-validated GitHub issues with task-specific tests. It is the best choice for proving that SG does not reduce tokens by starving the agent of needed context.

**Source:** [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

**Use for claims:**

- File localization: compare retrieved files against gold patch files.
- Execution quality: apply the agent patch and run FAIL_TO_PASS/PASS_TO_PASS tests.
- Token efficiency: compare paired native vs SG traces for the same agent and task.
- Credibility: report per-task data, aggregate statistics, confidence intervals, and evidence-quality warnings.

**Do not claim:** execution quality until patch apply and tests are actually run.

### Repos in SWE-bench Verified

| # | Repo | Approx files | Best for |
|:--|:---|---:|:---|
| 1 | `django/django` | ~4,600 | Large codebase scaling |
| 2 | `scikit-learn/scikit-learn` | ~1,200 | ML library architecture |
| 3 | `matplotlib/matplotlib` | ~2,000 | Deep class hierarchies |
| 4 | `astropy/astropy` | ~2,500 | Scientific computing patterns |
| 5 | `pandas-dev/pandas` | ~800 | Data transformation pipelines |
| 6 | `pylint-dev/pylint` | ~600 | Static analysis / AST code |
| 7 | `pytest-dev/pytest` | ~400 | Hook/plugin architecture |
| 8 | `psf/requests` | ~150 | Small HTTP library |
| 9 | `pallets/flask` | ~100 | Small web framework |
| 10 | `marshmallow-code/marshmallow` | ~100 | Serialization library |
| 11 | `mwaskom/seaborn` | ~200 | Visualization library |
| 12 | `getpelican/pelican` | ~150 | Static site generator |

### Recommended Order

1. `psf/requests`: small enough for parser and export rehearsal.
2. `pallets/flask`: tiny and fast, useful for end-to-end smoke.
3. `pytest-dev/pytest`: medium complexity and strong plugin structure.
4. `django/django`: scaling proof once the pipeline is already stable.

### Task Fields To Preserve

```text
instance_id        exact task id
repo               repository name
base_commit        clean checkout point
problem_statement  exact prompt for both native and SG runs
patch              human gold solution, used for file localization
test_patch         task test additions
FAIL_TO_PASS       tests that must pass after the fix
PASS_TO_PASS       tests that must not regress
```

## Dataset 2: Copilot Paired Real-Agent Benchmark

**Purpose:** This is the most direct product benchmark for the current goal: compare how a real IDE coding agent behaves with its normal pipeline versus the same agent with SG MCP tools enabled.

**Protocol:**

1. Same agent version, model, IDE, repo, base commit, and prompt.
2. Native run first, SG disabled.
3. Export raw native session immediately.
4. Reset repo to base commit.
5. SG run second, MCP enabled, same prompt.
6. Save `.skeletongraph/session/current.json` immediately.
7. Parse both traces and aggregate only paired tasks.

**Metrics:** token efficiency, retrieval quality against gold patch files, operational efficiency, evidence-quality warnings, and eventually execution quality if patches/tests are collected.

**Minimum credible sample:** 10 tasks for internal directional evidence, 30+ tasks for public early claims.

## Dataset 3: Smoke / Dry-Run Dataset

**Purpose:** Cheap verification that the pipeline works before spending quota.

A smoke run can use one or two known tasks and even tiny local traces, but its report must be labeled as smoke-only. It should prove:

- `skeletongraph build` succeeds.
- MCP tools are available and schemas are sane.
- `query_context` returns context.
- Native export parser handles the current export format.
- SG session parser attributes the active agent correctly.
- `eval-benchmark` emits Evidence Quality warnings instead of silently producing fake credibility.

Smoke runs are not benchmark results.

## Dataset 4: Large Repo Stress Test

**Purpose:** Scaling proof for the core SG thesis: the cost of high-level context discovery should grow much slower than repository size.

| Repo | Approx files | Why it matters |
|:---|---:|:---|
| `django/django` | ~4,600 | Large SWE-bench overlap |
| `huggingface/transformers` | 10,000+ | Huge Python/ML architecture |
| `vercel/next.js` | 20,000+ | Large JS/TS monorepo |

**Publishable claim shape:** "On large repositories, SG keeps first-pass context retrieval bounded while native broad search/read workflows grow with repo size."

**Not publishable without tests:** "SG fixes bugs better on large repos."

Any expected reduction table must be labeled as a hypothesis until measured. Do not publish hypothetical 10x/24x values as results.

## Dataset 5: CRG-Compatible Replay

**Purpose:** A fair comparison against `code-review-graph`-style methodology and similar static graph/context systems.

| Aspect | CRG-style static benchmark | SG target benchmark |
|:---|:---|:---|
| Baseline | Static file/context selection | Real native agent trace plus optional static replay |
| Token counter | Often approximate character/token heuristics | BPE/tokenizer counts where possible |
| Ground truth | Commit/review artifacts or graph edges | Gold patch files, task tests, real traces |
| Output | Aggregate reduction numbers | Per-task traces, evidence warnings, paired comparisons |

Use this benchmark to say whether SG is better on their terms. Do not make it the primary proof of real coding-agent quality.

## Dataset 6: Custom Golden Prompts

**Purpose:** Demonstrations and regression checks on familiar repos. These are useful for product storytelling and manual QA, not standalone research evidence.

| # | Project | Prompt type |
|:--|:---|:---|
| 1 | Flask | Small web framework routing/config bug |
| 2 | Requests | HTTP/session lifecycle bug |
| 3 | Pytest | Hook/fixture behavior bug |
| 4 | Pandas | Dataframe/index behavior bug |
| 5 | Transformers | Large ML repo generation behavior bug |

For credibility, every custom task should include repo URL, base commit, exact prompt, expected files, test command, and pass/fail criteria.

## Publication Guardrails

A result is publishable only if:

- The raw traces are retained.
- Every aggregate row is built from paired native/SG runs.
- The same task prompt and base commit are used for both conditions.
- Parser warnings are disclosed.
- Token levels are labeled as measured, estimated, or unavailable.
- Retrieval matching is path-level or explicitly marked as ambiguous.
- Execution-quality claims include patch application and tests.

A result is not publishable if it relies on placeholders, dummy traces, one-sided SG-only runs, hypothetical expected reductions, or manual cherry-picking without disclosure.
