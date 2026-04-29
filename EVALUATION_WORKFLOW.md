# SkeletonGraph Evaluation Workflow

> Canonical workflow for evaluating SkeletonGraph as an MCP context-optimization layer for coding agents.

SkeletonGraph should be evaluated as an additive pipeline: native coding agent baseline first, then the same agent with SG-enabled MCP tools. SG is not a replacement for the agent's native tools. The credible claim is: "SG reduces context/retrieval tokens while preserving or improving retrieval and patch quality."

## What Must Be 10/10 Before Spending Quota

These are non-result blockers. Do not start expensive Copilot/SWE runs if any fail:

| Area | Required state | Why critics will care |
|:---|:---|:---|
| Pairing | Every task has one native trace and one SG trace from the same agent, prompt, repo, and base commit | Otherwise token savings and quality deltas are confounded |
| Isolation | Repo is reset to the exact base commit before each run | Otherwise the second run benefits from the first run's edits |
| Export | Raw native export and raw SG session are saved before another chat/run can overwrite them | Otherwise results are not auditable |
| Parser | `eval-parse` succeeds for native and SG traces and reports non-placeholder prompts/tool calls | Otherwise the metrics are measuring parser failure |
| SG tools | MCP schemas expose optional parameters correctly and all integration prompts say SG-first, not SG-only | Otherwise SG can degrade agents by forbidding useful native fallback |
| Index | `skeletongraph build` and incremental updates handle modified/deleted files without stale edges | Otherwise retrieved context can be wrong |
| Evidence | `eval-benchmark` Evidence Quality has no warnings except intentional tiny smoke-run warnings | Otherwise published numbers look like marketing |
| Claims | No README/paper claim uses dummy traces, hypothetical reductions, or unverified smoke numbers | Otherwise competitors can dismiss the evaluation immediately |

## SG-First Retrieval Ladder

The intended pipeline is high-level to low-level, with controlled page faults:

1. `query_context`: first-pass context router from task prompt to zones, likely files, likely symbols, constraints, and confidence.
2. `search_index`: discovery when the target symbol/file is unclear or confidence is LOW/MEDIUM.
3. `view_file_outline`: file-level map before reading full code.
4. `show_graph`, `get_dependencies`, `get_blast_radius`: relationship traversal and impact analysis.
5. `expand_function`: full function/class body when skeleton context is insufficient.
6. `view_file_range`: bounded raw lines for exact verification.
7. `grep_codebase`: scoped textual fallback inside SG.
8. Native IDE tools: allowed fallback for low confidence, unindexed/generated files, missing paths, unsupported language/parser gaps, or small verification reads before editing.

This ladder is the product. The goal is not to force fewer tool calls at all costs; the goal is to make broad search cheap and let the agent spend tokens only where precision requires it.

## Confidence And Context Insufficiency

`confidence` answers "how likely did SG identify the right entry point?" It does not prove the returned context is sufficient for editing.

| State | Meaning | Correct next action |
|:---|:---|:---|
| HIGH | Exact file/function/entity match or strong structural hit | Use the returned context; expand only the specific body/range needed |
| MEDIUM | BM25/summary/broad index match with multiple plausible candidates | Use `search_index`, outlines, and graph tools to narrow |
| LOW | No direct match, weak keyword-only matches, or very few ambiguous hits | Use `search_index`, `grep_codebase`, then bounded native search if needed |
| Context insufficient | The entry point may be correct, but the agent lacks exact implementation details | Use `expand_function`, `view_file_range`, dependencies, or native bounded read |

A good SG run can still have "context insufficient" moments. That is not failure; it is the designed page-fault mechanism. Failure is when the pipeline leaves the agent with no clear next retrieval step or silently blocks native fallback.

## Metrics

| Dimension | Metrics | What it proves | Publishability |
|:---|:---|:---|:---|
| A: Token efficiency | Retrieval/tool output tokens, measurable conversation tokens, schema overhead, cost | SG reduces context load | Publish only with paired traces |
| B: Retrieval quality | Precision, recall, F1, MRR, Hit@k against gold patch files | SG finds the right files | Publish with path-level matches, not basename-only guesses |
| C: Execution quality | Patch apply rate, FAIL_TO_PASS, PASS_TO_PASS/regression rate | SG does not harm coding quality | Requires SWE-bench-style execution harness |
| D: Operational efficiency | Turns, tool calls, redundant file views, wall time | SG changes workflow efficiency | Useful secondary evidence |
| E: Evidence quality | paired trace count, placeholder prompts, missing calls, parser warnings | Results are auditable | Must be reported with every benchmark |

Token accounting must separate retrieval/tool-output tokens from total conversation tokens. If a platform export does not expose full model tokens, mark that level as unavailable instead of estimating silently.

## Recommended Benchmark Matrix

| Benchmark | Purpose | Minimum credible size | Claim allowed |
|:---|:---|---:|:---|
| Smoke/dry-run | Pipeline and parser readiness | 1-3 tasks | No product claim |
| Copilot paired real-agent run | Main product demonstration for IDE workflow | 10+ tasks before public claims | SG vs native agent token/workflow delta |
| SWE-bench Verified subset | Execution quality credibility | 30+ tasks for early paper-quality signal | Token savings plus no quality regression |
| SWE-bench Verified larger run | Stronger publication-grade result | 100+ tasks when quota allows | Statistically stronger aggregate claims |
| Large repo stress | Scaling and context-budget proof | 3+ repos, fixed tasks | Scaling/token behavior, not patch quality unless tested |
| CRG-compatible replay | Methodology comparison with code-review-graph style work | Same repos/commits as competitor | Comparative methodology, not primary quality proof |

SWE-bench Verified is the strongest public benchmark for execution quality. It is not the only benchmark needed because SG's thesis is context routing, token efficiency, and tool integration. Pair SWE-bench with real-agent IDE traces for the product claim.

## Copilot Quota-Safe Procedure

Use this order when quota is precious:

1. Checkout target repo at the task base commit.
2. Run native Copilot first with SG disabled and the exact task prompt.
3. Export immediately with VS Code `Chat: Export Session...` and save the raw file under `eval_logs/copilot/<project>/<task_id>/native_export.json`.
4. Reset the repo to the same base commit.
5. Enable SG MCP, run `skeletongraph build`, and run the exact same prompt.
6. Copy `.skeletongraph/session/current.json` immediately to `eval_logs/copilot/<project>/<task_id>/sg_session.json` before any other SG run.
7. Parse native: `skeletongraph eval-parse --agent copilot --mode native --file <native_export.json> --path <repo> --project <project>`.
8. Parse SG: `skeletongraph eval-parse --agent copilot --mode skeletongraph --file <sg_session.json> --path <repo> --project <project>`.
9. For benchmark aggregation, place parsed traces as `benchmark_traces/<task_id>/native_trace.json` and `benchmark_traces/<task_id>/sg_trace.json`.
10. Run `skeletongraph eval-benchmark --dataset swe-bench-verified --traces-dir benchmark_traces --output benchmark_results`.

Native-first is recommended for Copilot quota runs because it captures the unassisted baseline before any SG-induced mental/model state can affect the operator. If SG is run first, save its `current.json` immediately and still reset before native.

## SWE-bench Procedure

For each SWE-bench task:

1. Load `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS`.
2. Checkout the repo at `base_commit`.
3. Run native agent with the problem statement and save raw trace/export.
4. Reset to `base_commit`.
5. Build SG index and run the same agent/prompt with SG enabled.
6. Save raw SG session.
7. Parse both traces.
8. Apply each produced patch in a clean worktree.
9. Run the task tests: FAIL_TO_PASS must pass, PASS_TO_PASS must still pass.
10. Aggregate token, retrieval, execution, operational, and evidence-quality metrics.

Until steps 8-9 are automated and run, treat SWE-bench results as retrieval/workflow evidence only, not execution-quality evidence.

## Pre-Quota Readiness Checklist

Run these locally before burning agent quota:

```bash
python -m compileall -q src tests
python -m pytest -q
skeletongraph build --path <target_repo>
skeletongraph serve --path <target_repo>  # or connect through IDE and verify tools/list
skeletongraph eval-parse --agent copilot --mode native --file <one_export.json> --path <target_repo> --project <project>
skeletongraph eval-parse --agent copilot --mode skeletongraph --file <one_sg_session.json> --path <target_repo> --project <project>
skeletongraph eval-benchmark --dataset swe-bench-verified --limit 1 --traces-dir benchmark_traces --output benchmark_results
```

The final smoke report should show Evidence Quality warnings only for intentional tiny sample size. Warnings for placeholder prompts, missing paired traces, no tool calls, or parser failures must be fixed before real runs.

## Do Not Publish Criteria

Do not publish or market a result if any of these are true:

- The prompt is `dummy`, `unknown`, synthetic without disclosure, or not the original task text.
- There is no paired native trace for the SG trace.
- The repo was not reset between runs.
- The parser inferred files from ambiguous basenames only.
- Native exports failed and SG is compared against static whole-repo or hand-estimated tokens.
- Execution quality is claimed without patch apply and test outcomes.
- Large-repo stress numbers are presented as bug-fix quality results.
- Hypothetical expected reductions are shown as measured results.

## Brutal Current Gaps Before 10/10

These are the remaining non-result gaps to keep visible:

1. Confidence is currently heuristic; it should eventually include candidate margin, language/parser coverage, and whether the answer required fallback.
2. Context insufficiency is not yet a first-class result field; agents infer it from confidence/reason and their own need to expand.
3. Copilot export parsing must be dry-run on one fresh real export before quota runs because VS Code export schemas can drift.
4. Execution-quality automation for SWE-bench patch apply/tests is the biggest missing credibility piece after trace parsing.
5. Cross-language semantic quality is only as good as current parsers; unsupported/generated files need explicit fallback telemetry.
6. Tool schema overhead should be reported separately from code/context output tokens when possible.
7. Competitor comparisons must be framed carefully: CRG/Graphify-style static token reductions are useful but weaker than real-agent paired runs.
