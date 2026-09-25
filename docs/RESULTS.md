# Results

What better code retrieval buys a coding agent, measured with SkeletonGraph as the
retriever. This page gives every result with its uncertainty and its caveats; the
[README](../README.md) gives the short version and [`eval/README.md`](../eval/README.md)
how to reproduce all of it.

All numbers are computed by `python -m eval.scripts.paper_v2_analysis` from the run
records and agent transcripts, and are stored in
[`eval/results_summary.json`](../eval/results_summary.json).

## Design

| setting | model | tasks | compared |
|---|---|---|---|
| Retrieval only | none | 100 SWE-bench Verified | grep, BM25, `sg-rerank`, BM25 + dense, `sg-fusion` |
| ReAct loop | `nemotron-3-super-120b-a12b` | the same 100 | no search, grep, BM25, Graphify, Aider map, `sg-fusion`; an earlier run adds more variants |
| Claude Code 2.1.206–211 | `claude-sonnet-5` | the same 100 | built-in tools vs + SkeletonGraph (shipped integration) |
| Claude Code 2.1.211–214 | `claude-sonnet-5` | 50 SWE-rebench, original and code-stripped issues | built-in vs + SkeletonGraph (shipped) |
| Claude Code 2.1.274 | `claude-sonnet-5` | the same 100 Verified | built-in vs + SkeletonGraph (shipped) |
| Claude Code 2.1.278 | `claude-sonnet-5` | the same 100 | built-in vs + SkeletonGraph (plain) |
| Codex CLI 0.155.0 | `gpt-5.6-terra` | the same 100 | built-in vs + SkeletonGraph, shipped and plain |

Every patch is applied and the project's tests run in Docker; outcomes are never
self-reported. Every comparison is paired by task. In total: 3,433 verified runs.

The **shipped** integration is SkeletonGraph as `sg install` sets it up: the MCP server
plus a rules file, a prompt hint, session hooks, and guidance in tool results. The
**plain** integration is the same server with `SG_MCP_PLAIN=1` and none of that
guidance. Rankings are identical under both.

## 1. Retrieval quality

Without an agent, file level, 100 tasks:

| ranker | MRR | recall@5 | recall@10 |
|---|--:|--:|--:|
| grep | 0.159 | 0.236 | 0.348 |
| BM25 | 0.482 | 0.626 | 0.719 |
| `sg-rerank` | 0.518 | 0.701 | 0.824 |
| BM25 + dense | 0.551 | 0.714 | 0.843 |
| **`sg-fusion`** | **0.658** | **0.785** | **0.856** |

The structural signal mainly improves early ranking: against BM25 + dense it adds
+0.106 MRR (paired 95% CI +0.064 to +0.149) and +0.071 recall@5 (+0.017 to +0.124), but
only +0.013 recall@10 (−0.023 to +0.048). `sg-rerank` alone improves on BM25 by less,
with an interval that includes zero.

Inside the agents, on the tasks where the agent called SkeletonGraph (the built-in arm
recomputed on the same tasks):

| setting | called on | first search hit a gold file | gold file ranked first |
|---|--:|--:|--:|
| Claude Code 2.1.206–211 | 97/100 | 75% → **94%** | 65% → **75%** |
| Claude Code 2.1.274 | 94/100 | 73% → **90%** | 63% → **76%** |
| Claude Code 2.1.278 | 96/100 | 93% → **96%** | 88% → **93%** |
| Claude Code, SWE-rebench | 46/49 | 70% → **87%** | 46% → **54%** |
| Claude Code, SWE-rebench, code-stripped | 46/50 | 70% → **87%** | 59% → 59% |
| Codex CLI (plain) | 93/100 | 60% → **89%** | 32% → **71%** |

Removing code blocks and tracebacks from the issue text did not detectably change
SkeletonGraph's first-search recall (−0.008, 95% CI −0.067 to +0.049, 93 tasks).

## 2. The funnel: how far the gain travels

Share of tasks reaching each stage, built-in → with SkeletonGraph, all paired tasks.
A dagger marks a stage whose paired difference is significant (exact McNemar, p < 0.05).
*Differ*: tasks whose verified outcome differs between arms; *file* / *fn*: of those, tasks
where both arms had edited a gold file / a gold function.

| setting | n | first search hit | read gold code | edited gold file | edited gold function | made a patch | solved | differ / file / fn |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ReAct loop | 100 | 0 → 80† | 88 → 97† | 65 → 87† | 44 → 55† | 74 → 90† | 35 → 42 | 17 / 10 / 8 |
| ReAct loop, earlier run | 100 | 0 → 76† | 95 → 97 | 80 → 82 | 52 → 56 | 86 → 88 | 44 → 45 | 13 / 7 / 5 |
| Claude Code 2.1.206–211 | 100 | 73 → 91† | 99 → 99 | 96 → 96 | 69 → 66 | 100 → 100 | 74 → 75 | 13 / 11 / 10 |
| Claude Code 2.1.274 | 100 | 69 → 85† | 100 → 100 | 97 → 96 | 69 → 65 | 100 → 100 | 79 → 69† | 12 / 10 / 7 |
| Claude Code 2.1.278 | 100 | 90 → 92 | 98 → 98 | 97 → 97 | 68 → 69 | 100 → 100 | 71 → 73 | 10 / 9 / 6 |
| Claude Code, SWE-rebench | 49 | 67 → 82 | 96 → 98 | 90 → 92 | 63 → 63 | 98 → 98 | 55 → 53 | 11 / 10 / 6 |
| Claude Code, SWE-rebench, code-stripped | 50 | 66 → 80 | 98 → 96 | 88 → 88 | 66 → 66 | 100 → 98 | 50 → 44 | 7 / 7 / 3 |
| Codex CLI (plain) | 100 | 61 → 88† | 97 → 96 | 94 → 95 | 63 → 68 | 99 → 100 | 53 → 59 | 16 / 14 / 10 |

*Read gold code* requires code from a gold file to enter the agent's context (a file read, a
content search, a command that printed it, or a retriever payload); a path-only listing does
not count. *Edited gold function* maps each changed line onto the functions of the gold file at
the base commit.

The production agents read and edited the right file on nearly every task with or without
SkeletonGraph, and no stage after the first differs significantly in any production setting.
They also got there fast: on SWE-bench Verified the built-in agents read gold code with their
**first tool call** on 64–97% of tasks, and within three calls on 93–98%. The exception is
SWE-rebench, whose repositories postdate the models: there the built-in agent read gold code on
its first call on only 30–37% of tasks, and SkeletonGraph reached it sooner.

Function-level localization is far from saturated (63–69%), but SkeletonGraph, which returns
functions, did not move it in any production agent. Across the six production-agent
comparisons, outcomes differ on 69 tasks; on 61 both arms had already edited a gold file, and on
42 of the 68 with a resolvable gold function both had edited a gold function.

Only the ReAct loop, whose search-free arm often failed to reach the right file, turned the gain
into more edits of the right file (65% → 87%) and function (44% → 55%). Splitting its tasks by
which arms edited a gold file, the net solve change was +4 where only SkeletonGraph did, +6 where
both did, −2 where only the baseline did, and −1 where neither did: most of the outcome
difference arose where both arms were already in the right file.

The stricter measures are computed by `python -m eval.scripts.funnel_strict`.

## 3. Cost

Input tokens per task, built-in → with SkeletonGraph, paired; 95% CI from a paired
bootstrap (10,000 resamples). Dollars use each run's own price sheet.

| setting | n | tokens on its own | change | 95% CI | turns | $/task | p50 | p90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Claude Code 2.1.278 | 100 | 112,813 | +66.0k (+58.5%) | +49 to +68% | +2.5 | +$0.020 | +59% | +44% |
| Codex CLI (plain) | 100 | 134,955 | +7.7k (+5.7%) | −7 to +21% | −0.3 | −$0.004 | +27% | −10% |
| Codex CLI (shipped) | 100 | 134,955 | +53.5k (+39.6%) | +17 to +66% | +2.1 | +$0.023 | +72% | +4% |
| ReAct loop, earlier run | 100 | 244,123 | −69.3k (−28.4%) | −39 to −15% | −1.5 | −$0.019 | −29% | −24% |
| ReAct loop | 100 | 344,642 | −164.6k (−47.8%) | −59 to −34% | −1.7 | −$0.014 | −45% | −59% |
| Claude Code 2.1.274 | 100 | 530,478 | −93.3k (−17.6%) | −31 to −0% | −1.9 | −$0.026 | +10% | −36% |
| Claude Code 2.1.206–211 | 100 | 644,504 | −152.3k (−23.6%) | −35 to −8% | −3.1 | −$0.063 | +3% | −38% |
| Claude Code, SWE-rebench | 49 | 1,048,603 | −255.6k (−24.4%) | −39 to −5% | −4.7 | −$0.117 | −7% | −20% |
| Claude Code, SWE-rebench, code-stripped | 50 | 1,119,864 | −228.3k (−20.4%) | −33 to −6% | −4.0 | −$0.112 | −33% | −19% |

Where the agent spends a lot on its own, SkeletonGraph replaces searching, reading and
re-checking and saves; where the agent already finds the code in one or two calls, it is
added on top and costs a little more. In Claude Code 2.1.206–211 it cut reads from 3.8 to
1.4 per task, text searches from 3.2 to 1.4, and code runs from 2.7 to 0.5. Under Claude
Code 2.1.278 the agent kept its own reads and searches (1.3 and 1.5 per task, against 1.1
and 1.7 without SkeletonGraph) and added a tool-lookup call and 1.3 retriever calls; all
the extra turns came before the first edit (5.0 against 2.8), none after it.

Within a setting the same logic shapes the distribution: in Claude Code 2.1.206–211 the
median task cost about the same while the 90th percentile of billed cost fell 25.6% (95% CI
−51.0% to −4.3%), a pattern that held when tasks were ordered by SkeletonGraph's own cost
and that replicated under 2.1.274.

Where the saving comes from, splitting each run at its first edit (tool-using turns;
tasks that ran code or tests after the first edit):

| setting | before first edit | after first edit | ran code after editing |
|---|--:|--:|--:|
| Claude Code 2.1.206–211 | 6.7 → 6.6 | 6.8 → 3.8 | 49 → 11 |
| Claude Code 2.1.274 | 5.8 → 5.8 | 5.1 → 3.2 | 40 → 5 |
| Claude Code, SWE-rebench | 12.8 → 9.7 | 7.5 → 4.9 | 25 → 7 |
| Claude Code 2.1.278 | 2.8 → 5.0 | 1.9 → 2.0 | 2 → 2 |
| Codex CLI (actions) | 4.5 → 4.4 | 1.3 → 1.1 | 4 → 1 |

In the older Claude Code releases, whether the drop in post-edit checking came from
retrieval or from the shipped integration's instruction not to re-run code cannot be
separated; Codex checked too rarely in any arm to separate it there.

## 4. Agent releases

Claude Code's built-in agent on the same 100 tasks, 2.1.274 → 2.1.278 (three weeks apart;
web tools were denied in the later run):

| per task | 2.1.274 | 2.1.278 |
|---|--:|--:|
| turns with a tool call | 10.9 | 4.7 |
| before / after the first edit | 5.8 / 5.1 | 2.8 / 1.9 |
| file reads (characters returned) | 3.2 (10,135) | 1.1 (1,811) |
| searches (of which list-files) | 3.5 (0.9) | 1.7 (0.0) |
| code or test runs | 1.5 | 0.1 |
| tasks that ran code after editing | 40 | 2 |
| model output tokens | 5,093 | 1,447 |
| input tokens | 530,478 | 112,813 |
| own first search ranks a gold file first | 59 | 85 |
| solved | 79 | 71 (11 lost, 3 gained, p = 0.057) |

Claude Code's release notes for 2.1.275–2.1.278 document no change to how the search or
read tools return results beyond error-handling fixes, and none to default reasoning
effort. They do document the removal of a background-task tool and the loading of
account-synced skills and plugins into terminal sessions. The behaviour change is
measured here, not attributed.

## 5. Integration

On Codex CLI, same release: the shipped integration added 53.5k input tokens per task
(+39.6%, 95% CI +17 to +66%) and the plain one 7.7k (+5.7%, −7 to +21%). The shipped
integration's orientation tool ran on 89 of 100 tasks and added 1.8 actions before the
first edit; the plain one added none. Solved: 53 built-in, 58 shipped, 59 plain; neither
difference is statistically detectable. This is one controlled comparison of one set of
instructions, not a general result about behavioural guidance.

## Caveats

- **Solve-rate comparisons are underpowered.** At 100 tasks only differences of roughly
  ten points are resolvable, and no equivalence is claimed. The one significant
  difference (−10 on Claude Code 2.1.274, p = 0.006) did not appear in the same design
  under 2.1.206–211 (74 vs 75).
- **Settings differ in more than one way.** The ReAct loop, Claude Code and Codex differ
  in model, tools and prompts as well as in how much they explore, so the cross-setting
  cost pattern is an association consistent with the substitution account, not a causal
  estimate.
- **Release and integration are confounded on Claude Code:** the 2.1.274 comparison used
  the shipped integration and the 2.1.278 one the plain integration. The plain one had
  lower overhead on Codex, so the change cannot explain the switch from saving to cost.
- **The 2.1.278 comparison** pairs built-in and SkeletonGraph runs recorded a day apart
  under the same release, in two run tags.
- **Web access.** Runs before September allowed web tools. Excluding every task on which
  either arm fetched a web page changes the 2.1.206–211 token saving from −23.6% to
  −26.2% with solve counts unchanged, and the SWE-rebench saving from −24.4% to −10.1%:
  the direction holds, the SWE-rebench magnitude is sensitive to five tasks.
- **Scope.** Python repositories from two benchmarks, two production agents, one
  open-weight model in a controlled loop.

## Measurement notes

- **Report tokens, not dollars.** Fitting each run's reported cost against its token
  components recovers the provider's price sheet per release (mean absolute residual
  0–5%): cache-read tokens fell from $0.28–0.32 to $0.20–0.21 per million, and output
  tokens from $13.1–15.1 to $9.5–10.0, between releases 2.1.214 and 2.1.274.
- **Record the agent's version.** Context per turn was 35–45k tokens in every Claude Code
  setting from 2.1.206 to 2.1.274 and 18–23k under 2.1.278.
- **First-search hit is not a proxy for success.** Within an arm, runs whose first search
  hit a gold file often passed *less* often (Claude Code built-in, 2.1.206–211: 69.9% vs
  85.2%; Codex built-in: 39.3% vs 74.4%), because agents that already know where to look
  often open the file directly without searching.
- **Token fields differ by harness.** Claude Code and Codex report fresh and cached input
  separately; the ReAct loop's prompt-token count already includes cached tokens.
- **The ReAct `none` arm is not closed-book:** it has no search tool but can list and read
  files.
