# SkeletonGraph Blueprint

Status: canonical project handoff and implementation blueprint.
Last updated: 2026-05-13.
Authorial stance: this is the project as I would explain it to a new engineer
or a new model before asking them to implement anything.

This file is intentionally detailed. It is not marketing copy and it is not a
short README. It exists so the next model can understand why each piece exists,
what it depends on, what is implemented, what is pending, and how to evaluate
whether SkeletonGraph is actually useful.

## 1. One-Sentence Summary

SkeletonGraph is a wrapper-first context engine for coding agents that builds
an AST- and summary-backed code graph, uses a cheap retrieval planner to pick
targets, assembles a compact context packet, and optionally routes CLI
execution to the cheapest sufficient model tier.

## 2. The Core Idea

Modern AI coding tools are good at reasoning after they have the right files.
They are often expensive and noisy while discovering those files.

The common native agent loop looks like this:

```text
user asks a task
  -> model searches
  -> model reads a file
  -> model reads another file
  -> model reads tests
  -> model notices a caller
  -> model rereads context in later turns
  -> model finally patches
```

Every model turn re-ingests some combination of conversation history, tool
results, instructions, file contents, and schema overhead. Even if grep or file
read is computationally cheap, it is not token-free once it becomes part of the
next model input.

SkeletonGraph moves the repo-discovery step into deterministic code (with an
optional cheap retrieval planner on top of the index):

```text
user prompt
  -> (optional) small-model retrieval planner over AST/summaries/index
  -> classify task shape
  -> resolve likely target functions/files
  -> walk dependency/test/blast-radius graph
  -> assemble a bounded context packet
  -> expose route, confidence, and reasons
  -> IDE or CLI model executes with less exploration
```

The point is not only "save tokens." The real goal is:

- better first useful response
- fewer exploratory reads
- fewer missed callers/tests
- cheaper model choice when possible
- repeatable routing and packet logs
- workflows that work in IDEs and terminals

## 2.1 Planned Concept Shifts (2026)

These are plan-level changes (product concept), not just implementation tasks:

- Wrapper-first remains the main path: SG supplies context/routing, while edits
  and apply are owned by the IDE/CLI agent the user already trusts.
- IDE uses retrieval tools first: the agent can ask for a retrieval index and
  pick targets, or directly request a packet. SG does not depend on mid-chat
  model switching in IDEs.
- CLI becomes planner-first: a small model can produce a retrieval plan from
  the function index, and SG assembles a deterministic packet before any heavy
  model execution.
- Docstrings/comments are the primary summaries. Stored summaries are fallback
  only when docstrings/comments are missing or stale.
- Error-only follow-up packets allow the next run to carry only the failure
  signals instead of re-sending the full context packet.
- Auto rebuild after completion (IDE report_completion or CLI run) keeps the
  index and summaries fresh without manual rebuild calls.
- No BM25 default. Optional keyword fallback can exist, but it is not a core
  path and should stay off unless explicitly enabled.

## 3. What SkeletonGraph Is and Is Not

SkeletonGraph is:

- a static/deterministic codebase indexer
- a function-level graph and retrieval engine
- a context packet assembler
- an MCP server for IDE agents
- a CLI context workbench
- a model-tier router for CLI execution
- an evaluation harness for retrieval, cost, and compliance

SkeletonGraph is not, at least for the current release:

- a full autonomous agent competing with Claude Code, Codex, OpenCode, or Aider
- a TUI-first programming environment
- a generic provider platform
- a tool that should make mandatory LLM calls during retrieval
- a system that relies on IDEs switching model mid-conversation

The correct mental model is:

```text
SG IDE = context infrastructure for the agent the user already pays for
SG CLI = route + prepare + optional execute + verify + logs
```

A second, equally valid mental model is:

```text
SG IDE = either a full packet OR a retrieval index the IDE agent uses to pick
         targets (AST/summaries/graph/memory exposed via tools)
SG CLI = cheap retrieval planning + deterministic assembly + heavy model
```

## 4. Product Surfaces

## 4.1 SG IDE

SG IDE is the low-friction surface.

The user already has Cursor, Claude Code, Copilot, Codex, Antigravity,
Windsurf, or another MCP-capable coding environment. They do not want to paste
API keys into another tool just to get repo context. They want their existing
agent to stop wasting turns.

The SG IDE promise:

```text
Keep your IDE. SG gives its agent the right repo context earlier.
```

The ideal SG IDE workflow:

```text
sg init --agent cursor
sg build
IDE starts MCP server
agent sees compact SG instructions
agent calls SG before broad exploration
SG returns either:
  - a target/test/blast-radius packet (query_context), or
  - a retrieval index (get_retrieval_context) so the IDE agent can choose
    targets using AST/summaries/graph/memory signals
agent edits normally
agent reports completion or hook records it
SG updates memory
```

### 4.1.1 IDE Core Pipeline (planned)

In-depth flow (what is stored, retrieved, and passed):

```text
build:
  sg build
    -> parse AST/skeletons
    -> graph edges
    -> summaries (docstrings first)
    -> hashes + staleness
    -> store in .skeletongraph/*

retrieve:
  IDE agent -> get_retrieval_context(prompt)
    -> project_summary
    -> file_map
    -> function_index (docstring-first summaries)
    -> session_context
  IDE agent chooses targets (or skips and calls query_context)

assemble:
  IDE agent -> query_context(prompt or entities)
    -> SG resolves targets
    -> graph expansion (callers/callees/tests/blast radius)
    -> attach diagnostics + memory
    -> assemble packet + metadata

execute:
  IDE agent edits code
  if docstring summary is wrong for a modified function, update the docstring
  IDE agent -> report_completion(files_modified)
    -> SG updates index + summaries + session memory
```

Stored artifacts used by IDE:
- skeletons, graph, hashes, summaries, pagerank (when present)
- session memory and diagnostics
- run/query logs

Data passed to the IDE agent:
- retrieval index (project_summary, file_map, function_index, session_context)
- context packet text + metadata (mode, confidence, route, reasons)

The important product constraint:

Most IDEs do not let SG change models dynamically in the middle of a
conversation. Cursor, Copilot, Codex App, and many other environments put model
selection in the user/IDE layer. Claude Code has `/model`, but SG still should
not depend on Claude-only behavior.

Therefore SG IDE should treat model tiers as advisory metadata:

- SLM: good for lookup, docs, summaries, simple repo questions
- MLM: good for normal edits, tests, reviews, targeted debugging
- LLM: good for architecture, broad migrations, low-confidence tasks

SG IDE must win even when the active model does not change. That means packet
quality, MCP discipline, and small rules matter more than runtime model
switching.

SG IDE can also win by making retrieval cheaper for the agent: it can expose
AST skeletons, summaries, graph neighbors, and session memory so the IDE agent
can decide what to open without broad search.

## 4.2 SG CLI

SG CLI is the explicit model-routing surface.

The CLI user may want to use a paid provider, a local Ollama model, another
terminal agent, or a web UI. SG CLI should give them:

- route visibility
- context packet export
- dry-run planning
- optional provider execution
- run logs
- later: patch apply and verification

The SG CLI promise:

```text
Route the task, pack the repo context, use a cheap retrieval planner plus
deterministic assembly, call the cheapest sufficient heavy model, and record
what happened.
```

The intended CLI workflow:

```text
sg build
sg doctor
sg route "fix auth token validation"
sg prepare "fix auth token validation" --out .skeletongraph/context.md
sg run "fix auth token validation" --dry-run
sg run "fix auth token validation" --execute
sg verify
sg runs
```

### 4.2.1 CLI Core Pipeline (planned)

In-depth flow (what is stored, retrieved, and passed):

```text
build:
  sg build
    -> parse AST/skeletons
    -> graph edges
    -> summaries (docstrings first)
    -> hashes + staleness
    -> store in .skeletongraph/*

plan (turn 1):
  sg run --plan-first (or sg prepare/route)
    -> optional small-model planner reads function_index + file_map
    -> planner proposes targets + constraints + warnings

assemble (turn 2):
  SG assembles packet deterministically from plan
    -> graph expansion (callers/callees/tests/blast radius)
    -> attach diagnostics + memory
    -> packet + route + confidence

execute:
  sg run --execute
    -> provider call with packet
    -> if error, store error-only followup in session
    -> record run log

refresh:
  after completion, SG updates index + summaries + session memory
```

Stored artifacts used by CLI:
- same core index as IDE
- run logs and query logs
- error-followup cache in session

Data passed to the CLI user or provider:
- plan output (targets, filters, rationale)
- context packet text + metadata (mode, confidence, route, reasons)
- execution output and diagnostics

Current implementation has the first six pieces partly working. `sg verify`,
`sg runs`, and safe patch application are still planned.

## 5. Competitive Context and Wrapper Strategy

This section exists to keep SG from accidentally building the wrong product.
The current conclusion is wrapper-first, standalone-second.

SG should not spend its next major effort building a full editing agent from
scratch. Aider, Claude Code, Codex, Gemini CLI, and OpenCode already have years
of execution work: editing formats, approvals, shell/tool loops, git behavior,
undo, tests, LSP, extensions, and user trust. SG's strongest path is to become
the context/routing layer that makes those tools cheaper and less lost.

The near-term SG CLI should be:

```text
context engine + model router + adapter/orchestrator
```

not:

```text
full standalone terminal coding agent
```

Standalone `sg run --execute` can remain useful for experiments, local model
testing, and future automation, but it should not be the primary adoption path
until safety, apply, verify, and eval are mature.

### 5.1 Aider

Aider is excellent at git-native terminal pair programming. It has a repo map,
model config, auto-commits, `/diff`, `/undo`, `/commit`, and many editing
workflows.

SG should not try to beat Aider at git-native editing first. SG should produce
better graph-derived context that can be used by Aider:

```bash
sg prepare "fix auth token expiry" --out .skeletongraph/context.md
aider --read .skeletongraph/context.md src/auth/handler.py
```

What Aider is better at:

- applying edits safely through its mature edit formats
- git-native workflow, commits, undo, diff, lint, and test commands
- broad provider support and existing CLI adoption
- repo-map baseline that already works in real projects

What SG can be better at:

- function-level graph packets instead of a broad ranked map
- explicit target/caller/callee/test packet construction
- blast-radius and session-memory context
- route recommendation before choosing a model
- packet completeness and retrieval eval

Best integration:

```bash
sg route "fix auth token expiry"
sg prepare "fix auth token expiry" --for aider --out .skeletongraph/aider-context.md
aider --read .skeletongraph/aider-context.md <target-files>
```

Future adapter:

```bash
sg exec --backend aider "fix auth token expiry"
```

`sg exec --backend aider` should prepare context, identify likely edit files,
invoke Aider with read-only context and target files, then record the result in
SG run logs. SG should let Aider own patch application and git semantics.

### 5.2 Claude Code

Claude Code has deep terminal-agent behavior: hooks, subagents, skills, MCP,
and lifecycle integration.

SG should integrate deeply where possible:

- `CLAUDE.md` rules
- MCP tools
- prompt-submit hook that prepares context
- post-tool hook that records changed files/diagnostics
- stop/session hook that compresses memory

But SG must not depend on Claude-only hooks.

What Claude Code is better at:

- agentic terminal loop
- hook lifecycle around prompts, tool use, stop, compaction, and sessions
- subagents and skills
- permissioned shell/file execution

What SG can be better at:

- prompt-time graph packet injection
- stable project index independent of conversation state
- run/eval logs across tools
- lower repeated file-read pressure

Best integration:

- `CLAUDE.md` tells Claude when to call SG.
- MCP exposes compact SG tools.
- `UserPromptSubmit` hook can inject SG context for coding prompts.
- `PostToolUse` hook records changed files and diagnostics.
- `Stop`/`SessionEnd` hook compresses memory.

Claude Code is probably the strongest deep-integration target because its hook
system can inject context before a prompt is processed. This makes SG feel like
native context infrastructure rather than an extra command users must remember.

### 5.3 Codex

Codex has local workspace access, MCP support, approval modes, and strong
execution workflows.

SG should support:

- `AGENTS.md` instructions
- compact MCP profile
- `sg prepare --copy` fallback
- CLI packet generation for tasks where MCP behavior is uncertain

What Codex is better at:

- local coding agent loop
- approval/sandbox workflow
- applying patches and running validation in a controlled workspace
- multimodal/app workflows where available

What SG can be better at:

- precomputed graph context
- deterministic route and packet logs
- cross-tool memory and eval

Best integration:

```bash
sg prepare "task" --for codex --out .skeletongraph/codex-context.md
codex "Use .skeletongraph/codex-context.md and implement the task"
```

MCP and `AGENTS.md` remain the IDE/app path. CLI wrapper support should avoid
fighting Codex's approval model; SG should provide context and let Codex decide
edits through its own safety layer.

### 5.4 Copilot and Cursor

These are distribution-heavy IDE surfaces. They are attractive because users
already live there, but they are less controllable than a dedicated CLI.

SG should optimize:

- one-command setup
- short rules
- low-friction MCP
- stale-index warnings
- manual fallback

What Cursor/Copilot are better at:

- adoption and daily IDE habit
- inline editing UX
- native code review/chat surfaces
- user subscription billing

What SG can be better at:

- enforcing a compact, graph-aware first packet
- explaining why files were included
- stale-index and packet-completeness diagnostics
- manual packet fallback when IDE agents ignore tools

SG IDE should stay no-key and low-friction here. Users will not tolerate a
second provider setup just to improve a tool they already pay for.

### 5.5 OpenCode and Gemini CLI

These tools are full agent surfaces with provider/model flexibility. SG should
not race them on UI. SG should become a context provider and eval-backed
retrieval layer that can feed them.

OpenCode is better at:

- open-source terminal/desktop/IDE agent UX
- primary agents, subagents, permissions, LSP integration
- many providers and model flexibility
- multi-session workflows

SG can be better at:

- graph packet precision
- route/cost eval
- cross-agent context artifacts

Best OpenCode integration:

- MCP server with a compact tool profile.
- Optional `opencode.json` agent config that uses SG context.
- `sg prepare --for opencode` context file.
- Later: OpenCode plugin/extension that exposes `sg route`, `sg prepare`, and
  `sg metrics`.

Gemini CLI is better at:

- free/accessible Gemini path for many users
- large-context Gemini workflows
- extensions, MCP servers, custom commands, checkpointing, and headless mode
- Google ecosystem integration

SG can be better at:

- smaller targeted packets even when large context is available
- graph/test/blast-radius precision
- model/tool-agnostic eval records

Best Gemini integration:

- Gemini extension with SG MCP server and context file.
- `GEMINI.md` instructions.
- `sg prepare --for gemini`.
- Non-interactive `gemini -p` wrapper only after context injection is reliable.

### 5.6 Sourcegraph/Cody, Continue, and Code Search Tools

Sourcegraph/Cody-style systems are closest to SG at the company/product level:
code search, embeddings, repo intelligence, and IDE context. They are stronger
at enterprise-scale indexing, semantic search, hosted workflows, and broad
language support.

SG can still be differentiated if it is:

- local-first
- lightweight
- function/edge/test/blast-radius oriented
- easy to plug into many agents
- eval-transparent
- not tied to a hosted code intelligence platform

Continue-like tools are strong as IDE/open-source assistant frameworks with
embeddings and model provider configuration. SG should integrate rather than
replace them: generate graph packets, expose MCP, and let the assistant do its
normal edit UX.

### 5.7 Parser and Retrieval Infrastructure

Tree-sitter should be treated as the default parser strategy for polyglot AST
support where possible. It is widely used, incremental-friendly, and avoids SG
owning custom parsers for every language.

Language-specific tools can be plugged in where they provide better semantics:

- Jedi/rope for Python references/refactors
- TypeScript language server for TS/JS symbol/reference precision
- LSP diagnostics for post-edit feedback

Retrieval should not be graph-only. The target direction is hybrid:

```text
graph traversal + planner over function index + session memory
optional keyword fallback (off by default)
optional embeddings after eval data exists
```

Claude's point about vague prompts is correct. "Users get logged out" may not
lexically match `validate_token` or `refresh_session`. Query-time embedding over
summaries/body snippets should be added after a small retrieval eval dataset
exists, so improvements are measurable.

### 5.8 Competitive Position Summary

| Tool | Better than SG at | SG should beat it at | Integration priority |
| --- | --- | --- | --- |
| Aider | git editing, apply, undo, tests | graph packet precision, test/blast radius | highest CLI backend |
| Claude Code | hookable agent loop | prompt-time graph context | highest deep hook target |
| Codex | approval/sandbox execution | deterministic repo packet | high |
| Gemini CLI | extension ecosystem, free/large context | small graph packets | high |
| OpenCode | open agent UX, LSP, providers | retrieval/eval layer | high |
| Cursor/Copilot | IDE adoption | no-key context quality | high IDE |
| Sourcegraph/Cody | enterprise code intelligence | local graph packets | medium |
| Continue | IDE assistant framework | graph retrieval plugin | medium |

Strategic conclusion:

SG should win as the reusable context/routing layer, not as yet another agent
surface. If SG context becomes the thing Aider, Claude Code, Codex, Gemini CLI,
OpenCode, Cursor, and Copilot can all consume, the project has much stronger
open-source credibility and industry relevance.

## 6. Why Graph Context Matters

Plain lexical search finds words. Coding tasks often need structure:

- a function and its callers
- a handler and its route registration
- a validation function and its tests
- a class method and its overridden implementations
- a changed file and affected neighbors
- a public API and downstream consumers

The dependency graph turns a prompt into a structured packet:

```text
targets
  + callers
  + callees
  + imports
  + tests
  + changed files
  + constraints
  + memory
```

This is why SG is not just a grep wrapper. The graph lets SG include code that
does not share obvious keywords with the prompt but is behaviorally relevant.

## 7. Repository Architecture

High-level layout:

```text
src/skeletongraph/
  parser/       AST extraction and source-file parsing
  graph/        dependency graph, ranking, embeddings
  storage/      .skeletongraph persistence
  retrieval/    classification, resolution, model routing
  assembly/     prompt/context assembly
  session/      session memory and dedup
  server/       MCP server
  llm/          LiteLLM wrapper for optional CLI execution
  cli/          Click command surface
  engine.py     main query orchestrator
```

The intended architecture is one engine:

```python
engine = SGEngine(project_root)
result = engine.query(prompt, delivery="cli")
```

Everything should eventually call that engine:

- MCP `query_context`
- CLI `sg query`
- CLI `sg prepare`
- CLI `sg route`
- CLI `sg run`
- eval scripts
- hooks

Avoid duplicating retrieval/assembly paths. Duplicate paths caused earlier v4
confusion.

## 8. Build-Time Pipeline

Build time is deterministic and should not require provider API calls.

Pipeline:

```text
discover files
  -> parse source files
  -> extract functions/classes/symbols
  -> extract imports/calls/edges
  -> build lightweight keyword index (optional fallback)
  -> compute graph metadata
  -> persist .skeletongraph artifacts
```

Expected artifacts:

```text
.skeletongraph/
  meta.json
  skeletons.json
  graph.json
  index.json
  hashes.json
  bloom.bin
  summaries.json
  pagerank.json              planned/partial
  vocabulary.json            planned
  learned_edges.json         planned
  project.md                 planned/user-editable
  architecture.md            planned/generated or user-editable
  constraints.md             planned/user-editable
  session/
    current.md
    recent.md
    project_log.md
    dedup.json               planned
    diagnostics.json         planned
  eval/
    query_log.jsonl
    hit_log.jsonl
  runs/
    run_log.jsonl
```

Why each artifact exists:

- `skeletons.json`: compact symbol inventory; basis for target selection.
- `graph.json`: dependencies and traversal; basis for blast radius.
- `index.json`: lexical retrieval; cheap first-pass candidate search.
- `hashes.json`: stale detection and incremental rebuild.
- `bloom.bin`: quick membership/changed checks.
- `summaries.json`: optional semantic summaries; useful but not correctness
  critical. Use docstrings/comments first; fallback to summaries when missing,
  then refresh incrementally.
- `pagerank.json`: structural importance; helps when many candidates match.
- `vocabulary.json`: project-specific terms and noise words.
- `learned_edges.json`: memory of files/functions changed together.
- `session/*`: temporal memory, dedup, diagnostics.
- `eval/*`: retrieval and token measurement.
- `runs/*`: CLI execution records.

## 9. Query-Time Pipeline

The ideal query path:

```text
input prompt
  -> (optional) small-model retrieval plan over AST/summaries/index
  -> classify task mode/shape
  -> find seed candidates
  -> score candidates
  -> expand through graph
  -> attach tests/diagnostics/memory
  -> assemble packet under budget
  -> calculate confidence and route
  -> return result object
```

### 9.1 IDE Query Flow (planned)

```text
IDE prompt
  -> (optional) get_retrieval_context
     returns: function_index + file_map + session_context
  -> IDE agent selects targets
  -> query_context(prompt or entities)
  -> SG resolves targets and expands graph
  -> SG assembles packet + route + confidence
  -> IDE agent edits
  -> report_completion triggers rebuild + summary refresh
```

### 9.2 CLI Query Flow (planned)

```text
CLI prompt
  -> optional small-model plan over function_index
  -> deterministic assembly + graph expansion
  -> packet + route + confidence
  -> execute or export
  -> error-only followup stored on failure
  -> auto rebuild after completion
```

Current query path is partly legacy but usable. It has:

- mode classification
- candidate retrieval
- graph expansion
- prompt assembly
- confidence
- delivery-aware recommended model
- deterministic model routing

The v5 destination is a cleaner packet object with explicit components instead
of only a text blob.

## 10. Task Modes and Packet Shapes

Older docs had many modes and some were too model-oriented. The useful v5
packet shapes are:

| Shape | What it means | Default tier |
| --- | --- | --- |
| `targeted_fix` | known bug/function/file | MLM |
| `investigation` | uncertain bug, needs exploration | MLM or LLM |
| `feature_slice` | bounded feature across a few files | MLM |
| `review` | changed files and blast radius | MLM |
| `test_generation` | tests around existing code | MLM |
| `explain` | explain code/repo behavior | SLM or MLM |
| `architecture` | design, migration, broad reasoning | LLM |
| `docs` | docs/readme/comments | SLM |

Why modes exist:

- select packet shape
- select graph expansion depth
- decide whether tests matter
- decide whether callers/callees matter
- decide model tier
- set confidence expectations

Modes should not become a maze. If a mode does not change retrieval, assembly,
or routing behavior, remove or merge it.

## 11. Context Packet Contract

The packet should eventually include both text and metadata.

Text sections:

```text
1. Header
   prompt, mode, confidence, token budget, route

2. Project constraints
   user rules, architecture notes, coding constraints

3. Targets
   full bodies or focused snippets of functions/files to edit

4. Surrounding graph
   callers, callees, imports, types, route registration

5. Tests and diagnostics
   related tests, failing output, lint/type errors

6. Memory
   recent decisions, files changed together, previous attempts

7. Expansion suggestions
   what to request next if the model truly needs more

8. Task recap
   short mission statement at bottom for recency
```

Metadata fields:

- `packet_id`
- `prompt`
- `mode`
- `confidence`
- `confidence_reason`
- `targets`
- `components`
- `token_count`
- `route`
- `recommended_model`
- `expansion_suggestions`
- `included_files`
- `omitted_candidates`
- `reasons`

Why metadata matters:

- CLI can show route and reasons.
- Eval can measure packet completeness.
- IDE can decide when to ask for expansion.
- Run logs can compare cost/results later.

## 12. Token Budget Philosophy

SG should not maximize context. It should maximize useful context per token.

Normal targeted tasks should aim for roughly:

```text
1.5k-6k tokens
```

Architecture or migration tasks may need more, but if every task creates a
20k-token packet, SG has failed its own product thesis.

Important rules:

- include complete target bodies when small
- use focused extraction for very large functions
- include tests only when likely relevant
- include callers/callees with caps
- include diagnostics only if small and recent
- prefer one complete first packet over many follow-up tools
- treat expansion calls as a failure signal when they become common

## 13. Confidence Model

Confidence should tell the model and user how much to trust the packet.

High confidence:

- target is clear
- lexical/graph signals agree
- packet has obvious target and tests
- small candidate set

Medium confidence:

- plausible target but multiple candidates
- tests or callers uncertain
- prompt is broad but bounded

Low confidence:

- prompt vague
- many candidates
- no strong target
- packet may need expansion or native exploration

Low confidence is not a failure. Pretending high confidence is the failure.

Low-confidence packets should:

- say why confidence is low
- include top candidates
- suggest expansion
- route CLI to MLM/LLM depending on task risk
- avoid overclaiming completeness

## 14. Model Tiers

SG uses tier names because both IDE and CLI need a simple abstraction.

| Tier | Meaning | Good for | Risk |
| --- | --- | --- | --- |
| SLM | small/cheap/fast model | docs, lookup, summarization | unsafe for nontrivial edits |
| MLM | default coding model | targeted edits, tests, review | may struggle with broad architecture |
| LLM | strongest model | architecture, migrations, low-confidence tasks | expensive/overkill |

The key rule:

Code-changing tasks should not be routed below MLM unless explicitly allowed.

Cost optimization should come from:

- smaller context
- fewer turns
- fewer file reads
- avoiding LLM overkill
- using local/cheap models for safe tasks

It should not come from asking weak models to do dangerous edits.

## 15. SG IDE Model Behavior

IDE model names are labels only. SG stores and displays them, but it does not
control them.

Examples:

- Cursor: user manually chooses model in UI.
- Copilot: user manually chooses model/agent behavior.
- Codex App: model choice is outside SG.
- Claude Code: `/model` exists, but SG should not depend on it.

SG IDE should output:

- recommended tier
- short explanation
- confidence
- packet (or retrieval index for agent-driven target selection)

SG IDE should not require:

- provider API key
- extra billing
- SG-side model execution

## 16. SG CLI Model Behavior

CLI can actually route to provider model names because SG owns the command.

Current provider presets:

- `anthropic`
- `openai`
- `google`
- `local`

Current behavior:

- `sg route` shows recommended model.
- `sg run --dry-run` shows selected model and packet plan.
- `sg run --execute` calls the provider through LiteLLM.
- `--tier slm|mlm|llm` overrides routing.
- `--auto-model` is the default dynamic route behavior.
- API keys are never stored.
- local provider uses an Ollama/LiteLLM-compatible base URL.

Retrieval planning uses a small model when enabled; it reads the AST-based
function index and summaries to propose targets, then SG assembles the packet
deterministically before the heavy model executes.

No API key is needed for:

- `sg build`
- `sg doctor`
- `sg route`
- `sg prepare`
- `sg query`
- `sg run --dry-run`

API key or local endpoint is needed for:

- `sg run --execute`

Local model testing:

```bash
ollama pull qwen3-coder:latest
ollama serve
sg config --cli-provider local
sg run "fix validate_token" --path tests/fixtures/python_small --dry-run
sg run "fix validate_token" --path tests/fixtures/python_small --execute
```

Local models are good for cheap pipeline testing. They are not automatically
the quality benchmark unless the benchmark is specifically local-model quality.

## 17. Smart Routing

Smart routing is a good concept and should remain in SG CLI.

Inputs:

- query mode
- confidence
- context token count
- number of candidates
- complexity score
- whether the task changes code
- whether prompt implies architecture/migration

Outputs:

- base tier
- selected tier
- recommended model
- reason string

Current policy:

- docs/simple explain can route SLM
- targeted code edit routes MLM
- broad architecture routes LLM
- low confidence can upgrade
- code-changing tasks keep an MLM floor

Why deterministic routing first:

- transparent
- testable
- cheap
- no circular dependency on an LLM just to pick an LLM

Later improvement:

- train/evaluate thresholds from logs
- allow project-level overrides
- add route confidence
- add "manual override used" metric

## 18. CLI Commands, Wrappers, and Their Roles

Current or intended commands:

| Command | Role | Current state |
| --- | --- | --- |
| `sg init` | setup project/IDE/MCP/rules | exists |
| `sg build` | build index | exists |
| `sg status` | show index state | exists |
| `sg doctor` | readiness checks | added/improved |
| `sg config` | IDE/CLI model/provider config | added/improved |
| `sg route` | show deterministic route | added |
| `sg prepare` | output/copy context packet | added/improved |
| `sg query` | debug query and packet | exists/improved |
| `sg run --dry-run` | model-routed execution plan | added |
| `sg run --execute` | call provider and write output | added, experimental |
| `sg exec --backend aider` | prepare context and delegate editing to Aider | planned |
| `sg exec --backend claude-code` | prepare context and delegate to Claude Code | planned |
| `sg exec --backend codex` | prepare context and delegate to Codex | planned |
| `sg exec --backend gemini` | prepare context and delegate to Gemini CLI | planned |
| `sg exec --backend opencode` | prepare context and delegate to OpenCode | planned |
| `sg exec --backend continue` | prepare context for Continue-compatible workflows | planned |
| `sg exec --backend generic` | write context and command hints for any CLI/web UI | planned |
| `sg prepare --for <tool>` | backend-specific context export | partial/planned |
| `sg verify` | run tests/diagnostics | planned |
| `sg runs` | show run history/costs | planned |
| `sg metrics` | evaluation metrics | exists/partial |
| `sg serve` | MCP server | exists |

`sg run` should not silently modify files. There are two possible execution
lanes:

1. Wrapper lane:
   SG prepares context and delegates editing/apply/git behavior to an existing
   mature tool such as Aider, Claude Code, Codex, Gemini CLI, or OpenCode.

2. Native lane:
   SG directly calls a provider/local model and writes output. Native apply
   remains disabled until SG has safe patch parsing, approvals, and verify.

The wrapper lane should be the main near-term product bet. It lets SG keep its
unique value in context/routing while avoiding months of reimplementing editor
agent safety features.

The safe future flow:

```text
sg exec --backend <backend> "task"
  -> route task
  -> prepare backend-specific context
  -> choose target files
  -> call backend with context and route hint
  -> backend applies edits through its own safety model
  -> SG records route, packet, backend, tests, and cost where available
```

Wrapper backend matrix:

| Backend | Adapter goal | Context handoff | Who edits/applies | Notes |
| --- | --- | --- | --- | --- |
| `aider` | git-native pair-programming backend | `--read` context + target files | Aider | first backend to implement because edit/apply/git flow is mature |
| `claude-code` | hookable terminal-agent backend | `CLAUDE.md`, MCP, prompt/hook context | Claude Code | best deep lifecycle integration |
| `codex` | approval/sandbox backend | `AGENTS.md`, MCP, prepared context file | Codex | preserve Codex approval model |
| `gemini` | large/free-context backend | `GEMINI.md`, extension/MCP, context file | Gemini CLI | useful for broad adoption and headless runs |
| `opencode` | open agent UX backend | MCP/plugin config + context file | OpenCode | good fit for provider/LSP-heavy users |
| `continue` | IDE assistant framework backend | MCP/context provider/config snippet | Continue | primarily IDE-oriented but adapter-worthy |
| `cursor` | IDE/manual fallback | MCP/rules + `sg prepare --copy` | Cursor | not a true CLI backend, but export should exist |
| `copilot` | IDE/manual fallback | MCP/instructions + context file | Copilot | not a true CLI backend, but export should exist |
| `generic` | universal fallback | Markdown packet + JSON route metadata | user/chosen tool | supports web UIs and unknown CLIs |

Backend-specific examples:

```bash
sg prepare "fix auth token expiry" --for aider --out .skeletongraph/aider-context.md
sg exec --backend aider "fix auth token expiry"

sg prepare "explain checkout retry behavior" --for claude-code --out .skeletongraph/claude-context.md
sg exec --backend claude-code "fix checkout retry behavior"

sg prepare "add tests for validate_token" --for codex --out .skeletongraph/codex-context.md
sg exec --backend codex "add tests for validate_token"

sg prepare "refactor payment retry code" --for gemini --out .skeletongraph/gemini-context.md
sg exec --backend gemini "refactor payment retry code"

sg prepare "review changed auth files" --for opencode --out .skeletongraph/opencode-context.md
sg exec --backend opencode "review changed auth files"

sg prepare "fix session timeout bug" --for generic --out .skeletongraph/context.md
```

Adapter implementation order should not mean only one backend matters. Aider is
first because it is the fastest way to prove wrapper value, not because SG is
an Aider-only project. The adapter interface should be generic from day one:

```python
BackendAdapter:
  name
  detect()
  prepare_context(packet, route) -> PreparedContext
  build_command(task, prepared_context, route) -> list[str]
  parse_result(...) -> BackendRunResult
```

Each adapter should record:

- backend name/version when detectable
- task prompt
- SG route and selected tier
- packet path
- target files
- command preview
- execution status
- tests/verify status when available
- cost/tokens when available

Native future flow:

```text
sg run "task" --execute
  -> writes patch output
  -> parses patch
  -> shows affected files
  -> user runs sg run --apply or sg apply <run_id>
  -> SG verifies patch paths and git state
  -> patch applies
  -> sg verify runs tests
  -> run log records outcome
```

## 19. Safety Rules for Patch Apply

Patch apply is the biggest safety boundary for CLI release.

Requirements:

- parse unified diff, do not blindly shell out on arbitrary text
- reject absolute paths
- reject `..` path traversal
- reject paths outside project root
- show affected files before applying
- detect deletes/renames/large rewrites
- require explicit `--apply`
- never auto-commit
- warn if git worktree is dirty
- preserve unrelated user changes
- log applied patch and verification result

Until this exists, `sg run --execute` should write output and require manual
inspection.

## 20. Memory System

Memory should make later packets smaller and more accurate.

Memory types:

1. Session memory
   - current turn notes
   - recent files touched
   - decisions made
   - unresolved follow-ups

2. Dedup memory
   - functions already sent in earlier turns
   - allow later packets to use signatures/summaries instead of full bodies

3. Learned graph memory
   - files/functions often changed together
   - co-occurrence edges

4. Diagnostics memory
   - failing tests
   - type/lint errors
   - commands run

5. Error-only follow-up packets
  - when a tool or command fails, store the error separately
  - next run can send only the error instead of the full packet

Auto rebuild after completion:
- IDE: report_completion triggers index refresh if enabled
- CLI: after execute, SG can refresh index and summaries

Why memory matters:

- reduces repeated context
- resolves "fix it" or "continue"
- improves related-file inclusion
- lets eval track whether SG helped over a session, not just one prompt

Important risk:

Bad memory can poison packets. Memory snippets must be small, recent, and
labeled. Old or low-confidence memory should decay.

## 21. MCP Server and IDE Rules

MCP should expose a compact set of tools.

Current/desired tools:

- `get_retrieval_context`
- `query_context`
- `expand_context`
- `report_completion`
- maybe `index_status`

IDE agents can call `get_retrieval_context` first to choose targets using AST
and summary signals, or call `query_context` to get a full packet directly.

Rules should be short. Long rules become prompt overhead and are often ignored.

Good rules:

- Before broad repo exploration, call SG for targeted coding tasks.
- Use SG packet as primary context when confidence is high.
- If SG confidence is low, use expansion or normal exploration.
- Report completion with changed files/tests.

Bad rules:

- force too many tool calls
- force strict multi-phase rituals
- overemphasize SLM/MLM/LLM roles inside IDEs
- tell the model to do things the IDE cannot actually enforce

## 22. Current Implementation State

Implemented in current pass:

- deterministic model router
- delivery-aware recommended model
- CLI/IDE model separation
- CLI provider presets
- local/Ollama provider preset
- API-key-free dry-run path
- `sg route`
- `sg doctor`
- `sg run --dry-run`
- `sg run --execute` provider call path
- run log helper
- `sg prepare --out`
- `sg prepare --copy`
- dynamic/static routing config
- tests for router/config/run helpers
- Windows-safe ASCII output fix for blast-radius arrows
- `sg prepare --quiet` crash fix

Validated:

```bash
python -m compileall -q src tests
python -m pytest -q
python -m skeletongraph route "fix validate_token" --path tests/fixtures/python_small --json-output
python -m skeletongraph run "fix validate_token" --path tests/fixtures/python_small --dry-run --json-output
python -m skeletongraph config --path tests/fixtures/python_small --cli-provider local
python -m skeletongraph doctor --path tests/fixtures/python_small --json-output
```

Not yet validated:

- real paid-provider `sg run --execute`
- local Ollama `sg run --execute`
- patch apply, because it is not implemented

## 23. Important Files for New Implementers

Read these first:

```text
README.md
docs/blueprint.md
src/skeletongraph/engine.py
src/skeletongraph/config.py
src/skeletongraph/retrieval/model_router.py
src/skeletongraph/cli/main.py
src/skeletongraph/cli/run_exec.py
src/skeletongraph/cli/prepare.py
src/skeletongraph/assembly/prompt_builder.py
tests/unit/test_model_router.py
tests/unit/test_config_cli_provider.py
tests/unit/test_cli_run_exec.py
```

Then inspect:

```text
src/skeletongraph/storage/local.py
src/skeletongraph/retrieval/resolver.py
src/skeletongraph/retrieval/classifier.py
src/skeletongraph/server/mcp.py
eval/
```

## 24. Known Product Gaps

SG IDE gaps:

- rules need v5 rewrite
- MCP compact profile needs hardening
- stale-index warning should be first-class
- packet metadata not rich enough
- completion reporting semantics need tests
- memory persistence needs cleanup
- IDE compliance eval missing

SG CLI gaps:

- safe patch apply missing
- `sg verify` missing
- `sg runs` missing
- wrapper backends missing (`aider`, `claude-code`, `codex`, `gemini`, `opencode`)
- backend-specific context exporters are partial/planned
- route/cost UI incomplete
- real execute smoke missing
- local execute smoke missing
- no provider timeout/error UX tests
- no route evaluation vs static baselines

Retrieval gaps:

- v5 packet assembler missing
- project vocabulary missing
- personalized traversal missing
- PageRank persistence incomplete
- large-function focused extraction incomplete
- packet completeness eval missing

Docs gaps:

- README examples need fresh-env verification
- local provider docs need exact platform notes
- provider model names may need version refresh before release

## 25. Release Milestones

### Milestone A: Credible CLI Dry Run

Definition:

- `sg build`
- `sg doctor`
- `sg route`
- `sg prepare`
- `sg run --dry-run`
- README explains no-key path
- tests pass

Current state: mostly achieved.

### Milestone B: Safe CLI Execution

Definition:

- provider/local execution smoke tested
- output parsed as patch or explanation
- safe apply implemented
- dirty worktree handling
- `sg verify`
- run log records route/cost/verification

Current state: not achieved.

### Milestone C: Eval-Backed Routing Claim

Definition:

- route eval dataset exists
- always-MLM and always-LLM baselines
- cost per passing task reported
- underpower/overkill measured

Current state: planned.

### Milestone D: SG IDE Adoption Quality

Definition:

- compact rules
- stale-index check
- compliance eval
- first packet solves most targeted tasks

Current state: planned/partial.

## 26. Evaluation

The canonical evaluation plan is in `docs/evaluation.md`.

The short rule is:

```text
Never report token savings without pass rate or quality.
```

The required evaluation tracks are:

- retrieval and packet completeness
- prompt classifier reliability
- CLI model routing vs static baselines
- wrapper backend value across Aider, Claude Code, Codex, Gemini CLI, OpenCode,
  Continue, and generic context handoff
- native CLI execution safety
- IDE compliance
- human adoption friction
- cost per passing task

## 27. What Attracts Users Beyond Cost

For SG IDE:

- fewer frustrating exploration loops
- better first answer
- works with current IDE subscription
- no extra API key
- blast-radius awareness
- tests included earlier
- project memory
- manual fallback packet

For SG CLI:

- transparent route reasons
- local model testing
- provider presets
- no secret storage
- dry-run confidence before spending money
- export packets to other tools
- run history
- verification loop
- benchmark reports for their repo

For advanced users:

- JSON output
- reusable context artifacts
- integration with Aider/Codex/Claude/Gemini/OpenCode
- eval harness
- deterministic reproducibility

## 28. API Key and Model Testing Guidance

No key needed:

```bash
sg build
sg doctor
sg route "task"
sg prepare "task"
sg run "task" --dry-run
```

Provider key needed:

```bash
sg config --cli-provider anthropic
set ANTHROPIC_API_KEY=...
sg run "task" --execute
```

Local no-key path:

```bash
ollama pull qwen3-coder:latest
ollama serve
sg config --cli-provider local
sg run "task" --execute
```

OpenAI-compatible path:

```bash
sg config --cli-provider local --cli-api-base http://localhost:11434
sg config --cli-mlm ollama/qwen3-coder:latest
```

The local path is good for testing SG's pipeline, logs, patch parsing, and
verification. It is not a substitute for evaluating high-quality provider
models unless the product claim is explicitly "works with local models."

## 29. Implementation Priorities From Here

Highest leverage next steps:

1. Build a 50 prompt-target retrieval/classification eval dataset.
2. Harden the prompt classifier and ambiguity fallback.
3. Add stale-index detection to `sg doctor`, MCP responses, and wrappers.
4. Define the generic `BackendAdapter` interface.
5. Implement `sg prepare --for` exporters for `aider`, `claude-code`,
   `codex`, `gemini`, `opencode`, `continue`, and `generic`.
6. Implement `sg exec --backend aider` as the first executable adapter.
7. Implement backend run logging for wrapper executions.
8. Add `sg runs` so wrapper/native executions are inspectable.
9. Add `sg verify` as backend-independent test/diagnostic capture.
10. Add query-time embedding lookup for vague prompts after eval exists.
11. Rewrite IDE rule templates to be compact and v5-aligned.
12. Add packet component metadata and packet completeness eval.

Native provider execution remains useful, but it is not the first release bet.
The standalone native path should proceed only after wrapper orchestration and
evaluation prove the SG context layer is valuable.

Native execution follow-ups:

1. Smoke-test `sg run --execute` with local Ollama.
2. Smoke-test `sg run --execute` with one paid provider.
3. Implement safe patch parser/apply.
4. Add `--apply` approval gates.
5. Compare native SG execution against wrapper backends.

Do not start with:

- a full TUI
- subagent orchestration
- complicated LLM-based routing
- auto-commit behavior
- provider marketplace features
- full standalone editor-agent parity

Those are distractions until safe execution and eval are solid.

## 30. Two-Day Commit Plan

Use small commits because the worktree has unrelated churn. Prefer `git add -p`
when files contain mixed changes.

Day 1 commit 1:

```bash
git add -p src/skeletongraph/engine.py src/skeletongraph/retrieval/model_router.py tests/unit/test_model_router.py
git commit -m "Add deterministic SG model routing"
```

Day 1 commit 2:

```bash
git add -p src/skeletongraph/config.py tests/unit/test_config_cli_provider.py
git commit -m "Separate IDE and CLI model provider config"
```

Day 1 commit 3:

```bash
git add -p src/skeletongraph/cli/main.py src/skeletongraph/cli/run_exec.py tests/unit/test_cli_run_exec.py
git commit -m "Add model-routed SG CLI run pipeline"
```

Day 2 commit 1:

```bash
git add -p src/skeletongraph/cli/prepare.py src/skeletongraph/assembly/prompt_builder.py
git commit -m "Polish SG prepare and Windows-safe context output"
```

Day 2 commit 2:

```bash
git add .gitignore README.md
git add -A docs
git commit -m "Document SG IDE and CLI product blueprint"
```

Day 2 commit 3:

```bash
git add -p src/skeletongraph/config.py src/skeletongraph/cli/main.py tests/unit/test_config_cli_provider.py
git commit -m "Add local provider support for SG CLI execution"
```

Final check:

```bash
python -m compileall -q src tests
python -m pytest -q
git status --short
```

## 31. How To Brief a New Model

If passing this project to a new model, say:

```text
Read docs/blueprint.md and docs/evaluation.md first. SkeletonGraph is a graph
context engine with two surfaces: SG IDE and SG CLI. SG IDE is no-key MCP
context. SG CLI should become a wrapper/orchestrator for many existing tools,
not a full standalone agent first. The current work added model routing,
provider config, local provider support, sg route/doctor/run, and docs. Next
priority is classifier/retrieval eval, stale-index detection, backend adapters
for multiple CLIs, sg runs, and sg verify. Preserve unrelated worktree changes.
```

The new model should then inspect:

```text
git status --short
README.md
docs/blueprint.md
docs/evaluation.md
src/skeletongraph/cli/main.py
src/skeletongraph/config.py
src/skeletongraph/retrieval/model_router.py
tests/unit/
```

## 32. Final Product Position

SkeletonGraph should be positioned as:

```text
The context and routing layer that makes your coding agent start with the right
repo knowledge.
```

Not:

```text
Another coding agent.
```

That distinction matters. The market already has many agents. SG wins if it
makes all of them cheaper, faster, and less lost inside real repositories.
