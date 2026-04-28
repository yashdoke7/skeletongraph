# SkeletonGraph Evaluation Workflow

> **Research-Grade Benchmarking for AI Coding Agent Context Optimization**

This document is the canonical reference for running SkeletonGraph evaluations. It covers two evaluation modes:

1. **Quick Eval** — Per-project, per-agent comparison (SG ON vs OFF) using the legacy `eval` command
2. **Benchmark Eval** — Research-grade evaluation against SWE-bench Verified and standardized datasets

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Metrics Reference](#metrics-reference)
- [Part 1: Quick Eval (Per-Agent Playbooks)](#part-1-quick-eval-per-agent-playbooks)
- [Part 2: Benchmark Eval (SWE-bench)](#part-2-benchmark-eval-swe-bench)
- [Part 3: Supplementary Benchmarks](#part-3-supplementary-benchmarks)

---

## Prerequisites

### Install SkeletonGraph
```bash
cd /path/to/skeletongraph
pip install -e .

# Verify installation
skeletongraph --version
```

### Required dependencies for benchmarking
```bash
# For SWE-bench dataset loading
pip install datasets

# For precise token counting (already bundled, but verify)
pip install tiktoken
```

---

## Metrics Reference

### Metric Dimensions

All evaluation outputs report metrics organized into **4 dimensions**:

| Dimension | Metrics | What It Proves |
|:---|:---|:---|
| **A: Token Efficiency** | Retrieval tokens, total tokens, reduction ratio, API cost | SG uses fewer tokens → lower cost, fewer rate limit hits |
| **B: Retrieval Quality** | Precision, Recall, F1, MRR, Hit@k | SG finds the right files → better context relevance |
| **C: Execution Quality** | Test pass rate, regression rate, apply rate | SG doesn't sacrifice code quality for token savings |
| **D: Operational Efficiency** | Turns, tool calls, redundant views, wall time | SG makes agents more efficient in their workflow |

### Metric Details

#### Dimension A: Token Efficiency
| Metric | Formula | Description |
|:---|:---|:---|
| **Retrieval Tokens** | `sum(tool_call.output_tokens)` | Total tokens from all file reads, searches, and context fetches |
| **Total Conversation Tokens** | `retrieval + response + history + schema` | All tokens consumed in the session |
| **Reduction Ratio** | `native_tokens / sg_tokens` | How many times smaller SG is (higher = better) |
| **API Cost (USD)** | `total_tokens × $3.00/1M` | Estimated cost at standard input pricing |
| **Cost Savings %** | `((native - sg) / native) × 100` | Percentage of cost saved by using SG |

#### Dimension B: Retrieval Quality
| Metric | Formula | Description |
|:---|:---|:---|
| **Precision** | `\|retrieved ∩ ground_truth\| / \|retrieved\|` | % of retrieved files that were actually needed |
| **Recall** | `\|retrieved ∩ ground_truth\| / \|ground_truth\|` | % of needed files that were actually retrieved |
| **F1** | `2 × P × R / (P + R)` | Harmonic mean (balanced quality score) |
| **MRR** | `1 / rank_of_first_correct` | How early the first relevant file appears |
| **Hit@k** | binary | Was ≥1 correct file in the first k retrievals? |

#### Dimension C: Execution Quality
| Metric | Source | Description |
|:---|:---|:---|
| **Resolved Rate** | `pytest` on agent patch | % of tasks where failing tests now pass |
| **Regression Rate** | `pytest` on agent patch | % of tasks where passing tests broke |
| **Files Modified Overlap** | Jaccard similarity | Overlap between agent edits and gold patch |

#### Dimension D: Operational Efficiency
| Metric | Source | Description |
|:---|:---|:---|
| **Total Turns** | Trace count | Agent←→LLM round trips |
| **Total Tool Calls** | Trace count | All tool invocations |
| **Redundant File Views** | Dedup check | Files read more than once |
| **SG-Specific Queries** | Trace filter | query_context, expand_function, search_index calls |

### Token Counting
- **SkeletonGraph uses tiktoken BPE (cl100k_base)** — exact token counts
- CRG uses `len(text) // 4` — approximate, ~15% error margin
- Graphify uses `len(text) // 4` — approximate

### Data Source Per Agent

| Agent | SG Data Source | Native Export Method | L1 Retrieval | L2 Response | L3 History | L4 Reasoning |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Antigravity** | `.skeletongraph/session/current.json` | Export Chat button | ✅ | ✅ | ✅ computed | ❌ |
| **Cursor** | `.skeletongraph/session/current.json` | Copy chat pane text | ✅ | ✅ | ✅ computed | ❌ |
| **Claude Code** | `.skeletongraph/session/current.json` | `/export` command | ✅ | ✅ | ✅ computed | ✅ |
| **Codex** | `.skeletongraph/session/current.json` | Auto from `~/.codex/` | ✅ | ✅ | ✅ computed | ✅ |
| **Copilot** | `.skeletongraph/session/current.json` | `Ctrl+Shift+P` → Export Session | ✅ | ✅ | ✅ computed | ✅ |

---

## Part 1: Quick Eval (Per-Agent Playbooks)

> Use this for rapid per-project comparisons. Run the same prompt with SG ON and SG OFF, then generate a comparison report.

### Universal Setup (Do Once Per Project)

```bash
# 1. Clone a target project
git clone https://github.com/pallets/flask.git
cd flask

# 2. Build the SkeletonGraph index
skeletongraph build

# 3. Install IDE integration (auto-detects and configures)
skeletongraph install
```

### Universal Evaluation Pattern

For EVERY agent, the pattern is identical:

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Run WITH SkeletonGraph (Target)                │
│  ├─ Ensure MCP server is connected                      │
│  ├─ Run the evaluation prompt                           │
│  ├─ SG session auto-saved to current.json               │
│  └─ `git reset --hard` to reset repo state              │
│                                                         │
│  Step 2: Run WITHOUT SkeletonGraph (Baseline)           │
│  ├─ Disconnect/remove MCP server                        │
│  ├─ Run the EXACT SAME prompt natively                  │
│  ├─ Export the chat (method varies per agent)            │
│  └─ Save export to eval_logs/{agent}/{project}/         │
│                                                         │
│  Step 3: Generate Comparison Report                     │
│  └─ skeletongraph eval --agent {agent} --native-file... │
└─────────────────────────────────────────────────────────┘
```

> **CRITICAL**: You must run the SG session FIRST (Step 1), then the native session (Step 2). The SG session data is stored in `.skeletongraph/session/current.json` and will be overwritten if you run SG again.

---

### 1. Antigravity

**Step 1: SG Run (Target)**
1. Ensure `mcp_config.json` has `skeletongraph` server active.
2. Run the evaluation prompt.
3. Session auto-saved. Reset repo: `git reset --hard`

**Step 2: Native Run (Baseline)**
1. Remove SkeletonGraph from `mcp_config.json`.
2. Open new chat. Run the exact same prompt.
3. Click **"Export Chat"** button at top of conversation.
4. Save to:
```
eval_logs/antigravity/{project}/native_export.txt
```

**Step 3: Generate Report**
```bash
skeletongraph eval \
  --agent antigravity \
  --native-file eval_logs/antigravity/flask/native_export.txt \
  --project flask \
  --path /path/to/flask
```

---

### 2. Cursor

**Step 1: SG Run (Target)**
1. Connect MCP server in Cursor Settings → MCP.
2. Run the evaluation prompt.
3. Session auto-saved. Reset repo: `git reset --hard`

**Step 2: Native Run (Baseline)**
1. Disconnect the MCP server in Cursor settings.
2. Delete `.cursorrules` if it references SG.
3. Open new chat. Run the exact same prompt.
4. Select all text in the chat pane → Copy → Save to:
```
eval_logs/cursor/{project}/native_export.txt
```

**Step 3: Generate Report**
```bash
skeletongraph eval \
  --agent cursor \
  --native-file eval_logs/cursor/flask/native_export.txt \
  --project flask \
  --path /path/to/flask
```

---

### 3. Claude Code (CLI)

**Step 1: SG Run (Target)**
1. Run `skeletongraph install claude_code` to configure `claude.json`.
2. Run `claude` in the project directory.
3. Execute the evaluation prompt.
4. Session auto-saved. Reset repo: `git reset --hard`

**Step 2: Native Run (Baseline)**
1. Remove SkeletonGraph from `~/.claude.json`.
2. Run `claude` again. Execute the exact same prompt.
3. Type `/export` in the Claude CLI.
4. Save the exported markdown to:
```
eval_logs/claude_code/{project}/native_export.md
```

**Step 3: Generate Report**
```bash
skeletongraph eval \
  --agent claude_code \
  --native-file eval_logs/claude_code/flask/native_export.md \
  --project flask \
  --path /path/to/flask
```

---

### 4. GitHub Copilot

**Pre-requisite**: Enable debug logging in VS Code:
- Settings → Search `github.copilot.chat.agentDebugLog.fileLogging.enabled` → Enable

**Step 1: SG Run (Target)**
1. Connect MCP server via VS Code settings or Windsurf/Continue IDE.
2. Run the evaluation prompt.
3. Session auto-saved. Reset repo: `git reset --hard`

**Step 2: Native Run (Baseline)**
1. Disconnect the MCP server.
2. Open new Copilot chat. Run the exact same prompt.
3. Press `Ctrl+Shift+P` → Type "Chat: Export Session..." → Save as:
```
eval_logs/copilot/{project}/native_export.json
```

**Step 3: Generate Report**
```bash
skeletongraph eval \
  --agent copilot \
  --native-file eval_logs/copilot/flask/native_export.json \
  --project flask \
  --path /path/to/flask
```

---

### 5. Codex (CLI)

**Step 1: SG Run (Target)**
1. Run `skeletongraph install codex` to configure.
2. Run codex with the evaluation prompt.
3. Session auto-saved. Reset repo: `git reset --hard`

**Step 2: Native Run (Baseline)**
1. Remove SkeletonGraph from codex config.
2. Run the same prompt natively.
3. Codex automatically saves JSONL tracking to `~/.codex/` — no manual export needed.

**Step 3: Generate Report**
```bash
skeletongraph eval \
  --agent codex \
  --project flask \
  --path /path/to/flask
```

---

### Quick Eval Output

Each `eval` run produces:
- `{project}/.skeletongraph/eval/report.md` — Human-readable comparison table
- `{project}/.skeletongraph/eval/comparison.json` — Machine-readable results

---

## Part 2: Benchmark Eval (SWE-bench)

> This is the research-grade evaluation. It uses standardized datasets with human-verified ground truth, enabling statistically rigorous metrics with confidence intervals.

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK PIPELINE                          │
│                                                                 │
│  1. Download SWE-bench Verified dataset (500 tasks, 12 repos)   │
│  2. For each task:                                              │
│     a. Clone repo at base_commit                                │
│     b. Build SG index                                           │
│     c. Run agent WITH SG → save trace as sg_trace.json          │
│     d. git reset --hard                                         │
│     e. Run agent WITHOUT SG → save trace as native_trace.json   │
│  3. Run: skeletongraph eval-benchmark                           │
│  4. Get: P/R/F1/MRR + token reduction + cost savings            │
│     with stdev and 95% CI across N tasks                        │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: List Available Repos

```bash
skeletongraph eval-list
```

Output:
```
              SWE-bench Verified Repos
┌───────────────────────────────┬───────┬────────┐
│ Repo                          │ Tasks │ Size   │
├───────────────────────────────┼───────┼────────┤
│ django/django                 │   118 │ large  │
│ scikit-learn/scikit-learn     │    56 │ large  │
│ matplotlib/matplotlib         │    48 │ large  │
│ astropy/astropy               │    22 │ large  │
│ pandas-dev/pandas             │    17 │ medium │
│ pylint-dev/pylint             │    15 │ medium │
│ pytest-dev/pytest             │    14 │ medium │
│ psf/requests                  │     8 │ small  │
│ pallets/flask                 │     3 │ small  │
│ ...                           │       │        │
└───────────────────────────────┴───────┴────────┘
```

### Step 2: Prepare Trace Directory

The benchmark runner expects traces in this structure:

```
benchmark_traces/
├── django__django-16527/
│   ├── sg_trace.json          # Trace from SG-augmented session
│   └── native_trace.json      # Trace from native session
├── django__django-16820/
│   ├── sg_trace.json
│   └── native_trace.json
└── ...
```

**How to create traces:**

For each SWE-bench task (`instance_id`):

```bash
# 1. Clone the repo at the specific commit
git clone https://github.com/django/django.git benchmark_repos/django
cd benchmark_repos/django
git checkout <base_commit>

# 2. Build SkeletonGraph index
skeletongraph build

# 3. Run agent WITH SG
#    Save the problem_statement from the dataset as your prompt.
#    After the agent finishes, copy/export the trace.
#    The SG session is auto-saved at .skeletongraph/session/current.json.

# 4. Convert to benchmark trace format
skeletongraph eval --agent <agent_name> \
  --path . \
  --project django
# Copy the generated comparison JSON to your traces dir

# 5. Reset and run WITHOUT SG
git reset --hard
# Remove MCP server, run same prompt, export native chat
```

### Step 3: Run Benchmark

```bash
# Full benchmark against SWE-bench Verified
skeletongraph eval-benchmark \
  --dataset swe-bench-verified \
  --traces-dir ./benchmark_traces \
  --repos-dir ./benchmark_repos \
  --output ./benchmark_results

# Filter to specific repos
skeletongraph eval-benchmark \
  --dataset swe-bench-verified \
  --repos "django/django,psf/requests" \
  --traces-dir ./benchmark_traces \
  --output ./benchmark_results

# Limit to N tasks
skeletongraph eval-benchmark \
  --dataset swe-bench-verified \
  --limit 30 \
  --traces-dir ./benchmark_traces \
  --output ./benchmark_results
```

### Step 4: Read Results

The benchmark produces two files:

**`benchmark_results/benchmark_report.md`** — Publishable markdown:
```
| Metric                    | Native Agent | SkeletonGraph | Improvement |
|:--------------------------|:-------------|:--------------|:----------|
| Avg Retrieval Tokens      | 32,441       | 4,210         | 7.7× ↓    |
| File Precision            | 0.31         | 0.72          | +132%     |
| File Recall               | 0.85         | 0.93          | +9%       |
| File F1                   | 0.45         | 0.81          | +80%      |
| MRR                       | 0.42         | 0.88          | +110%     |
```

**`benchmark_results/benchmark_results.json`** — Machine-readable with per-task detail, aggregate stats, and 95% confidence intervals.

---

## Part 3: Supplementary Benchmarks

> SWE-bench proves execution quality on real GitHub issues. These supplementary benchmarks prove additional dimensions.

### Large Repo Stress Test

Proves SG scales to massive codebases where native agents run out of context.

Target repos (from SWE-bench or standalone):
| Repo | Files | Why |
|:---|---:|:---|
| `django/django` | ~4,600 | SWE-bench overlap, web framework |
| `huggingface/transformers` | ~10,000+ | Massive ML repo, native agents fail here |
| Next.js monorepo | ~27,000+ | JS monorepo, proves cross-language scaling |

Key metric: **Token reduction ratio should INCREASE with repo size** (SG gets more valuable as repos get bigger — this is our key selling point over CRG which showed diminishing returns on some repos).

### Custom Dataset (Your Own Projects)

Create a JSONL file with one task per line:

```json
{"task_id": "flask-trailing-slash", "repo": "pallets/flask", "repo_url": "https://github.com/pallets/flask.git", "base_commit": "HEAD", "problem_statement": "Trailing slash routing skips global config. Fix Blueprint init.", "gold_patch": "", "ground_truth_files": ["src/flask/blueprints.py", "src/flask/scaffold.py"]}
```

Run:
```bash
skeletongraph eval-benchmark \
  --dataset custom \
  --dataset-file my_tasks.jsonl \
  --traces-dir ./my_traces \
  --output ./my_results
```

### CRG-Compatible Replay

Direct comparison against code-review-graph's published numbers using their exact repos and commits:

| Repo | CRG Commits | CRG Claimed Reduction |
|:---|:---|:---|
| express | 2 | 0.7x |
| fastapi | 2 | 8.1x |
| flask | 2 | 9.1x |
| gin | 3 | 16.4x |
| httpx | 2 | 6.9x |
| nextjs | 2 | 8.0x |

We re-run with tiktoken BPE (not CRG's `len//4`) and real agent traces (not static file reads).

---

## Comparison: Our Methodology vs Competitors

| Aspect | CRG | Graphify | **SkeletonGraph** |
|:---|:---|:---|:---|
| Token counter | `len(text)//4` (est.) | `len(text)//4` (est.) | **tiktoken BPE (exact)** |
| Baseline | Static file reads | Raw file dump | **Real agent session traces** |
| Ground truth | Self-referential graph | None | **SWE-bench human PRs** |
| Quality metric | None | None | **P/R/F1/MRR + Test Pass Rate** |
| Sample size | 13 commits | 1 folder | **30+ SWE-bench tasks** |
| Confidence | None | None | **95% CI with std dev** |
| Reproducible | Partially | No | **Public dataset + documented steps** |
