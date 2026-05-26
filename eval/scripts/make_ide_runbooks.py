"""Write a per-repo SG_EVAL_RUNBOOK.md into every task repo for IDE testing.

For each task in the dataset, drops a prompt/control runbook into its repo_path.
The repo-local file intentionally excludes gold files, verifying tests, and
patch hints so an IDE agent cannot leak the answer by opening the runbook. Open
the repo in your IDE, paste the Prompt, run once native and once with the
SkeletonGraph MCP server, then compare tokens/turns/tool-calls externally.

The runbook filename is in isolation._SG_ARTIFACTS, so it is stripped from eval
workspaces and can never leak into automated retrieval results.

    python -m eval.scripts.make_ide_runbooks
    python -m eval.scripts.make_ide_runbooks --dataset eval/datasets/contextbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = EVAL_DIR / "datasets" / "stage0.jsonl"

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_LEAKY_RESOLUTION_RE = re.compile(
    r"(?ims)^#{0,6}\s*(?:potential\s+resolution|possible\s+solution)\b.*\Z"
)
_MAYBE_SOLUTION_RE = re.compile(
    r"(?ims)^Maybe\s+one\s+solution\s+would\s+be\s+to\s+do\s*:.*\Z"
)


def sanitize_issue_prompt(query: str) -> str:
    """Strip issue-template noise and solution leaks for manual IDE prompts."""
    text = (query or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _LEAKY_RESOLUTION_RE.sub("", text)
    text = _MAYBE_SOLUTION_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_TMPL = """# SkeletonGraph IDE Evaluation — {task_id}

> **This file is an untracked working doc.** Every reset uses
> `git clean -fd -e SG_EVAL_RUNBOOK.md` so it survives. Regenerate any time:
> `python -m eval.scripts.make_ide_runbooks`

## The task

Use the external SWE-bench/eval manifest after the run to score files, tests,
and correctness. Do not place gold files or patch hints inside the repo before
the agent finishes.

## How the comparison works

Each IDE is compared **against itself**: a baseline run (no SkeletonGraph) and
an SG run. **Never compare a Cursor number to a Claude Code number** — they are
different agents with different models and scaffolds. Valid comparisons:
`Cursor-baseline vs Cursor-SG` and `ClaudeCode-baseline vs ClaudeCode-SG`.
Within one IDE keep the **same model** for both runs.

Two repo states must be exactly right:

| | baseline run | SG run |
|---|---|---|
| `.skeletongraph/` | absent | present (built) |
| `.mcp.json` / `.cursor/` / `.claude/` | absent | present (installed) |

## THE PROMPT — paste exactly, identical for every run, both IDEs

The block below is the IDE prompt: one standard instruction line, a blank line,
then the SWE-bench issue text with issue-template HTML comments and leaky
"Potential resolution" sections removed. Do not reword, trim, or "clarify" it
when comparing native vs SG.

This repo has no test environment installed. If the agent tries to run tests and
they fail, that is expected — **let it proceed, do not intervene**. It happens
equally in baseline and SG runs, so it does not bias the comparison.

````
{query}
````

## One-time setup (do once, ever)

```powershell
pip install -e "{repo_root}[mcp]"
sg --help     # confirm the `sg` command works
```

---

# CLAUDE CODE

### CC-1 · Baseline (no SkeletonGraph)

```powershell
cd "{repo_path}"
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
Remove-Item -Recurse -Force .skeletongraph, .mcp.json, .claude, .cursor -ErrorAction SilentlyContinue
git status        # must be clean (only SG_EVAL_RUNBOOK.md untracked)
```
Open Claude Code in this folder → run `/mcp` (must show **no** servers) → paste
THE PROMPT → let it finish untouched.
```powershell
git diff > "{results_dir}\\{task_id}_cc_baseline.patch"
```
In the session: run `/cost` and `/context`, note the numbers. Then reset:
```powershell
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
```

### CC-2 · With SkeletonGraph

```powershell
cd "{repo_path}"
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
sg build                       # builds .skeletongraph/ — note the time printed
sg install --ide claude-code   # writes .mcp.json + .claude/ hooks
```
Open a **fresh** Claude Code session here → `/mcp` (must now show the SG server
+ tools) → paste the SAME PROMPT → let it finish.
```powershell
git diff > "{results_dir}\\{task_id}_cc_sg.patch"
```
`/cost`, `/context`. Then reset:
```powershell
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
```

---

# CURSOR

### CU-1 · Baseline (no SkeletonGraph)

```powershell
cd "{repo_path}"
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
Remove-Item -Recurse -Force .skeletongraph, .mcp.json, .cursor, .claude -ErrorAction SilentlyContinue
git status
```
Open this folder in Cursor → Settings → MCP: confirm **no** `skeletongraph`
server. New **Agent** chat, pick your model (note which) → paste THE PROMPT →
let it finish.
```powershell
git diff > "{results_dir}\\{task_id}_cursor_baseline.patch"
```
Reset:
```powershell
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
```

### CU-2 · With SkeletonGraph

```powershell
cd "{repo_path}"
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
sg build
sg install --ide cursor        # writes .cursor/mcp.json + rules
```
**Reload Cursor** (Command Palette → "Reload Window") → Settings → MCP: confirm
`skeletongraph` is listed and enabled. New Agent chat, **same model as CU-1** →
paste the SAME PROMPT → let it finish.
```powershell
git diff > "{results_dir}\\{task_id}_cursor_sg.patch"
```
Reset:
```powershell
git checkout -- .
git clean -fd -e SG_EVAL_RUNBOOK.md
```

---

# Verification

Score the saved patch outside the repo using the SWE-bench/eval manifest. Keep
gold files, verifying tests, and patch descriptions out of this runbook so the
agent cannot read the answer during the run.

---

# RECORD

| Metric | CC baseline | CC + SG | Cursor baseline | Cursor + SG |
|---|---|---|---|---|
| Found expected edit target(s)? | | | | |
| Fix plausible by external rubric? | | | | |
| Assistant turns / agent steps | | | | |
| Total tool calls | | | | |
| SG tools used (which / count) | n/a | | n/a | |
| Input tokens | | | n/a | n/a |
| Output tokens | | | n/a | n/a |
| Requests used (Cursor) | n/a | n/a | | |
| Wall-clock time | | | | |
| Peak context | | | n/a | n/a |
| `sg build` time | n/a | | n/a | |

**Observations** (pipeline issues, where SG helped/hurt, retrieval confidence):
-

<!-- meta: repo={repo}  base={base_commit}  path={repo_path} -->
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--results-dir", default=str(EVAL_DIR.parent / "temp" /
                                                 "ide_eval" / "results"),
                    help="where IDE patch/metric files are saved")
    args = ap.parse_args()

    repo_root = str(EVAL_DIR.parent)
    tasks = [json.loads(l) for l in
             args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    written = skipped = 0
    for t in tasks:
        repo_path = Path(t["repo_path"])
        if not repo_path.is_dir():
            print(f"  skip {t['task_id']}: repo_path missing ({repo_path})")
            skipped += 1
            continue
        body = _TMPL.format(
            task_id=t["task_id"], repo=t.get("repo", ""),
            base_commit=t.get("base_commit", "")[:12],
            repo_path=str(repo_path),
            repo_root=repo_root,
            results_dir=args.results_dir,
            query="Fix the following GitHub issue in this repository:\n\n"
                  + sanitize_issue_prompt(t["query"]),
        )
        (repo_path / "SG_EVAL_RUNBOOK.md").write_text(body, encoding="utf-8")
        written += 1
    print(f"Wrote {written} runbooks ({skipped} skipped) from {args.dataset}")
    print(f"IDE task list = the {written} tasks in {args.dataset.name}")


if __name__ == "__main__":
    main()
