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

_TMPL = """# {task_id}   ({repo})

base: {base_commit}   path: {repo_path}

## 1) Native — Claude Code
```
cd "{repo_path}"
git checkout -- . && git clean -fd -e SG_EVAL_RUNBOOK.md
claude
# paste the Prompt below; let it finish
git --no-pager diff HEAD -- . ":(exclude)SG_EVAL_RUNBOOK.md"
```

## 2) +SG — Claude Code
```
cd "{repo_path}"
git checkout -- . && git clean -fd -e SG_EVAL_RUNBOOK.md
sg install      # adds SG MCP server + hooks to Claude Code
sg build        # build the index for this repo
claude
# paste the same Prompt; SG tools (sg_search/sg_expand) are now active
git --no-pager diff HEAD -- . ":(exclude)SG_EVAL_RUNBOOK.md"
```

## 3) Native — Cursor
```
cd "{repo_path}"
git checkout -- . && git clean -fd -e SG_EVAL_RUNBOOK.md
# open in Cursor with SG MCP OFF (Settings -> MCP); paste the Prompt
git --no-pager diff HEAD -- . ":(exclude)SG_EVAL_RUNBOOK.md"
```

## 4) +SG — Cursor
```
cd "{repo_path}"
git checkout -- . && git clean -fd -e SG_EVAL_RUNBOOK.md
sg install      # writes .cursor/mcp.json
sg build
# open in Cursor with SG MCP ON; paste the Prompt
git --no-pager diff HEAD -- . ":(exclude)SG_EVAL_RUNBOOK.md"
```

## Prompt (paste exactly)
```
{query}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = ap.parse_args()

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
            query="Fix the following GitHub issue in this repository:\n\n"
                  + sanitize_issue_prompt(t["query"]),
        )
        (repo_path / "SG_EVAL_RUNBOOK.md").write_text(body, encoding="utf-8")
        written += 1
    print(f"Wrote {written} runbooks ({skipped} skipped) from {args.dataset}")


if __name__ == "__main__":
    main()
