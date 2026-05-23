"""Write a per-repo SG_EVAL_RUNBOOK.md into every task repo for IDE testing.

For each task in the dataset, drops a runbook into its repo_path containing the
exact raw SWE-bench prompt (verbatim — no modifications), the gold files (for
your verification only), and the native-vs-+SG procedure. Open the repo in your
IDE, paste the Prompt, run once native and once with the SkeletonGraph MCP
server, then compare tokens/turns/tool-calls (Agent Debug Logs) and whether the
patch touches the gold files.

The runbook filename is in isolation._SG_ARTIFACTS, so it is stripped from eval
workspaces and can never leak into automated retrieval results.

    python -m eval.scripts.make_ide_runbooks
    python -m eval.scripts.make_ide_runbooks --dataset eval/datasets/contextbench.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = EVAL_DIR / "datasets" / "stage0.jsonl"

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

## Verify — does the diff touch these gold files?
{gold_files}

## Prompt (paste verbatim)
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
        gold = "\n".join(f"- `{g}`" for g in t.get("gold_files", [])) or "- (none)"
        body = _TMPL.format(
            task_id=t["task_id"], repo=t.get("repo", ""),
            base_commit=t.get("base_commit", "")[:12],
            repo_path=str(repo_path),
            gold_files=gold, query=t["query"],
        )
        (repo_path / "SG_EVAL_RUNBOOK.md").write_text(body, encoding="utf-8")
        written += 1
    print(f"Wrote {written} runbooks ({skipped} skipped) from {args.dataset}")


if __name__ == "__main__":
    main()
