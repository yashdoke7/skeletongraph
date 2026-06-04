"""Tighten gold_fqns: derive the ACTUALLY-patched function from the diff's
changed line numbers + the base-commit AST, instead of the unreliable hunk-header
text. Writes an enriched dataset so localization metrics stop being gold-noise.

Why: parse_patch() in make_dataset takes the function name from the '@@ ... def x'
hunk-header CONTEXT, which often names the wrong scope (a class, a parent, or
nothing). The true gold function is whichever function in the BASE-COMMIT file
spans the changed lines. We compute that with stdlib `ast` (repos are Python).

    python -m eval.scripts.tighten_gold_fqns \
        --in  C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl \
        --out C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100_fqn.jsonl

Needs the gold PATCH. We pull it from the SWE-bench Verified HF dataset in
OFFLINE cache mode (you already downloaded it to build the dataset). If the
cache miss happens, it says so — re-run once online.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")      # use local cache, no network
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def changed_old_lines(patch: str) -> dict:
    """{file_path: set(old_line_numbers touched)} from a unified diff.

    Old-side lines = context (' ') + removed ('-'); these exist in the base file,
    so they map cleanly onto base-commit AST spans. Pure insertions still land
    inside the enclosing function via the surrounding context lines.
    """
    out: dict = {}
    cur = None
    old_ln = 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:].strip()
            out.setdefault(cur, set())
            continue
        m = _HUNK.match(line)
        if m:
            old_ln = int(m.group(1))
            continue
        if cur is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            out[cur].add(old_ln); old_ln += 1
        elif line.startswith(" "):
            out[cur].add(old_ln); old_ln += 1
        elif line.startswith("+") and not line.startswith("+++"):
            out[cur].add(old_ln)         # insertion point → enclosing fn via this line
    return {f: s for f, s in out.items() if s}


class _FnSpans(ast.NodeVisitor):
    """Collect (qualname, start_line, end_line) for every def/class."""
    def __init__(self):
        self.spans = []
        self._stack = []

    def _add(self, node):
        self._stack.append(node.name)
        self.spans.append((".".join(self._stack), node.lineno,
                           getattr(node, "end_lineno", node.lineno)))
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _add
    visit_AsyncFunctionDef = _add
    visit_ClassDef = _add


def fqns_for_file(repo: Path, rel: str, lines: set) -> set:
    """Innermost def/class span containing each changed line → {rel::qualname}."""
    if not rel.endswith(".py"):
        return set()
    fp = repo / rel
    try:
        src = fp.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return set()
    v = _FnSpans(); v.visit(tree)
    out = set()
    for ln in lines:
        best = None
        for name, s, e in v.spans:
            if s <= ln <= e and (best is None or (e - s) < (best[2] - best[1])):
                best = (name, s, e)
        if best:
            out.add(f"{rel}::{best[0]}")
    return out


def load_patches() -> dict:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {r["instance_id"]: r.get("patch", "") for r in ds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100_fqn.jsonl"))
    args = ap.parse_args()

    try:
        patches = load_patches()
        print(f"loaded {len(patches)} gold patches from HF cache")
    except Exception as e:
        raise SystemExit(
            f"could not load SWE-bench Verified patches from HF cache: {e}\n"
            f"Run once ONLINE (unset HF_DATASETS_OFFLINE) to populate the cache, "
            f"then re-run this offline.")

    tasks = [json.loads(l) for l in args.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_changed = improved = 0
    with args.out.open("w", encoding="utf-8") as f:
        for t in tasks:
            patch = patches.get(t["task_id"], "")
            new_fqns = set()
            if patch:
                repo = Path(t["repo_path"])
                for rel, lines in changed_old_lines(patch).items():
                    new_fqns |= fqns_for_file(repo, rel, lines)
            old = set(t.get("gold_fqns", []))
            if new_fqns:
                t["gold_fqns_hunk"] = sorted(old)        # keep the old for comparison
                t["gold_fqns"] = sorted(new_fqns)
                n_changed += 1
                if new_fqns != old:
                    improved += 1
            f.write(json.dumps(t) + "\n")
    print(f"wrote {args.out.name}: {n_changed}/{len(tasks)} tasks got AST gold_fqns "
          f"({improved} differ from hunk-header version)")


if __name__ == "__main__":
    main()
