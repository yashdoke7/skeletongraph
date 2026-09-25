"""Stricter funnel measures, answering three construct-validity questions.

    python -m eval.scripts.funnel_strict [--json out.json]

1. Content exposure. The main funnel's "gold-file exposure" counts any tool result
   that mentions a gold path, including a path-only search hit. Here a run counts
   only if code from a gold file entered its context: a file read of it, a
   content-mode search result from it, a shell command that printed it, or a
   retriever payload for it. Path-only listings (Glob, Grep files-mode, `ls`,
   `rg --files`, `-l`) do not count.
2. Time to content exposure: the tool-using turn (Claude Code) or action index
   (Codex) at which that first happened, on tasks where both arms reached it.
3. Function-level edit: whether the patch changes a function the reference patch
   changes. Each gold file is parsed at the task's base commit (Python ast); every
   line the patch removes, or inserts after, is mapped to its enclosing functions
   and classes and compared with the gold functions. Base files come from the
   local checkout, else from GitHub at the base commit (cached in tmp/base_files/).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics as st
import subprocess
import urllib.request
from pathlib import Path

from eval.scripts.paper_v2_analysis import ROOT, load, norm

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "eval" / "datasets"
CACHE = REPO / "tmp" / "base_files"
PANELS = [
    ("Claude exploratory (July)", "claude_v7", "native", "claude_v7", "sg-fusion", "claude"),
    ("Claude exploratory (Sept.)", "claude_v7_rep2", "native", "claude_v7_rep2", "sg-fusion", "claude"),
    ("Claude lean", "claude_v8", "native", "claude_v7_rep2", "sg-fusion-plain", "claude"),
    ("Claude exploratory, SWE-rebench", "claude_rebench_v1", "native", "claude_rebench_v1", "sg-fusion", "claude"),
    ("Codex CLI", "codex_v1", "codex-native", "codex_v1", "codex-sg-plain", "codex"),
]
PATH_ONLY = re.compile(r"(^|\s)(ls|dir|tree|find)\b|--files\b|\s-l\b|files_with_matches|Get-ChildItem", re.I)
READ_CMD = re.compile(r"\b(cat|type|sed|head|tail|less|more|nl|awk|Get-Content|gc|python|rg|grep|"
                      r"Select-String|findstr)\b", re.I)
HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+\d+")


def _tasks() -> dict:
    out = {}
    for f in ("swebench_100.jsonl", "swe_rebench_100.jsonl"):
        for line in (DATASETS / f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["task_id"]] = r
    return out


TASKS = _tasks()


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return json.dumps(c) if c else ""


# ── 1-2. content exposure and when it happened ────────────────────────────────
def claude_content_turn(tag: str, d: dict):
    """First tool-using turn at which gold-file code entered the context, or None."""
    gold = [norm(g) for g in d.get("gold_files") or []]
    t = ROOT / tag / "_claude_transcripts" / (d["run_id"] + ".jsonl")
    if not gold or not t.exists():
        return None
    calls, turn = {}, 0
    for line in open(t, encoding="utf-8"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        content = (o.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        if o.get("type") == "assistant":
            used = False
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    used = True
                    calls[b.get("id")] = (turn, b.get("name", ""), b.get("input") or {})
            turn += used
            continue
        if o.get("type") != "user":
            continue
        for b in content:
            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                continue
            at, name, inp = calls.get(b.get("tool_use_id"), (None, "", {}))
            if at is None:
                continue
            txt = norm(_text(b.get("content")))
            hit = any(g in txt for g in gold)
            target = norm(inp.get("file_path") or inp.get("path") or "")
            in_gold = any(target.endswith(g) for g in gold)
            if name == "Read" and in_gold and txt.strip():
                return at
            if name == "Grep" and inp.get("output_mode") == "content" and (hit or (in_gold and txt.strip())):
                return at
            if name == "Bash":
                cmd = norm(inp.get("command", ""))
                if any(g in cmd for g in gold) and txt.strip() and READ_CMD.search(cmd) \
                        and not PATH_ONLY.search(cmd):
                    return at
            if "skeletongraph" in name and hit and "\n" in txt.strip():
                return at
    return None


def codex_content_turn(tag: str, d: dict):
    gold = [norm(g) for g in d.get("gold_files") or []]
    t = ROOT / tag / "_codex_transcripts" / (d["run_id"] + ".jsonl")
    if not gold or not t.exists():
        return None
    k = 0
    for line in open(t, encoding="utf-8"):
        o = json.loads(line)
        if o.get("type") != "item.completed":
            continue
        it = o.get("item") or {}
        typ = it.get("type")
        if typ not in ("command_execution", "mcp_tool_call", "file_change"):
            continue
        k += 1
        if typ == "command_execution":
            cmd = norm(it.get("command", ""))
            out = norm(it.get("aggregated_output") or "")
            if PATH_ONLY.search(cmd) or not out.strip():
                continue
            if any(g in cmd for g in gold) and READ_CMD.search(cmd):
                return k
            if any((g + ":") in out for g in gold):
                return k
        elif typ == "mcp_tool_call":
            res = norm(json.dumps(it.get("result") or it.get("output") or ""))
            if any(g in res for g in gold):
                return k
    return None


# ── 3. did the patch change a gold function ──────────────────────────────────
def _base_file(task_id: str, path: str):
    t = TASKS[task_id]
    key = CACHE / task_id / path.replace("/", "__")
    if key.exists():
        return key.read_text(encoding="utf-8", errors="replace")
    src = None
    repo = Path(t.get("repo_path", ""))
    if repo.is_dir():
        r = subprocess.run(["git", "-C", str(repo), "show", f"{t['base_commit']}:{path}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            src = r.stdout
    if src is None:
        url = f"https://raw.githubusercontent.com/{t['repo']}/{t['base_commit']}/{path}"
        try:
            src = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
        except Exception:
            return None
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(src, encoding="utf-8")
    return src


def _spans(src: str):
    """(qualname, start, end) for every function and class; nested names dotted."""
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                out.append((q, ch.lineno, ch.end_lineno or ch.lineno))
                walk(ch, q)
            else:
                walk(ch, prefix)
    walk(ast.parse(src), "")
    return out


def _touched_lines(patch: str) -> dict:
    """{path: base-file line numbers the patch removes, or inserts after}."""
    out, path, old = {}, None, 0
    for line in patch.splitlines():
        if line.startswith("--- "):
            continue
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            continue
        m = HUNK.match(line)
        if m:
            old = int(m.group(1))
            continue
        if path is None or not line:
            continue
        if line[0] == "-":
            out.setdefault(path, set()).add(old)
            old += 1
        elif line[0] == "+":
            out.setdefault(path, set()).add(max(old - 1, 1))
        elif line[0] == " ":
            old += 1
    return out


def edited_gold_function(d: dict):
    """True/False, or None when no gold function could be resolved for the task."""
    gold = {}
    for f in TASKS.get(d["task_id"], {}).get("gold_fqns") or []:
        if "::" in f:
            path, q = f.split("::", 1)
            gold.setdefault(path, set()).add(q)
    if not gold:
        return None
    touched = _touched_lines(d.get("model_patch") or "")
    resolved = False
    for path, qs in gold.items():
        src = _base_file(d["task_id"], path)
        if src is None:
            continue
        try:
            spans = _spans(src)
        except SyntaxError:
            continue
        resolved = True
        for ln in touched.get(path, ()):
            # every span enclosing the line is checked, so an edit nested inside a
            # gold function (an inner def, a branch) counts as editing it
            if any(a <= ln <= b and q in qs for q, a, b in spans):
                return True
    return False if resolved else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    report, tot = {}, {"differ": 0, "both_fn": 0}
    for lab, ta, aa, tb, ab, kind in PANELS:
        A, B = load(ta)[aa], load(tb)[ab]
        c = sorted(set(A) & set(B))
        fn = claude_content_turn if kind == "claude" else codex_content_turn
        xa = {t: fn(ta, A[t]) for t in c}
        xb = {t: fn(tb, B[t]) for t in c}
        ea = {t: edited_gold_function(A[t]) for t in c}
        eb = {t: edited_gold_function(B[t]) for t in c}
        ok = [t for t in c if ea[t] is not None and eb[t] is not None]
        both = [t for t in c if xa[t] is not None and xb[t] is not None]
        differ = [t for t in ok if A[t]["resolved"] != B[t]["resolved"]]
        both_fn = [t for t in differ if ea[t] and eb[t]]
        tot["differ"] += len(differ)
        tot["both_fn"] += len(both_fn)
        row = {
            "n": len(c),
            "content_exposure": [100 * sum(v is not None for v in xa.values()) / len(c),
                                 100 * sum(v is not None for v in xb.values()) / len(c)],
            "both_exposed": len(both),
            "turn_to_content_mean": [st.mean(xa[t] for t in both), st.mean(xb[t] for t in both)],
            "turn_to_content_median": [st.median(xa[t] for t in both), st.median(xb[t] for t in both)],
            "fn_resolved": len(ok),
            "edited_gold_function": [100 * sum(ea[t] for t in ok) / len(ok),
                                     100 * sum(eb[t] for t in ok) / len(ok)],
            "differ": len(differ), "differ_both_edited_function": len(both_fn),
        }
        report[lab] = row
        print(f"{lab:<33} n={len(c):>3} | content exposure {row['content_exposure'][0]:>3.0f}% -> "
              f"{row['content_exposure'][1]:>3.0f}% | turn to it (n={len(both)}) mean "
              f"{row['turn_to_content_mean'][0]:.1f} -> {row['turn_to_content_mean'][1]:.1f}, median "
              f"{row['turn_to_content_median'][0]:g} -> {row['turn_to_content_median'][1]:g} | "
              f"edited gold function {row['edited_gold_function'][0]:.0f}% -> "
              f"{row['edited_gold_function'][1]:.0f}% (n={len(ok)}) | differ {len(differ)}, "
              f"both edited a gold fn {len(both_fn)}")
    print(f"\nproduction panels: outcomes differ on {tot['differ']} tasks with resolvable gold functions;"
          f" on {tot['both_fn']} both arms edited a gold function")
    report["_totals"] = tot
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
