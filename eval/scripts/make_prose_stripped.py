"""Strip location-giving cues out of every issue, to test SG under the hardest,
most realistic condition: a plain description with no traceback and no repro code.

WHAT THIS DOES (and does NOT do)
---------------------------------
Removes, from each task's `query` text:
  - Python tracebacks (the "Traceback (most recent call last): ... ErrorType: msg"
    block, and any stray "File "...", line N" frames outside a full traceback)
  - Fenced code blocks (```...```)

Deliberately does NOT remove explicit file/function names mentioned in prose
(e.g. "Header.fromstring does not accept..."). Surgically deleting a symbol name
out of a sentence leaves damaged, ungrammatical text that does not resemble a
real prose-only issue — natural prose-only issues read that way because the
reporter never named anything, not because a name was cut out. So the 12 "points
at code" tasks in the original 100 keep their naming here; this run converts the
other 78 (17 crash/traceback + 61 repro-snippet) into bare prose.

This is a SYNTHETIC ablation condition, not a naturally-occurring SWE-bench
subset — label it that way in any report (see project_headline_finding memory).
It makes the task harder for BOTH arms, not just the baseline: a repro snippet
sometimes carries information a model needs to understand the bug, not only to
locate it. So this is a fair stress test, not a rigged one.

Usage:
    python -m eval.scripts.make_prose_stripped
    python -m eval.scripts.make_prose_stripped --in eval/datasets/other.jsonl --out eval/datasets/other_stripped.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_IN = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl"
DEFAULT_OUT = "eval/datasets/swebench_100_prose_stripped.jsonl"

# A short line that exists only to introduce the payload immediately after it
# (a filename in backticks, a lone markdown heading, "Steps to reproduce:") —
# consumed TOGETHER with that payload below, not detected afterward from
# whitespace. Detecting it post-hoc (from gap size) is unreliable: blank-run
# collapsing happens before the check, so a 2-blank-line gap is already a
# 1-blank-line gap by the time you'd look for it. Fusing the intro into the
# same removal avoids that entirely, and correctly chains when several
# intro+payload pairs are stacked back to back (`file1.rst`: fence, blank,
# `file2.rst`: fence, ...), since each pair is matched and removed on its own.
_OPTIONAL_INTRO = r"(?:^[ \t]*(?:#{1,6}[ \t]*\S.{0,60}|`[^`\n]{1,80}`:?|\S.{0,60}:)[ \t]*\n[ \t]*\n)?"

_TRACEBACK_RE = re.compile(
    _OPTIONAL_INTRO + r"Traceback \(most recent call last\):.*?(?:\n\s*\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
# A stray frame line that survived outside a full traceback block (rare, but the
# archetype classifier treats a bare `File "...", line N` as traceback evidence
# too, so strip it for consistency with how tasks were originally classified).
_FRAME_RE = re.compile(r'^\s*File "[^"]+", line \d+.*$\n?', re.MULTILINE)
_FENCED_CODE_RE = re.compile(_OPTIONAL_INTRO + r"```.*?```", re.DOTALL | re.MULTILINE)


def strip_prose(query: str) -> str:
    # CRLF line endings are common in these GitHub issue bodies. Every regex
    # above anchors on bare \n; a stray \r before it silently breaks the match
    # (found empirically: an intro-then-fence pair failed to fuse because the
    # intro line ended in "...\r\n" not "...\n", so [ \t]*\n never matched
    # right after the colon). Normalize once, up front, rather than teach every
    # pattern to tolerate \r.
    query = query.replace("\r\n", "\n").replace("\r", "\n")
    q = _TRACEBACK_RE.sub("\n", query)
    q = _FRAME_RE.sub("", q)
    q = _FENCED_CODE_RE.sub("\n", q)
    q = re.sub(r"\n{3,}", "\n\n", q).strip()
    return q


def has_traceback(q: str) -> bool:
    return "Traceback" in q or 'File "' in q


def has_code(q: str) -> bool:
    return q.count("```") >= 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]

    before_tb = before_code = after_tb = after_code = changed = 0
    out_rows = []
    for r in rows:
        q0 = r.get("query", "") or ""
        q1 = strip_prose(q0)
        before_tb += has_traceback(q0)
        before_code += has_code(q0)
        after_tb += has_traceback(q1)
        after_code += has_code(q1)
        if q1 != q0:
            changed += 1
        r2 = dict(r)
        r2["query"] = q1
        r2["_query_orig_len"] = len(q0)
        r2["_query_stripped_len"] = len(q1)
        out_rows.append(r2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"read  {len(rows)} tasks from {args.inp}")
    print(f"wrote {len(out_rows)} tasks to {args.out}")
    print()
    print(f"tasks with a traceback : {before_tb} -> {after_tb}")
    print(f"tasks with a code block: {before_code} -> {after_code}")
    print(f"tasks whose issue text changed: {changed}/{len(rows)}")
    shrink = sum(r["_query_orig_len"] - r["_query_stripped_len"] for r in out_rows)
    print(f"total characters removed: {shrink:,}")
    if after_tb or after_code:
        print(f"\n  NOTE: {after_tb} traceback(s) and {after_code} code block(s) survived "
              f"stripping — inspect these tasks, the regexes are best-effort on "
              f"free-form GitHub issue text.")


if __name__ == "__main__":
    main()
