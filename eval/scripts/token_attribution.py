"""Where do the tokens ACTUALLY go in an agentic loop? Attribute total input
tokens to the tool results that caused them, accounting for CARRY-FORWARD.

The key mechanic this measures: in an agent loop, `total_input_tokens` is the
SUM over turns of the context size at each turn. A tool result that lands at
turn 3 of a 15-turn run is re-sent to the model 12 more times. So a result's
true cost is (its size) x (turns remaining after it lands), NOT just its size.
That means an early, fat tool result is catastrophically more expensive than a
late one of the same size — and it's why "retrieval quality" and "token cost"
are only loosely coupled.

Usage:
    python -m eval.scripts.token_attribution --tag claude_v7 --arms native,sg-fusion
    python -m eval.scripts.token_attribution --tag claude_v7 --arms native,sg-fusion --limit 20

Prints per arm:
  1. Total tool-result bytes, and carry-weighted bytes (bytes x turns-remaining),
     broken down by tool. Carry-weighted share is the number that matters.
  2. Phase split: RETRIEVAL/EXPLORE tools vs EDIT/VERIFY tools vs everything else
     — answers "if retrieval were FREE, what's the theoretical max saving?"
     (Amdahl ceiling on any retrieval-side optimization.)
  3. Where each arm's context peaks.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

# Tool classification. RETRIEVAL = finding/reading code (what SG replaces or
# accelerates). EDIT_VERIFY = changing code + running tests (what SG cannot
# touch — this is the Amdahl denominator).
RETRIEVAL_TOOLS = {
    "Grep", "Glob", "Read", "ToolSearch", "WebSearch", "WebFetch",
    "mcp__skeletongraph__sg_search", "mcp__skeletongraph__sg_expand",
    "mcp__skeletongraph__sg_overview", "mcp__skeletongraph__sg_get",
}
EDIT_VERIFY_TOOLS = {"Edit", "Write", "Bash", "NotebookEdit", "PowerShell", "MultiEdit"}


def approx_tokens(s: str) -> int:
    """~4 chars/token, the standard rough English/code heuristic. We only need
    RELATIVE attribution here, so a consistent approximation is fine."""
    return len(s) // 4


def parse_transcript(path: Path) -> dict:
    """Return {tool_name: [(tokens, turn_index), ...]}, total_turns."""
    try:
        objs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return {}, 0
    pending = {}
    results = defaultdict(list)
    turn = 0
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            turn += 1
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    pending[b.get("id")] = str(b.get("name", ""))
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    name = pending.pop(tid)
                    c = b.get("content")
                    txt = c if isinstance(c, str) else " ".join(
                        x.get("text", "") for x in c if isinstance(x, dict))
                    results[name].append((approx_tokens(txt), turn))
    return results, turn


def classify(name: str) -> str:
    short = name.split("__")[-1] if name.startswith("mcp__") else name
    if name in RETRIEVAL_TOOLS or short in {"sg_search", "sg_expand", "sg_overview", "sg_get"}:
        return "retrieval"
    if name in EDIT_VERIFY_TOOLS:
        return "edit_verify"
    # competitor MCP tools are all retrieval-side
    if name.startswith("mcp__"):
        return "retrieval"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N tasks per arm (0=all)")
    args = ap.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    tdir = Path(f"eval/results/agent/{args.tag}/_claude_transcripts")

    for arm in arm_names:
        files = sorted(glob.glob(str(tdir / f"*__{arm}__*.jsonl")))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f"\n!! no transcripts for {arm}")
            continue

        raw_by_tool = defaultdict(int)     # plain result size
        carry_by_tool = defaultdict(int)   # size x turns-remaining
        n_tasks = 0
        total_turns = 0
        for f in files:
            results, turns = parse_transcript(Path(f))
            if not results:
                continue
            n_tasks += 1
            total_turns += turns
            for name, entries in results.items():
                for tok, at_turn in entries:
                    raw_by_tool[name] += tok
                    # carry-forward: re-sent on every subsequent assistant turn
                    carry_by_tool[name] += tok * max(1, turns - at_turn + 1)

        print(f"\n{'='*78}\n{arm}  (n={n_tasks} transcripts, avg {total_turns/max(1,n_tasks):.1f} turns)\n{'='*78}")
        tot_raw = sum(raw_by_tool.values()) or 1
        tot_carry = sum(carry_by_tool.values()) or 1
        print(f"{'tool':34} {'raw tok/task':>13} {'carry tok/task':>15} {'carry share':>12}")
        for name in sorted(carry_by_tool, key=lambda k: -carry_by_tool[k]):
            short = name.split("__")[-1] if name.startswith("mcp__") else name
            print(f"  {short:32} {raw_by_tool[name]/n_tasks:>13,.0f} "
                  f"{carry_by_tool[name]/n_tasks:>15,.0f} {carry_by_tool[name]/tot_carry*100:>11.1f}%")

        # phase split
        phase_raw, phase_carry = defaultdict(int), defaultdict(int)
        for name in raw_by_tool:
            p = classify(name)
            phase_raw[p] += raw_by_tool[name]
            phase_carry[p] += carry_by_tool[name]
        print(f"\n  PHASE SPLIT (carry-weighted — the number that matters):")
        for p in ("retrieval", "edit_verify", "other"):
            if phase_carry[p]:
                print(f"    {p:14} {phase_carry[p]/n_tasks:>13,.0f} tok/task  "
                      f"{phase_carry[p]/tot_carry*100:>5.1f}%")
        print(f"\n  => AMDAHL CEILING: even if {arm}'s ENTIRE retrieval surface cost ZERO "
              f"tokens,\n     max achievable reduction = {phase_carry['retrieval']/tot_carry*100:.1f}% "
              f"of tool-result tokens.")


if __name__ == "__main__":
    main()
