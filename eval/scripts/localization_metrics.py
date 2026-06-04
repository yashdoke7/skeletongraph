"""Re-score COMPLETED agent runs at FILE vs FUNCTION granularity (no repos, no
re-run). Reads the raw `file::symbol` results the model actually saw (stored in
each run's turns) and compares them to gold_fqns.

Why: Agentless error analysis shows file-level localization is ~solved (Hit@1
~85%) while the real failures are wrong-element/-line. We've been scoring FILE
recall (no headroom). SG emits function FQNs — this surfaces where the headroom
(and SG's structural edge) actually is. CodeRAG-Bench-style intrinsic metric.

Function match is fuzzy: same file AND the last name token matches (gold
'header.py::fromstring' matches retrieved 'header.py::Header.fromstring').

    python -m eval.scripts.localization_metrics \
        --runs-dir eval/results/agent/nemotron3_final \
        --dataset C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

_RANK_LINE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
_DEFAULT_DS = Path(r"C:/Users/ASUS/Desktop/CS/Projects/swebench-data/swebench_100_fqn.jsonl")


def _norm(p: str) -> str:
    return p.replace("\\", "/").strip()


def _file_of(fqn: str) -> str:
    return _norm(fqn.split("::", 1)[0])


def _func_of(fqn: str) -> str:
    # last name token: 'Header.fromstring' -> 'fromstring'; 'fromstring' -> 'fromstring'
    tail = fqn.split("::", 1)[-1] if "::" in fqn else ""
    return tail.split(".")[-1].strip() if tail else ""


def _parse_ranked(result: str) -> list:
    """Ordered FQNs from a search_code result block ('N. file::symbol')."""
    if not result or result.startswith(("ERROR", "No results")):
        return []
    out = []
    for line in result.splitlines():
        m = _RANK_LINE.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def _first_search_fqns(rec: dict) -> list:
    for t in rec.get("turns", []):
        for call in t.get("tool_calls", []):
            if call.get("name") == "search_code":
                return _parse_ranked(call.get("result", "") or "")
    return []


def _all_search_fqns(rec: dict) -> list:
    seen, out = set(), []
    for t in rec.get("turns", []):
        for call in t.get("tool_calls", []):
            if call.get("name") == "search_code":
                for fqn in _parse_ranked(call.get("result", "") or ""):
                    if fqn not in seen:
                        seen.add(fqn); out.append(fqn)
    return out


def _func_match(retrieved_fqn: str, gold_fqn: str) -> bool:
    return (_file_of(retrieved_fqn) == _file_of(gold_fqn)
            and _func_of(retrieved_fqn) and _func_of(retrieved_fqn) == _func_of(gold_fqn))


def _recall_rank(retrieved: list, golds: list, match) -> tuple:
    """(recall@10, hit, rank-of-first-gold) for top-10."""
    if not golds:
        return (None, None, None)
    top = retrieved[:10]
    matched = sum(1 for g in golds if any(match(r, g) for r in top))
    rank = 0
    for i, r in enumerate(top, 1):
        if any(match(r, g) for g in golds):
            rank = i; break
    return (matched / len(golds), 1 if matched else 0, rank)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("eval/results/agent/nemotron3_final"))
    ap.add_argument("--dataset", type=Path, default=_DEFAULT_DS)
    args = ap.parse_args()

    gold = {}
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            gold[d["task_id"]] = (d.get("gold_files", []), d.get("gold_fqns", []))
    print(f"dataset: {args.dataset.name}  ({len(gold)} tasks with gold)\n")

    by_arm = defaultdict(list)
    for p in sorted(args.runs_dir.glob("*.json")):
        if p.name.startswith(("_", "summary")):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("arm") and r.get("task_id") in gold:
            by_arm[r["arm"]].append(r)

    def _m(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    print("First-search localization — FILE vs FUNCTION granularity")
    print(f"{'arm':<18}{'n':>4}{'fileR@10':>9}{'fileHit':>8}{'funcR@10':>9}"
          f"{'funcHit':>8}{'funcMRR':>8}{'funcRank':>9}{'cumFuncHit':>11}")
    print("-" * 84)
    rows = []
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        fileR, fileH, funcR, funcH, funcMRR, funcRank, cumH = [], [], [], [], [], [], []
        for r in rs:
            gf, gq = gold[r["task_id"]]
            first = _first_search_fqns(r)
            fr, fh, _ = _recall_rank(first, gf, lambda a, b: _file_of(a) == _file_of(b))
            qr, qh, qk = _recall_rank(first, gq, _func_match)
            fileR.append(fr); fileH.append(fh)
            funcR.append(qr); funcH.append(qh)
            funcMRR.append(1.0 / qk if qk else (0.0 if gq else None))
            if qk:
                funcRank.append(qk)
            # cumulative func hit across ALL searches
            _, ch, _ = _recall_rank(_all_search_fqns(r), gq, _func_match)
            cumH.append(ch)
        rows.append((arm, len(rs), _m(fileR), _m(fileH), _m(funcR), _m(funcH),
                     _m(funcMRR), (st.median(funcRank) if funcRank else float("nan")),
                     _m(cumH)))
    for arm, n, fr, fh, qr, qh, mrr, rk, ch in sorted(rows, key=lambda x: -(x[4] if x[4]==x[4] else -1)):
        print(f"{arm:<18}{n:>4}{fr:>9.2f}{fh*100:>7.0f}%{qr:>9.2f}{qh*100:>7.0f}%"
              f"{mrr:>8.2f}{rk:>9.1f}{ch*100:>10.0f}%")

    print("\nRead: funcR@10 << fileR@10 = right FILE, wrong FUNCTION (the Agentless gap, where the "
          "real headroom is). cumFuncHit = did the agent EVER retrieve the gold function across all "
          "its searches.")


if __name__ == "__main__":
    main()
