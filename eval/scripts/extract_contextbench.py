"""ContextBench → stage-compatible jsonl loader (second benchmark).

ContextBench (EuniAI/ContextBench, arXiv 2602.05892) is a process-oriented
context-retrieval benchmark: ~1,136 issue-resolution tasks over 66 repos / 8
languages, each with HUMAN-ANNOTATED gold contexts (files + functions). It is
the ideal second benchmark — its own headline finding is the consolidation gap
("substantial gaps between explored and utilized context"), exactly SG's C2.

This emits eval/datasets/contextbench.jsonl in the SAME schema as stage0.jsonl
(task_id, repo, base_commit, repo_path, query, gold_files, gold_fqns), after
which every existing command works by pointing a stage / --dataset at it.

    # 1. discover the real field names (RUN THIS FIRST — schema unknown here):
    python -m eval.scripts.extract_contextbench --inspect
    # 2. build (adjust --field-* if --inspect shows different names):
    python -m eval.scripts.extract_contextbench --n 60 --lang python

⚠ The field mapping below is best-effort (the exact ContextBench column names
could not be verified offline). `--inspect` prints the first record's keys so
you can confirm/override via the --field-* flags before a full build.

Source: https://github.com/EuniAI/ContextBench  ·  arXiv 2602.05892
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = EVAL_DIR / "datasets" / "contextbench.jsonl"

# Best-effort field-name candidates (first present wins). Override with --field-*.
_CANDIDATES = {
    "task_id":      ["instance_id", "task_id", "id"],
    "repo":         ["repo", "repository", "repo_name"],
    "base_commit":  ["base_commit", "commit", "sha", "base_sha"],
    "query":        ["problem_statement", "issue", "issue_body", "query", "prompt"],
    # gold context — ContextBench's annotated files/functions
    "gold_files":   ["gold_files", "gold_context_files", "context_files",
                     "annotated_files", "files"],
    "gold_fqns":    ["gold_fqns", "gold_functions", "gold_context_functions",
                     "annotated_functions", "functions", "symbols"],
}


def _first(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in record and record[k] not in (None, "", []):
            return record[k]
    return None


def _as_file_list(val: Any) -> List[str]:
    """Normalise a gold-context value to a flat list of file paths."""
    if val is None:
        return []
    if isinstance(val, str):
        # could be JSON-encoded, newline-, or comma-separated
        s = val.strip()
        if s.startswith("["):
            try:
                return _as_file_list(json.loads(s))
            except json.JSONDecodeError:
                pass
        sep = "\n" if "\n" in s else ","
        return [p.strip() for p in s.split(sep) if p.strip()]
    if isinstance(val, list):
        out: List[str] = []
        for item in val:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # {"file": "...", ...} or {"path": "..."}
                for k in ("file", "path", "file_path", "filename"):
                    if k in item and isinstance(item[k], str):
                        out.append(item[k]); break
        return out
    return []


def _load_contextbench(split: str, source: Optional[str], lang: Optional[str]) -> List[dict]:
    if source:
        # local jsonl/json file
        p = Path(source)
        text = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            data = json.loads(text)
            rows = data if isinstance(data, list) else data.get("data", [])
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: pip install datasets  (or pass --source <local file>)")
            sys.exit(1)
        ds = load_dataset(split)
        # pick the first available split
        split_name = "test" if "test" in ds else list(ds.keys())[0]
        rows = [dict(x) for x in ds[split_name]]
    if lang:
        rows = [r for r in rows
                if str(r.get("language", r.get("lang", ""))).lower() == lang.lower()]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="ContextBench → stage jsonl")
    ap.add_argument("--split", default="EuniAI/ContextBench",
                    help="HF dataset id (verify the exact id on the repo page)")
    ap.add_argument("--source", default=None,
                    help="local ContextBench json/jsonl instead of HF")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--lang", default="python", help="filter to one language")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--inspect", action="store_true",
                    help="print the first record's keys + a sample, then exit")
    # field overrides (use after --inspect if names differ)
    for f in _CANDIDATES:
        ap.add_argument(f"--field-{f.replace('_', '-')}", default=None,
                        help=f"override the source field used for {f}")
    args = ap.parse_args()

    rows = _load_contextbench(args.split, args.source, args.lang)
    if not rows:
        raise SystemExit("no ContextBench records loaded")

    if args.inspect:
        print(f"Loaded {len(rows)} records. First record keys:")
        for k in sorted(rows[0].keys()):
            v = rows[0][k]
            preview = str(v)[:120].replace("\n", " ")
            print(f"  {k:24s} = {preview}")
        print("\nIf these differ from the assumed names, re-run with --field-* "
              "overrides (e.g. --field-gold-files context_files).")
        return

    # resolve field names (override > candidate auto-detect)
    fields = {}
    for f, cands in _CANDIDATES.items():
        override = getattr(args, f"field_{f}")
        fields[f] = [override] if override else cands

    # reuse the SWE-bench repo-clone machinery
    sys.path.insert(0, str(EVAL_DIR))
    from make_dataset import setup_repo  # clones + worktree at base_commit

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_rows: List[dict] = []
    picked = rows[: args.n]
    for i, r in enumerate(picked, 1):
        tid = _first(r, fields["task_id"])
        repo = _first(r, fields["repo"])
        commit = _first(r, fields["base_commit"])
        query = _first(r, fields["query"])
        if not (tid and repo and commit and query):
            print(f"  [{i}/{len(picked)}] skip — missing core fields "
                  f"(id/repo/commit/query). Check --inspect.")
            continue
        gold_files = _as_file_list(_first(r, fields["gold_files"]))
        gold_fqns = _as_file_list(_first(r, fields["gold_fqns"]))
        if not gold_files and gold_fqns:
            gold_files = [g.split("::")[0] for g in gold_fqns]
        if not gold_files:
            print(f"  [{i}/{len(picked)}] {tid}: skip — no gold context")
            continue
        print(f"[{i}/{len(picked)}] {tid}  ({repo})")
        repo_path = setup_repo(repo, commit, str(tid))
        if repo_path is None:
            print("  skip (checkout failed)")
            continue
        out_rows.append({
            "task_id": str(tid), "repo": repo, "base_commit": commit,
            "repo_path": str(repo_path), "query": query,
            "gold_files": gold_files, "gold_fqns": gold_fqns,
        })

    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(out_rows)}/{len(picked)} tasks to {args.out}")
    print("Run retrieval/agent stages with --dataset", args.out)


if __name__ == "__main__":
    main()
