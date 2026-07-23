"""Retro-fix hybrid `model_patch` values that were polluted by the
`.hybrid_index/embeddings.npz` binary diff (the bug fixed in this commit's
isolation.py + hybrid.py changes).

The hybrid backend used to cache its embeddings inside the workspace at
<repo>/.hybrid_index/. The baseline commit committed that file; the agent's
final state changed it; `git diff HEAD` captured the binary churn as the
*first* hunk of the patch. SWE-bench's `git apply` rejects binary hunks
without `+/-` chunks → 29/30 NIM v2 hybrid runs erred at verify time.

This script strips the `.hybrid_index/` hunk(s) from every saved hybrid
prediction (a) in the per-task run JSONs and (b) in the bundled
`_predictions_hybrid.jsonl`, so verify.py can re-run without redoing the
~30-task NIM run.

Usage:
    python -m eval.scripts.clean_hybrid_predictions \
        --results-dir eval/results/agent/nim70b_swebench_v2 \
        --arm hybrid

Idempotent — running it twice does nothing on the second pass (the marker
string is already absent).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Matches a complete `diff --git a/<cache>/...` hunk and consumes everything
# up to (but not including) the next `diff --git ` header or EOF.
_HUNK_RE = re.compile(
    r"diff --git a/\.(?:hybrid_index|graphify|bm25_cache|skeletongraph)/"
    r".*?(?=(?:^diff --git )|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _strip(patch: str) -> str:
    """Drop every cache-dir hunk from a unified-diff string."""
    if not patch:
        return patch
    cleaned = _HUNK_RE.sub("", patch)
    # Tidy any leftover blank lines at the very top
    return cleaned.lstrip("\n")


def _clean_run_json(path: Path) -> bool:
    """Update one per-task run JSON in place. Returns True if it changed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    p = data.get("model_patch")
    if not isinstance(p, str) or not p:
        return False
    cleaned = _strip(p)
    if cleaned == p:
        return False
    data["model_patch"] = cleaned
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def _clean_predictions_jsonl(path: Path) -> tuple[int, int]:
    """Re-emit the predictions JSONL in place. Returns (n_changed, n_total)."""
    if not path.exists():
        return (0, 0)
    rows = []
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("model_patch", "")
        cleaned = _strip(p)
        if cleaned != p:
            changed += 1
            r["model_patch"] = cleaned
        rows.append(r)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return (changed, len(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="e.g. eval/results/agent/nim70b_swebench_v2")
    ap.add_argument("--arm", default="hybrid",
                    help="arm to clean (default: hybrid; the only arm hit "
                         "by the .hybrid_index leak, but the regex is "
                         "permissive so other arms are no-ops)")
    args = ap.parse_args()

    rdir = Path(args.results_dir).resolve()
    if not rdir.is_dir():
        raise SystemExit(f"not a directory: {rdir}")

    # 1) Per-task run JSONs
    pat = f"*__{args.arm}__*__r*.json"
    n_changed_runs = 0
    n_total_runs = 0
    for p in sorted(rdir.glob(pat)):
        n_total_runs += 1
        if _clean_run_json(p):
            n_changed_runs += 1
    print(f"[runs]        {n_changed_runs}/{n_total_runs} {args.arm} run JSONs cleaned")

    # 2) Bundled predictions JSONL
    preds = rdir / f"_predictions_{args.arm}.jsonl"
    n_changed_preds, n_total_preds = _clean_predictions_jsonl(preds)
    print(f"[predictions] {n_changed_preds}/{n_total_preds} rows cleaned in {preds.name}")

    if n_changed_runs == 0 and n_changed_preds == 0:
        print("Nothing to clean — predictions are already free of cache hunks.")
    else:
        print("\nNext: re-run verify on the cleaned predictions, e.g.")
        print(f"  python -m eval.agent.verify --stage baseline --run-tag {rdir.name.split('_', 1)[-1] if '_' in rdir.name else rdir.name}")


if __name__ == "__main__":
    main()
