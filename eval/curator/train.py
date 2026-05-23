"""Train the learned curator on a DISJOINT task set (no leakage).

Procedure (offline, CPU, no agent, no Docker):
  1. Load training tasks (same jsonl schema as stage0; MUST be disjoint from the
     eval set — build it from SWE-bench full minus the eval instance_ids).
  2. For each task, run SG retrieval under each candidate mode (mode_hint) and
     score recall@k against the gold files; label the query with the best mode.
  3. Fit TF-IDF + LogisticRegression (query text -> best mode).
  4. Save {vectorizer, clf, labels} to eval/curator/curator_model.pkl.

The `sg-learned` arm then loads this and routes each query to the predicted
mode. If accuracy/coverage is poor, that's a finding too (the rule-based router
is already near-optimal — see docs/CURATOR.md).

    python -m eval.curator.train --train-data eval/datasets/curator_train.jsonl
    python -m eval.curator.train --train-data ... --k 10

Requires: scikit-learn, skeletongraph (+ its embedding stack).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import List

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EVAL_DIR.parent
OUT_MODEL = Path(__file__).resolve().parent / "curator_model.pkl"


def _candidate_modes() -> List[str]:
    """The QueryMode names the router can choose among (the curator's labels)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT))
    from skeletongraph.retrieval.classifier import QueryMode
    return [m.value for m in QueryMode]


def _recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(r.split("::")[0].replace("\\", "/") for r in retrieved[:k])
    hit = sum(1 for g in gold if g.replace("\\", "/") in top)
    return hit / len(gold)


def _best_mode_for(engine, query: str, gold: List[str], modes: List[str],
                   k: int) -> str:
    """Run SG once per candidate mode; return the mode with the best recall@k."""
    best, best_r = modes[0], -1.0
    for mode in modes:
        try:
            res = engine.heuristic_query(query, top_n=max(k * 5, 50),
                                         mode_hint=mode)
            hits = [c.skeleton.fqn for c in res.candidates]
        except Exception:
            hits = []
        r = _recall_at_k(hits, gold, k)
        if r > best_r:
            best, best_r = mode, r
    return best


def _label_tasks(tasks: list, modes: List[str], k: int) -> list:
    from skeletongraph.engine import SGEngine
    rows = []
    for i, t in enumerate(tasks, 1):
        repo = Path(t["repo_path"])
        gold = [g.replace("\\", "/") for g in t.get("gold_files", [])]
        if not gold or not repo.is_dir():
            continue
        try:
            engine = SGEngine(project_root=repo)
            label = _best_mode_for(engine, t["query"], gold, modes, k)
        except Exception as e:
            print(f"  [{i}/{len(tasks)}] {t['task_id']}: skip ({e})")
            continue
        rows.append((t["query"], label))
        print(f"  [{i}/{len(tasks)}] {t['task_id']}: best_mode={label}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True, type=Path,
                    help="jsonl of DISJOINT training tasks (not the eval set)")
    ap.add_argument("--k", type=int, default=10, help="recall@k for labelling")
    ap.add_argument("--out", type=Path, default=OUT_MODEL)
    args = ap.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        raise SystemExit("pip install scikit-learn")

    modes = _candidate_modes()
    print(f"Candidate modes: {modes}")
    tasks = [json.loads(l) for l in
             args.train_data.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Labelling {len(tasks)} disjoint tasks (recall@{args.k}) ...")
    rows = _label_tasks(tasks, modes, args.k)
    if len(rows) < 20:
        print(f"WARNING: only {len(rows)} labelled examples — too few to train a "
              f"reliable curator. Provide more disjoint tasks.")
    if not rows:
        raise SystemExit("no labelled examples")

    queries = [q for q, _ in rows]
    labels = [m for _, m in rows]
    print(f"Label distribution: {dict(Counter(labels))}")

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                          stop_words="english")
    X = vec.fit_transform(queries)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    train_acc = clf.score(X, labels)
    print(f"Train accuracy (in-sample, optimistic): {train_acc:.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({"vectorizer": vec, "clf": clf,
                     "labels": sorted(set(labels))}, f)
    print(f"Wrote {args.out}  ({len(rows)} examples)")
    print("The `sg-learned` arm will now use it. If train_acc is near the "
          "majority-class baseline, the rule-based router is already fine "
          "(report that — it's a valid finding).")


if __name__ == "__main__":
    main()
